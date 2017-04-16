# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import h5py

from fuel.datasets.hdf5 import H5PYDataset

from dictlearn.corenlp import start_corenlp
from dictlearn.h5py_conversion import (
    text_to_h5py_dataset, squad_to_h5py_dataset, add_words_ids_to_squad)
from dictlearn.datasets import SQuADDataset
from dictlearn.vocab import Vocabulary
from dictlearn.util import get_free_port

from tests.util import TEST_SQUAD_RAW_DATA

def test_text_to_h5py_dataset():
    test_dir = tempfile.mkdtemp()
    text_path = os.path.join(test_dir, 'text.txt')
    h5_path = os.path.join(test_dir, 'words.h5')
    with open(os.path.join(test_dir, 'text.txt'), 'w') as dst:
        print('abc', file=dst)
        print('été', file=dst)
        print('abc Δίας', file=dst)
    text_to_h5py_dataset(text_path, h5_path)

    f = H5PYDataset(h5_path, ('train',))
    it = f.get_example_stream().get_epoch_iterator()
    assert next(it)[0] == 'abc'
    assert next(it)[0] == 'été'
    assert next(it)[0] == 'abc'
    assert next(it)[0] == 'Δίας'

    os.remove(text_path)
    os.remove(h5_path)
    os.rmdir(test_dir)


def test_squad_to_h5py_dataset():
    corenlp = None
    try:
        port = get_free_port()
        corenlp = start_corenlp(port)

        test_dir = tempfile.mkdtemp()
        json_path = os.path.join(test_dir, 'data.json')
        h5_path = os.path.join(test_dir, 'data.h5')
        with open(json_path, 'w') as json_file:
            print(TEST_SQUAD_RAW_DATA, file=json_file)
        squad_to_h5py_dataset(json_path, h5_path, "http://localhost:{}".format(port))
        with h5py.File(h5_path, 'r') as h5_file:
            vocab = Vocabulary.build(h5_file['text'], top_k=100)
        add_words_ids_to_squad(h5_path, vocab)

        dataset = SQuADDataset(h5_path, ('all',))
        stream = dataset.get_example_stream()
        stream = dataset.apply_default_transformers(stream)
        example = next(stream.get_epoch_iterator(as_dict=True))
        answer_span = slice(example['answer_begins'][0], example['answer_ends'][0])
        assert example['questions'].tolist() == map(vocab.word_to_id, [
            u'To', u'whom', u'did', u'the', u'Virgin', u'Mary',
            u'allegedly', u'appear', u'in', u'1858', u'in', u'Lourdes',
            u'France', u'?'])
        assert example['contexts'][answer_span].tolist() == map(vocab.word_to_id,
            [u'Saint', u'Bernadette', u'Soubirous'])
    finally:
        if corenlp and corenlp.returncode is None:
            corenlp.kill()
