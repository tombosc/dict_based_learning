from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import base64

from fuel.datasets import IndexableDataset
from fuel.streams import DataStream

from dictlearn.data import (
    LanguageModellingData, ExtractiveQAData, RandomSpanScheme)
from dictlearn.vocab import Vocabulary
from dictlearn.util import vec2str

from tests.util import TEST_TEXT, TEST_SQUAD_BASE64_HDF5_DATA

def test_languge_modelling_data():
    temp_dir = tempfile.mkdtemp()
    train_path = os.path.join(temp_dir, "train.txt")
    with open(train_path, 'w') as dst:
        print(TEST_TEXT, file=dst)

    data = LanguageModellingData(temp_dir, 'standard')

    # test without batches
    stream = data.get_stream('train')
    it = stream.get_epoch_iterator()
    example = next(it)
    # skip one
    example = next(it)
    assert len(example) == 1
    assert len(example[0]) == 4
    assert example[0][0][:5].tolist() == map(ord, '<bos>')
    assert example[0][1][:5].tolist() == [ord('d'), ord('e'), ord('f'), 0, 0]
    assert example[0][2][:5].tolist() == [ord('d'), ord('e'), ord('f'), 0, 0]
    assert example[0][3][:5].tolist() == [ord('x'), ord('y'), ord('z'), 0, 0]

    # test with batches
    stream = data.get_stream('train', batch_size=2)
    it = stream.get_epoch_iterator()
    example = next(it)
    # skip one
    example = next(it)
    assert len(example) == 2
    assert example[0].shape == (2, 4, 100)
    assert example[1].shape == (2, 4)
    assert example[0][1, 2, :5].tolist() == [ord('d'), ord('e'), ord('f'), 0, 0]
    assert example[1].tolist() == [[1., 1., 0., 0.], [1., 1., 1., 1.]]

    os.remove(train_path)
    os.rmdir(temp_dir)


def test_squad_data():
    temp_dir = tempfile.mkdtemp()
    train_path = os.path.join(temp_dir, 'train.h5')
    with open(train_path, 'wb') as dst:
        print(base64.b64decode(TEST_SQUAD_BASE64_HDF5_DATA), file=dst)

    data = ExtractiveQAData(path=temp_dir, layout='squad')
    stream = data.get_stream('train', batch_size=3, shuffle=True, seed=3)
    assert set(stream.sources) == set(['contexts', 'contexts_mask',
                                       'questions', 'questions_mask',
                                       'answer_begins',
                                       'answer_ends'])
    batch = next(stream.get_epoch_iterator(as_dict=True))
    assert batch['contexts'].ndim == 2
    assert batch['contexts_mask'].ndim == 2
    assert batch['questions'].ndim == 2
    assert batch['questions_mask'].ndim == 2
    assert batch['answer_begins'].tolist() == [45, 78, 117]
    assert batch['answer_ends'].tolist() == [46, 80, 118]

    longest = batch['contexts_mask'].sum(axis=1).argmax()
    assert batch['contexts'][longest][-1] == data.vocab.eos


def test_random_span_scheme():
    scheme = RandomSpanScheme(10000, 100, 1)
    req_it = scheme.get_request_iterator()
    assert next(req_it) == slice(235, 335, None)

    dataset = IndexableDataset(['abc', 'def', 'xyz', 'ter'])
    scheme = RandomSpanScheme(4, 2, 1)
    stream = DataStream(dataset, iteration_scheme=scheme)
    it = stream.get_epoch_iterator()
    assert next(it) == (['def', 'xyz'],)
