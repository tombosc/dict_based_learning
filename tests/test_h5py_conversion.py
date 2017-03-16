# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from fuel.datasets.hdf5 import H5PYDataset

from dictlearn.h5py_conversion import text_to_h5py_dataset

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
