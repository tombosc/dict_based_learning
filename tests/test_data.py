from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from dictlearn.data import Data
from tests.util import TEST_TEXT

def test_data():
    temp_dir = tempfile.mkdtemp()
    train_path = os.path.join(temp_dir, "train.txt")
    with open(train_path, 'w') as dst:
        print(TEST_TEXT, file=dst)

    data = Data(temp_dir, 'standard')

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
