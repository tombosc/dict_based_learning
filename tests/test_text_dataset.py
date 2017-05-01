import cPickle

from dictlearn.datasets import TextDataset

from tests.util import (
    TEST_TEXT, temporary_content_path)

def test_text_dataset():
    with temporary_content_path(TEST_TEXT) as path:
        dataset = TextDataset(path)
        stream = dataset.get_example_stream()
        it = stream.get_epoch_iterator()
        assert next(it) == (['abc', 'abc', 'def'],)
        it = cPickle.loads(cPickle.dumps(it))
        assert next(it) == (['def', 'def', 'xyz'],)
        assert next(it) == (['xyz'],)
