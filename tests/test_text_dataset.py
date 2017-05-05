import cPickle

from dictlearn.datasets import TextDataset

from tests.util import (
    TEST_TEXT, temporary_content_path)

def test_text_dataset():
    with temporary_content_path(TEST_TEXT) as path:
        dataset = TextDataset(path, 100)
        stream = dataset.get_example_stream()
        it = stream.get_epoch_iterator()

        d = next(it)
        assert d == (['abc', 'abc', 'def'],)
        pickled_it = cPickle.dumps(it)

        d = next(it)
        assert d == (['def', 'def', 'xyz'],)

        it = cPickle.loads(pickled_it)
        d = next(it)
        assert d == (['def', 'def', 'xyz'],)

        d = next(it)
        assert d == (['xyz'],)
