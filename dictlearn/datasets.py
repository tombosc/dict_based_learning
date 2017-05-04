"""Data.

Fuel provides two solutions for reading text, and both don't seem
suitable:

- TextFile only reads sequentially
- H5PyDataset is built with an assumption that there exists
  a non-overlapping set of examples.

That said, we need a different basic solution.

"""

from fuel.datasets import Dataset
from fuel.datasets.hdf5 import H5PYDataset
from fuel.utils import do_not_pickle_attributes
from fuel.transformers import Transformer


class PicklableFile(object):
    """
    Iterable file handler. WARNING: empty line raise EOF
    """

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._file = open(*args, **kwargs)

    def __getstate__(self):
        state = dict(self.__dict__)
        if hasattr(self, '_file'):
            state['_pos'] = self._file.tell()
            del state['_file']
        return state

    def __setstate__(self, state):
        state['_file'] = open(*state['_args'], **state['_kwargs'])
        if hasattr(self, '_pos'):
            state['_file'].seek(state['_pos'])
            del state['_pos']
        self.__dict__.update(state)

    def read(self, *args):
        #TODO(tombosc): remove: useless?
        return self._file.read(*args)

    def __iter__(self):
        return self

    def next(self):
        # note, that we can't use next(self._file) because this
        # will trigger the use of a lookahead buffer, and
        # self._file.tell() will become unreliable
        l = self._file.readline()
        if l == "":
            raise StopIteration
        return l


class TextDataset(Dataset):
    """
    Provides basic access to lines of a text file.
    WARNING: text file shouldn't have any empty lines
    """
    provides_sources = ('words',)
    example_iteration_scheme = None

    def __init__(self, path, max_length, **kwargs):
        """
        max_length: max sentence length
        """
        self._path = path
        if not max_length:
            max_length = 100000
        self._max_length = max_length
        super(TextDataset, self).__init__(**kwargs)

    def open(self):
        return PicklableFile(self._path, 'r')

    def get_data(self, state, request=None):
        return (next(state).strip().split()[:self._max_length],)


class PutTextTransfomer(Transformer):

    def __init__(self, data_stream, dataset, raw_text=False, **kwargs):
        super(PutTextTransfomer, self).__init__(data_stream, **kwargs)
        self.produces_examples = data_stream.produces_examples
        self._dataset = dataset
        self._raw_text = raw_text

    @property
    def _text(self):
        """Not making this a member to avoid pickling it."""
        return self._dataset.text if self._raw_text else self._dataset.text_ids

    def transform_example(self, example):
        c_pos = self.sources.index('contexts')
        q_pos = self.sources.index('questions')
        example = list(example)
        example[c_pos] = self._text[example[c_pos][0]:example[c_pos][1]]
        example[q_pos] = self._text[example[q_pos][0]:example[q_pos][1]]
        return tuple(example)

    def transform_batch(self, batch):
        return (self.transform_example(example) for example in batch)


@do_not_pickle_attributes('text', 'text_ids')
class SQuADDataset(H5PYDataset):
    """Adds default transformers."""
    def __init__(self, *args, **kwargs):
        super(SQuADDataset, self).__init__(*args, **kwargs)
        self.default_transformers = [(PutTextTransfomer, [self], {},)]

    def load(self):
        super(SQuADDataset, self).load()
        self._out_of_memory_open()
        self.text = self._file_handle['text'][:]
        self.text_ids = self._file_handle['text_ids'][:]
        self._out_of_memory_close()
