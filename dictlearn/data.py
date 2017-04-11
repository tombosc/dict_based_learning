"""Dataset layout and data preparation.

Currently the following layouts are supported:

- standard
    The training, validation and development data are in
    train.txt, valid.txt and test.txt. All files are read
    sequentially.
- lambada
    Like standard, but the training data is stored in an
    HDF5 file "train.h5". The training data is read randomly
    by taking random spans.

"""

import os
import functools

import numpy

import fuel
from fuel.transformers import Mapping, Batch, Padding, AgnosticSourcewiseTransformer
from fuel.schemes import IterationScheme, ConstantScheme, ShuffledExampleScheme
from fuel.streams import DataStream
from fuel.datasets import H5PYDataset

from dictlearn.vocab import Vocabulary
from dictlearn.datasets import TextDataset, SQuADDataset
from dictlearn.util import str2vec

# We have to pad all the words to contain the same
# number of characters.
MAX_NUM_CHARACTERS = 100


def vectorize(words):
    """Replaces words with vectors."""
    return [str2vec(word, MAX_NUM_CHARACTERS) for word in words]


def listify(example):
    return tuple(list(source) for source in example)


def add_bos(bos, example):
    return tuple([bos] + source for source in example)


class SourcewiseMapping(AgnosticSourcewiseTransformer):
    def __init__(self, data_stream, mapping, *args, **kwargs):
        kwargs.setdefault('which_sources', data_stream.sources)
        super(SourcewiseMapping, self).__init__(
            data_stream, data_stream.produces_examples, *args, **kwargs)
        self._mapping = mapping

    def transform_any_source(self, source_data, _):
        return self._mapping(source_data)


class RandomSpanScheme(IterationScheme):
    requests_examples = True

    def __init__(self, dataset_size, span_size, seed=None):
        self._dataset_size = dataset_size
        self._span_size = span_size
        if not seed:
            seed = fuel.config.default_seed
        self._rng = numpy.random.RandomState(seed)

    def get_request_iterator(self):
        # As for now this scheme produces an infinite stateless scheme,
        # it can itself play the role of an iterator. If we want to add
        # a state later, this trick will not cut it any more.
        return self

    def __iter__(self):
        return self

    def next(self):
        start = self._rng.randint(0, self._dataset_size - self._span_size)
        return slice(start, start + self._span_size)


class Data(object):
    """Builds the data stream for different parts of the data."""
    def __init__(self, path, layout):
        self._path = path
        self._layout = layout
        if not self._layout in ['standard', 'lambada', 'squad']:
            raise "layout {} is not supported".format(self._layout)

        self._vocab = None
        self._dataset_cache = {}

    @property
    def vocab(self):
        if not self._vocab:
            self._vocab = Vocabulary(
                os.path.join(self._path, "vocab.txt"))
        return self._vocab

    def get_dataset(self, part):
        if not part in self._dataset_cache:
            if self._layout == 'standard':
                part_map = {'train': 'train.txt',
                            'valid': 'valid.txt',
                            'test': 'test.txt'}
            elif self._layout == 'lambada':
                part_map = {'train' : 'train.h5',
                            'valid' : 'lambada_development_plain_text.txt',
                            'test' : 'lambada_test_plain_text.txt'}
            elif self._layout == 'squad':
                part_map = {'train' : 'train.h5',
                            'valid' : 'valid.h5'}
            part_path = os.path.join(self._path, part_map[part])
            if self._layout == 'lambada' and part == 'train':
                self._dataset_cache[part] = H5PYDataset(part_path, ('train',))
            elif self._layout == 'squad':
                self._dataset_cache[part] = SQuADDataset(part_path, ('all',))
            else:
                self._dataset_cache[part] = TextDataset(part_path)
        return self._dataset_cache[part]

    def get_stream(self, *args, **kwargs):
        raise NotImplementedError()


class LanguageModellingData(Data):
    def get_stream(self, part, batch_size=None, max_length=None, seed=None):
        dataset = self.get_dataset(part)
        if self._layout == 'lambada' and part == 'train':
            stream = DataStream(
                dataset,
                iteration_scheme=RandomSpanScheme(
                    dataset.num_examples, max_length, seed))
            stream = Mapping(stream, listify)
        else:
            stream = dataset.get_example_stream()

        stream = Mapping(stream, functools.partial(add_bos, Vocabulary.BOS))
        stream = SourcewiseMapping(stream, vectorize)
        if not batch_size:
            return stream
        stream = Batch(
            stream,
            iteration_scheme=ConstantScheme(batch_size))
        stream = Padding(stream)
        return stream


def select_random_answer(rng, example):
    index = rng.randint(0, len(example['answer_begins']))
    example['answer_begins'] = example['answer_begins'][index]
    example['answer_ends'] = example['answer_ends'][index]
    return example


class ExtractiveQAData(Data):
    def get_stream(self, part, batch_size=None, shuffle=False, max_length=None, seed=None):
        if not seed:
            seed = fuel.config.default_seed
        rng = numpy.random.RandomState(seed)
        dataset = self.get_dataset(part)
        if shuffle:
            stream = DataStream(
                dataset,
                iteration_scheme=ShuffledExampleScheme(dataset.num_examples, rng=rng))
        else:
            stream = dataset.get_example_stream()
        stream = dataset.apply_default_transformers(stream)
        stream = Mapping(stream, functools.partial(select_random_answer, rng),
                         mapping_accepts=dict)
        stream = SourcewiseMapping(stream, vectorize, which_sources=('contexts', 'questions'))
        if not batch_size:
            return stream
        stream = Batch(stream, iteration_scheme=ConstantScheme(batch_size))
        stream = Padding(stream, mask_sources=('contexts', 'questions'))
        return stream
