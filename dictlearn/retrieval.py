"""Retrieve the dictionary definitions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter, defaultdict

import numpy
import json
import nltk

from dictlearn.util import vec2str


class Dictionary(object):
    """The dictionary of definitions.

    The native representation of the dictionary is a mapping from a word
    to a list of definitions, each of which is a sequence of words. All
    the words are stored as strings.

    TODO: ask Wordnik for help when queries for an unknown word

    """
    def __init__(self, path=None):
        self._data = None
        if path:
            self.load(path)

    def load(self, path):
        with open(path, 'r') as src:
            self._data = json.load(src)

    def get_definitions(self, key):
        return self._data[key]


class Retrieval(object):

    def __init__(self, vocab, dictionary):
        self._vocab = vocab
        self._dictionary = dictionary
        self._stemmer = nltk.PorterStemmer()

        # Preprocess all the definitions to see token ids instead of chars

    def retrieve(self, batch):
        """Retrieves all definitions for a batch of words sequences.

        TODO: definitions of phrases, phrasal verbs, etc.

        """
        definitions = []
        stem_def_indices = defaultdict(list)
        def_map = []

        for seq_pos, sequence in enumerate(batch):
            for word_pos, word in enumerate(sequence):
                if isinstance(word, numpy.ndarray):
                    word = vec2str(word)
                stem  = self._stemmer.stem(word)
                if stem not in stem_def_indices:
                    # The first time a stem is encountered in a batch
                    stem_defs = self._dictionary.get_definitions(stem)
                    if not stem_defs:
                        # No defition for this stem
                        continue
                    for i, def_ in enumerate(stem_defs):
                        def_ = [self._vocab.word_to_id(token) for token in def_]
                        stem_def_indices[stem].append(len(definitions))
                        definitions.append(def_)
                for def_index in stem_def_indices[stem]:
                    def_map.append((seq_pos, word_pos, def_index))

        return definitions, def_map
