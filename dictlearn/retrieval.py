"""Retrieve the dictionary definitions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter, defaultdict

import numpy
import json
import nltk


def vec2str(vector):
    return "".join(map(chr, vector)).strip('\00')


class Vocabulary(object):
    """Class that holds a vocabulary for the dataset."""
    BOS = '<bos>'
    EOS = '<eos>'
    UNK = '<unk>'

    def __init__(self, filename_or_words):
        if isinstance(filename_or_words, str):
            with open(filename_or_words) as f:
                words = [line.strip() for line in f]
        else:
            words = list(filename_or_words)

        self._id_to_word = []
        self._word_to_id = {}
        self._unk = -1
        self._bos = -1
        self._eos = -1

        for idx, word_name in enumerate(words):
            if word_name == Vocabulary.BOS:
                self.bos = idx
            elif word_name == Vocabulary.EOS:
                self.eos = idx
            elif word_name == Vocabulary.UNK:
                self.unk = idx

            self._id_to_word.append(word_name)
            self._word_to_id[word_name] = idx

    @property
    def size(self):
        return len(self._id_to_word)

    def word_to_id(self, word):
        if word in self._word_to_id:
            return self._word_to_id[word]
        return self.unk

    def id_to_word(self, cur_id):
        return self._id_to_word[cur_id]

    def decode(self, cur_ids):
        return ' '.join([self.id_to_word(cur_id) for cur_id in cur_ids])

    def encode(self, sentence):
        word_ids = [self.word_to_id(cur_word) for cur_word in sentence.split()]
        return numpy.array([self.bos] + word_ids + [self.eos],
                            dtype=numpy.int32)

    @staticmethod
    def build(filename, top_k=None):
        # For now let's use a very stupid tokenization
        with open(filename) as file_:
            def data():
                for line in file_:
                    for word in line.strip().split():
                        yield word
            counter = Counter(data())
        # It was not immediately clear to me
        # if counter.most_common() selects consistenly among
        # the words with the same counts. Hence, let's just sort.
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        words = [w for w, _ in count_pairs]
        if top_k:
            words = words[:top_k]
        for special in [Vocabulary.BOS, Vocabulary.EOS, Vocabulary.UNK]:
            if not special in words:
                words.append(special)

        return Vocabulary(words)

    def save(self, filename):
        with open(filename, 'w') as f:
            for word in self._id_to_word:
                print(word, file=f)


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

    def __init__(self, dictionary, vocab):
        self._dictionary = dictionary
        self._vocab = vocab
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
