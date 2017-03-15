from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter

import numpy

class Vocabulary(object):
    """Class that holds a vocabulary for the dataset."""
    BOS = '<bos>'
    EOS = '<eos>'
    UNK = '<unk>'

    def __init__(self, filename_or_words, top_k=None):
        """Initialize the vocabulary.

        filename_or_words
            Either a list of words or a path to the file with them.
        top_k
            If not `None`, only the first `top_k` entries will be left.
            Note, this does not include the special tokens.

        """
        if isinstance(filename_or_words, str):
            with open(filename_or_words) as f:
                words = [line.strip() for line in f]
        else:
            words = list(filename_or_words)

        self._id_to_word = []
        self._word_to_id = {}
        self.unk = -1
        self.bos = -1
        self.eos = -1

        n_regular_tokens = 0
        for idx, word_name in enumerate(words):
            if top_k and n_regular_tokens == top_k:
                break

            if word_name == Vocabulary.BOS:
                self.bos = idx
            elif word_name == Vocabulary.EOS:
                self.eos = idx
            elif word_name == Vocabulary.UNK:
                self.unk = idx
            else:
                n_regular_tokens += 1

            self._id_to_word.append(word_name)
            self._word_to_id[word_name] = idx

        if -1 in [self.unk, self.bos, self.eos]:
            raise ValueError("special token not found in the vocabulary")

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
        words = [Vocabulary.BOS, Vocabulary.EOS, Vocabulary.UNK] + words

        return Vocabulary(words)

    def save(self, filename):
        with open(filename, 'w') as f:
            for word in self._id_to_word:
                print(word, file=f)
