from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

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
