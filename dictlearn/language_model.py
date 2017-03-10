"""A dictionary-equipped language model."""
from blocks.bricks import Initializable
from blocks.bricks.base import application

from dictlearn.ops import WordToIdOp, RetrievalOp

class LanguageModel(Initializable):
    def __init__(self, vocab, dict_):
        self._vocab = vocab
        self._dict = dict_

        self._word_to_id = WordToIdOp(self._vocab)
        self._retrieve = RetrievalOp(self._vocab, self._dict)
        super(LanguageModel, self).__init__()

    @application
    def apply(self, words):
        word_ids = self._word_to_id(words)
        defs, def_map = self._retrieve(words)

        # From here just pseudocode:
        # - embed word_ids
        # - embed definitions (`defs`) by running the LSTM
        # - fetch the right definitions according to `def_map`
        # - compute ranges of definitions corresponding to each of the positions
        # - run the scan with LSTM to compute the states
        # - softmax and the final cost computation


