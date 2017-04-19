"""
Methods of constructing word embeddings

TODO(kudkudak): Add unit test for it
TODO(kudkudak): Refactor (non-dict) lookup from snli_baseline_model to a class here (name for instance EnchancedLookup)
TODO(kudkudak): Add multiplicative compose_type

TODO(kudkudak): Dict fetching also as Fuel stream (would put it on other thread then)
"""
from blocks.bricks import Initializable, Linear, MLP, Tanh, Rectifier
from blocks.bricks.base import application
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import LSTM
from blocks.initialization import Uniform, Constant

import theano
import theano.tensor as T

from dictlearn.inits import GlorotUniform
from dictlearn.util import masked_root_mean_square
from dictlearn.ops import RetrievalOp

import logging
logger = logging.getLogger(__file__)


# TODO: Implement
class LookupWithTranslation(Initializable):
    pass


class ReadDefinitions(Initializable):
    """
    Converts definition into embeddings.

    Parameters
    ----------
    num_input_words: int, default: -1
        If non zero will (a bit confusing name) restrict dynamically vocab.
        WARNING: it assumes word ids are monotonical with frequency!

    dim : int
        Dimensionality of the def rnn.

    emb_dim : int
        Dimensionality of word embeddings

    """
    def __init__(self, num_input_words, emb_dim, dim,
                 vocab=None, retrieval=None, **kwargs):
        self._num_input_words = num_input_words
        self._retrieval = retrieval
        self._vocab = vocab

        if self._retrieval:
            self._retrieve = RetrievalOp(retrieval)

        children = []

        self._def_lookup = LookupTable(self._num_input_words, dim, name='def_lookup',
            weights_init=GlorotUniform(), biases_init=Constant(0))
        # NOTE: It also has the translate layer inside
        self._def_fork = Linear(emb_dim, 4 * dim, name='def_fork',
            weights_init=GlorotUniform(), biases_init=Constant(0))
        # TODO(kudkudak): Better LSTM weight init
        self._def_rnn = LSTM(dim, name='def_rnn',
                             weights_init=Uniform(width=0.1), biases_init=Constant(0))
        children.extend([self._def_lookup, self._def_fork, self._def_rnn])

        super(ReadDefinitions, self).__init__(children=children, **kwargs)

    @application
    def apply(self, application_call, words, words_mask):
        """
        Returns vector per each word in sequence using the dictionary based lookup
        """
        defs, def_mask, def_map = self._retrieve(words)
        return self.apply_with_given_defs(words, words_mask,
                                          defs, def_mask, def_map)


    @application
    def apply_with_given_defs(self, application_call,
                              words, words_mask,
                              defs, def_mask, def_map):
        """
        Returns vector per each word in sequence using the dictionary based lookup
        """
        # Short listing
        defs = (T.lt(defs, self._num_input_words) * defs
                + T.ge(defs, self._num_input_words) * self._vocab.unk)

        embedded_def_words = self._def_lookup.apply(defs)
        def_embeddings = self._def_rnn.apply(
            T.transpose(self._def_fork.apply(embedded_def_words), (1, 0, 2)),
            mask=def_mask.T)[0][-1]
        # Reorder and copy embeddings so that the embeddings of all the definitions
        # that correspond to a position in the text form a continuous span of a T
        def_embeddings = def_embeddings[def_map[:, 2]]

        # Compute the spans corresponding to text positions
        num_defs = T.zeros((1 + words.shape[0] * words.shape[1],), dtype='int32')
        num_defs = T.inc_subtensor(
            num_defs[1 + def_map[:, 0] * words.shape[1] + def_map[:, 1]], 1)
        cum_sum_defs = T.cumsum(num_defs)
        def_spans = T.concatenate([cum_sum_defs[:-1, None], cum_sum_defs[1:, None]],
            axis=1)
        application_call.add_auxiliary_variable(def_spans, name='def_spans')
        return def_embeddings, def_spans


class MeanPoolCombiner(Initializable):

    @application
    def mean_pool_and_sum(self, application_call,
                          def_embeddings, def_spans, def_map,
                          word_embs, words_mask):
        batch_shape = word_embs.shape

        # Mean-pooling of definitions
        def_sum = T.zeros((batch_shape[0] * batch_shape[1], def_embeddings.shape[1]))
        def_sum = T.inc_subtensor(
            def_sum[def_map[:, 0] * batch_shape[1] + def_map[:, 1]],
            def_embeddings)
        def_lens = (def_spans[:, 1] - def_spans[:, 0]).astype(theano.config.floatX)
        def_mean = def_sum / T.maximum(def_lens[:, None], 1)
        def_mean = def_mean.reshape((batch_shape[0], batch_shape[1], -1))

        application_call.add_auxiliary_variable(
            masked_root_mean_square(def_mean, words_mask), name='def_mean_rootmean2')

        return word_embs + def_mean
