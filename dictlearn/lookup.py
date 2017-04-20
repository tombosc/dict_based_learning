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
    def apply(self, application_call,
              defs, def_mask):
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

        return def_embeddings


class MeanPoolCombiner(Initializable):

    @application
    def apply(self, application_call,
              word_embs, words_mask,
              def_embeddings, def_map):
        batch_shape = word_embs.shape

        # Mean-pooling of definitions
        def_sum = T.zeros((batch_shape[0] * batch_shape[1], def_embeddings.shape[1]))
        def_lens = T.zeros_like(def_sum[:, 0])
        flat_indices = def_map[:, 0] * batch_shape[1] + def_map[:, 1]
        def_sum = T.inc_subtensor(def_sum[flat_indices],
                                  def_embeddings[def_map[:, 2]])
        def_lens = T.inc_subtensor(def_lens[flat_indices], 1)
        def_mean = def_sum / T.maximum(def_lens[:, None], 1)
        def_mean = def_mean.reshape((batch_shape[0], batch_shape[1], -1))

        application_call.add_auxiliary_variable(
            masked_root_mean_square(def_mean, words_mask), name='def_mean_rootmean2')

        return word_embs + def_mean
