"""
Methods of constructing word embeddings

TODO(kudkudak): Add multiplicative compose_type
"""
from blocks.bricks import Initializable, Linear, MLP, Tanh, Rectifier
from blocks.bricks.base import application, _variable_name
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
    def __init__(self, num_input_words, emb_dim, dim, vocab, **kwargs):
        self._num_input_words = num_input_words
        self._vocab = vocab

        self._def_lookup = LookupTable(self._num_input_words, emb_dim, name='def_lookup')
        self._def_fork = Linear(emb_dim, 4 * dim, name='def_fork')
        self._def_rnn = LSTM(dim, name='def_rnn')
        children = [self._def_lookup, self._def_fork, self._def_rnn]

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

        return def_embeddings


class MeanPoolCombiner(Initializable):
    """
    Parameters
    ----------
    dim: int

    dropout_type: str, default: "regular"
        "regular": applies regular dropout to both word and def emb

        "multimodal": If set applies "multimodal" dropout, i.e. drops at once whole word and *independentyl*
        whole def emb.

        Note: It does it independently for simplicity (otherwise inference would require some extra
        training). Maybe we could start with the dependent dropout and then phase into
        independent one. Could work, but adds complexity.

    dropout: float, defaut: 0.0

    emb_dim: int

    compose_type : str
        If 'sum', the definition and word embeddings are averaged
        If 'fully_connected_linear', a learned perceptron compose the 2
        embeddings linearly
        If 'fully_connected_relu', ...
        If 'fully_connected_tanh', ...
    """

    def __init__(self, emb_dim, dim, dropout=0.0, dropout_type="regular", compose_type="sum", **kwargs):
        self._dropout = dropout
        self._dropout_type = dropout_type
        self._compose_type = compose_type

        if dropout_type not in {"regular", "multimodal"}:
            raise NotImplementedError()

        children = []

        if compose_type == 'fully_connected_tanh':
            self._def_state_compose = MLP(activations=[Tanh(name="def_state_compose")], dims=[emb_dim + dim, dim]
                , weights_init=GlorotUniform(), biases_init=Constant(0))
            children.append(self._def_state_compose)
        elif compose_type == 'fully_connected_relu':
            self._def_state_compose = MLP(activations=[Rectifier(name="def_state_compose")],
                dims=[emb_dim + dim, dim], weights_init=GlorotUniform(), biases_init=Constant(0))
            children.append(self._def_state_compose)
        elif compose_type == 'fully_connected_linear':
            self._def_state_compose = MLP(activations=[None],
                dims=[emb_dim + dim, dim], weights_init=GlorotUniform(), biases_init=Constant(0))
            children.append(self._def_state_compose)
        elif compose_type == 'fully_connected_linear':
            self._def_state_compose = Linear(emb_dim + dim, dim, weights_init=GlorotUniform(), biases_init=Constant(0))
            children.append(self._def_state_compose)
        elif compose_type == 'sum':
            if not emb_dim == dim:
                raise ValueError("Embedding has different dim! Cannot use compose_type='sum'")
        else:
            raise NotImplementedError()

        super(MeanPoolCombiner, self).__init__(children=children, **kwargs)

    def get_cg_transforms(self):
        return self._cg_transforms

    @application
    def apply(self, application_call,
              word_embs, words_mask,
              def_embeddings, def_map, call_name=""):
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
            masked_root_mean_square(def_mean, words_mask), name=call_name + '_def_mean_rootmean2')

        self._cg_transforms = []
        if self._dropout != 0.0 and self._dropout_type == "regular":
            logger.info("Adding drop on dict and normal emb")
            self._cg_transforms.append(['dropout', self._dropout, word_embs])
            self._cg_transforms.append(['dropout', self._dropout, def_mean])
        elif self._dropout != 0.0 and self._dropout_type == "multimodal":
            logger.info("Adding multimod drop on dict and normal emb")
            # We dropout mask
            mask_defs = T.ones((batch_shape[0],))
            mask_we = T.ones((batch_shape[0],))

            # Mask dropout
            self._cg_transforms.append(['dropout', self._dropout, mask_defs])
            self._cg_transforms.append(['dropout', self._dropout, mask_we])

            # this reduces variance. If both 0 will select both. Classy
            where_both_zero = T.eq((mask_defs + mask_we), 0)

            mask_defs = (where_both_zero + mask_defs).dimshuffle(0, "x", "x")
            mask_we = (where_both_zero + mask_we).dimshuffle(0, "x", "x")

            def_mean = mask_defs * def_mean
            word_embs = mask_we * word_embs

        if self._compose_type == 'sum':
            final_embeddings = word_embs + def_mean
        elif self._compose_type.startswith('fully_connected'):
            concat = T.concatenate([word_embs, def_mean], axis=2)
            final_embeddings = self._def_state_compose.apply(concat)
        else:
            raise NotImplementedError()

        application_call.add_auxiliary_variable(
            masked_root_mean_square(final_embeddings, words_mask),
            name=call_name + '_merged_input_rootmean2')

        application_call.add_auxiliary_variable(
            def_mean.copy(),
            name=call_name + '_dict_word_embeddings')

        application_call.add_auxiliary_variable(
            word_embs.copy(),
            name=call_name + '_word_embeddings')

        return final_embeddings

