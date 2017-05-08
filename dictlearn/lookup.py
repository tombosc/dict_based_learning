"""
Methods of constructing word embeddings

Idea: Learn to pick by gating that's learnable (and would ignore unknown words)
Idea: Then start tarining gate late?

"""
from blocks.bricks import Initializable, Linear, MLP, Tanh, Rectifier
from blocks.bricks.base import application, _variable_name
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import LSTM
from blocks.bricks.simple import Softmax
from blocks.bricks.bn import BatchNormalization
from blocks.initialization import Uniform, Constant
from blocks.bricks import Softmax, Rectifier, Logistic

import theano
import theano.tensor as T

from dictlearn.inits import GlorotUniform
from dictlearn.util import masked_root_mean_square
from dictlearn.theano_util import apply_dropout, unk_ratio
from dictlearn.ops import RetrievalOp

import logging
logger = logging.getLogger(__file__)


class LSTMReadDefinitions(Initializable):
    """
    Converts definition into embeddings.

    Parameters
    ----------
    num_input_words: int, default: -1
        If non zero will (a bit confusing name) restrict dynamically vocab.
        WARNING: it assumes word ids are monotonical with frequency!

    emb_dim : int
        Dimensionality of word embeddings

    dim : int
        Dimensionality of the def rnn.

    lookup: None or LookupTable

    fork_and_rnn: None or tuple (Linear, RNN)
    """
    def __init__(self, num_input_words, emb_dim, dim, vocab, lookup=None, fork_and_rnn=None, **kwargs):

        if num_input_words > 0:
            logger.info("Restricting def vocab to " + str(num_input_words))
            self._num_input_words = num_input_words
        else:
            self._num_input_words = vocab.size()

        self._vocab = vocab

        children = []

        if lookup is None:
            self._def_lookup = LookupTable(self._num_input_words, emb_dim, name='def_lookup')
        else:
            self._def_lookup = lookup

        if fork_and_rnn is None:
            self._def_fork = Linear(emb_dim, 4 * dim, name='def_fork')
            self._def_rnn = LSTM(dim, name='def_rnn')
        else:
            self._def_fork, self._def_rnn = fork_and_rnn

        children.extend([self._def_lookup, self._def_fork, self._def_rnn])

        super(LSTMReadDefinitions, self).__init__(children=children, **kwargs)


    @application
    def apply(self, application_call,
              defs, def_mask):
        """
        Returns vector per each word in sequence using the dictionary based lookup
        """
        # Short listing
        defs = (T.lt(defs, self._num_input_words) * defs
                + T.ge(defs, self._num_input_words) * self._vocab.unk)
        application_call.add_auxiliary_variable(
            unk_ratio(defs, def_mask, self._vocab.unk),
            name='def_unk_ratio')

        embedded_def_words = self._def_lookup.apply(defs)
        def_embeddings = self._def_rnn.apply(
            T.transpose(self._def_fork.apply(embedded_def_words), (1, 0, 2)),
            mask=def_mask.T)[0][-1]

        return def_embeddings


class MeanPoolReadDefinitions(Initializable):
    """
    Converts definition into embeddings using simple sum + translation

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
    def __init__(self, num_input_words, emb_dim, dim, vocab,
                 lookup=None, translate=True, normalize=True, **kwargs):

        if num_input_words > 0:
            logger.info("Restricting def vocab to " + str(num_input_words))
            self._num_input_words = num_input_words
        else:
            self._num_input_words = vocab.size()

        self._vocab = vocab
        self._translate = translate
        self._normalize = normalize

        children = []

        if lookup is None:
            logger.info("emb_dim={}".format(emb_dim))
            self._def_lookup = LookupTable(self._num_input_words, emb_dim, name='def_lookup')
        else:
            self._def_lookup = lookup

        # Makes sense for shared lookup. Then we precondition embeddings.
        # Doesn't makes otherwise (WH = W')
        # TODO(kudkudak): Refactor redundant translate parameter
        if self._translate:
            if emb_dim == dim:
                raise Exception("Redundant layer")

            self._def_translate = Linear(emb_dim, dim, name='def_translate')
            children.extend([self._def_translate])
        else:
            if emb_dim != dim:
                raise Exception("Please pass translate=True if emb_dim != dim")

        children.append(self._def_lookup)

        super(MeanPoolReadDefinitions, self).__init__(children=children, **kwargs)


    @application
    def apply(self, application_call,
              defs, def_mask):
        """
        Returns vector per each word in sequence using the dictionary based lookup
        """
        # Short listing
        defs = (T.lt(defs, self._num_input_words) * defs
                + T.ge(defs, self._num_input_words) * self._vocab.unk)
        # Memory bottleneck:
        # For instance (16101,52,300) ~= 32GB.
        # [(16786, 52, 1), (16786, 52, 100)]
        # TODO: Measure memory consumption here and check if it is in sensible range
        # or maybe introduce some control in Retrieval?
        defs_emb = self._def_lookup.apply(defs)
        application_call.add_auxiliary_variable(
            unk_ratio(defs, def_mask, self._vocab.unk),
            name='def_unk_ratio')

        if self._translate:
            logger.info("Translating in MeanPoolReadDefinitions")
            # Translate. Crucial for recovering useful information from embeddings
            defs_emb = self._def_translate.apply(defs_emb)

        def_emb_mask = def_mask[:, :, None]
        defs_emb = (def_emb_mask * defs_emb).sum(axis=1)
        if self._normalize:
            defs_emb = defs_emb / def_emb_mask.sum(axis=1)

        return defs_emb


class MeanPoolCombiner(Initializable):
    """
    Parameters
    ----------
    dim: int
        Dimensionality of def embedding

    dropout_type: str

    dropout: float, defaut: 0.0

    emb_dim: int
        Dimensionality of word embeddings, as well as final output

    compose_type : str
        If 'sum', the definition and word embeddings are averaged
        If 'fully_connected_linear', a learned perceptron compose the 2
        embeddings linearly
        If 'fully_connected_relu', ...
        If 'fully_connected_tanh', ...
    """

    def __init__(self, emb_dim, dim, dropout=0.0,
            def_word_gating="none",
            dropout_type="per_unit", compose_type="sum",
            word_dropout_weighting="no_weighting",
            shortcut_unk_and_excluded=False,
            num_input_words=-1, exclude_top_k=-1, vocab=None,
            **kwargs):

        self._dropout = dropout
        self._num_input_words = num_input_words
        self._exclude_top_K = exclude_top_k
        self._dropout_type = dropout_type
        self._compose_type = compose_type
        self._vocab = vocab
        self._shortcut_unk_and_excluded = shortcut_unk_and_excluded
        self._word_dropout_weighting = word_dropout_weighting
        self._def_word_gating = def_word_gating

        if def_word_gating not in {"none", "self_attention"}:
            raise NotImplementedError()

        if word_dropout_weighting not in {"no_weighting"}:
            raise NotImplementedError("Not implemented " + word_dropout_weighting)

        if dropout_type not in {"per_unit", "per_example", "per_word"}:
            raise NotImplementedError()

        children = []

        if self._def_word_gating=="self_attention":
            self._gate_mlp = Linear(dim, dim,  weights_init=GlorotUniform(), biases_init=Constant(0))
            self._gate_act = Logistic()
            children.extend([self._gate_mlp, self._gate_act])

        if compose_type == 'fully_connected_linear':
            self._def_state_compose = MLP(activations=[None],
                dims=[emb_dim + dim, emb_dim], weights_init=GlorotUniform(), biases_init=Constant(0))
            children.append(self._def_state_compose)
        if compose_type == "gated_sum" or compose_type == "gated_transform_and_sum":
            if dropout_type == "per_word" or dropout_type == "per_example":
                raise RuntimeError("I dont think this combination makes much sense")

            self._compose_gate_mlp = Linear(dim + emb_dim, emb_dim,
                                            weights_init=GlorotUniform(),
                                            biases_init=Constant(0),
                                            name='gate_linear')
            self._compose_gate_act = Logistic()
            children.extend([self._compose_gate_mlp, self._compose_gate_act])
        if compose_type == 'sum':
            if not emb_dim == dim:
                raise ValueError("Embedding has different dim! Cannot use compose_type='sum'")
        if compose_type == 'transform_and_sum' or compose_type == "gated_transform_and_sum":
            self._def_state_transform = Linear(dim, emb_dim, name='state_transform')
            children.append(self._def_state_transform)

        super(MeanPoolCombiner, self).__init__(children=children, **kwargs)

    @application
    def apply(self, application_call,
              word_embs, words_mask,
              def_embeddings, def_map, train_phase=False, word_ids=False, call_name=""):
        batch_shape = word_embs.shape
        flat_indices = def_map[:, 0] * batch_shape[1] + def_map[:, 1] # Index of word in flat

        # def_map is (seq_pos, word_pos, def_index)
        # def_embeddings is (id, emb_dim)

        def_sum = T.zeros((batch_shape[0] * batch_shape[1], def_embeddings.shape[1]))
        def_lens = T.zeros_like(def_sum[:, 0])
        def_lens = T.inc_subtensor(def_lens[flat_indices], 1)

        if self._def_word_gating == "none":
            def_sum = T.inc_subtensor(def_sum[flat_indices],
                def_embeddings[def_map[:, 2]])
            def_mean = def_sum / T.maximum(def_lens[:, None], 1)
        elif self._def_word_gating == "self_attention":
            gates = def_embeddings[def_map[:, 2]]
            gates = self._gate_mlp.apply(gates)[:, 0]
            application_call.add_auxiliary_variable(gates, name='def_gates')

            # Dima: this is numerically unstable. But maybe it can work.
            # If it can work, we can avoid too much coding.
            def_normalization = T.zeros_like(def_lens)
            def_normalization = T.inc_subtensor(
                def_normalization[flat_indices], T.exp(gates))
            gates = T.exp(gates) / def_normalization[flat_indices]

            def_mean = T.inc_subtensor(def_sum[flat_indices],
                gates[:, None] * def_embeddings[def_map[:, 2]])
        else:
            raise NotImplementedError()
        def_mean = def_mean.reshape((batch_shape[0], batch_shape[1], -1))

        application_call.add_auxiliary_variable(
            masked_root_mean_square(def_mean, words_mask), name=call_name + '_def_mean_rootmean2')

        if train_phase and self._dropout != 0.0:
            if self._dropout_type == "per_unit":
                logger.info("Adding per_unit drop on dict and normal emb")
                word_embs = apply_dropout(word_embs, drop_prob=self._dropout)
                def_mean = apply_dropout(def_mean, drop_prob=self._dropout)
            elif self._dropout_type == "per_example":
                logger.info("Adding per_example drop on dict and normal emb")
                # We dropout mask
                mask_defs = T.ones((batch_shape[0],))
                mask_we = T.ones((batch_shape[0],))

                # Mask dropout
                mask_defs = apply_dropout(mask_defs, drop_prob=self._dropout)
                mask_we = apply_dropout(mask_we, drop_prob=self._dropout)

                # this reduces variance. If both 0 will select both
                where_both_zero = T.eq((mask_defs + mask_we), 0)

                mask_defs = (where_both_zero + mask_defs).dimshuffle(0, "x", "x")
                mask_we = (where_both_zero + mask_we).dimshuffle(0, "x", "x")

                def_mean = mask_defs * def_mean
                word_embs = mask_we * word_embs
            elif self._dropout_type == "per_word_independent":
                # TODO: Maybe we also want to have possibility of including both (like in per_example)
                pass # TODO: implement
            elif self._dropout_type == "per_word":
                # TODO(kudkudak): This dropout doesn't really work for me

                # Drop with dropout percentage. If dropout -> selects at random word vs df
                mask_higher = T.ones((batch_shape[0], batch_shape[1]))
                mask_higher = apply_dropout(mask_higher, drop_prob=self._dropout)
                mask_higher = mask_higher.dimshuffle(0, 1, "x")

                logger.info("Apply per_word dropou on dict and normal emb")
                mask = T.ones((batch_shape[0], batch_shape[1]))
                mask = apply_dropout(mask, drop_prob=0.5)
                mask = mask.dimshuffle(0, 1, "x")

                # Competitive
                def_mean = mask_higher * def_mean + (1 - mask_higher) * mask * def_mean
                word_embs = word_embs * def_mean + (1 - mask_higher) * (1 - mask) * word_embs

                # TODO: Smarter weighting (at least like divisor in dropout)

                if not self._compose_type == "sum" and not self._compose_type == "transform_and_sum":
                    raise NotImplementedError()

        application_call.add_auxiliary_variable(
            def_mean.copy(),
            name=call_name + '_dict_word_embeddings')

        application_call.add_auxiliary_variable(
            word_embs.copy(),
            name=call_name + '_word_embeddings')

        if self._compose_type == 'sum':
            final_embeddings = word_embs + def_mean
        elif self._compose_type == 'transform_and_sum':
            final_embeddings = (word_embs +
                                self._def_state_transform.apply(def_mean))
        elif self._compose_type == 'gated_sum' or self._compose_type == 'gated_transform_and_sum':
            concat = T.concatenate([word_embs, def_mean], axis=2)
            gates = concat.reshape((batch_shape[0] * batch_shape[1], -1))
            gates = self._compose_gate_mlp.apply(gates)
            gates = self._compose_gate_act.apply(gates)
            gates = gates.reshape((batch_shape[0], batch_shape[1], -1))

            if self._compose_type == 'gated_sum':
                final_embeddings = gates * word_embs + (1 - gates) * def_mean
            else:
                final_embeddings = gates * word_embs + (1 - gates) * self._def_state_transform.apply(def_mean)

            application_call.add_auxiliary_variable(
                masked_root_mean_square(gates.reshape((batch_shape[0], batch_shape[1], -1)), words_mask),
                name=call_name + '_compose_gate_rootmean2')
        elif self._compose_type.startswith('fully_connected'):
            concat = T.concatenate([word_embs, def_mean], axis=2)
            final_embeddings = self._def_state_compose.apply(concat)
        else:
            raise NotImplementedError()

        if self._shortcut_unk_and_excluded:
            # NOTE: It might be better to move it out of Lookup, because it breaks API a bit
            # but at the same time it makes sense to share this code

            # 1. If no def, just go with word emb
            final_embeddings = word_embs * T.lt(word_ids, self._exclude_top_K).dimshuffle(0, 1, "x") + \
                               final_embeddings * T.ge(word_ids, self._exclude_top_K).dimshuffle(0, 1, "x")

            # 2. UNKs always get def embeddings (UNK can happen for dev/test set of course)
            final_embeddings = final_embeddings * T.neq(word_ids, self._vocab.unk).dimshuffle(0, 1, "x") + \
                               def_mean * T.eq(word_ids, self._vocab.unk).dimshuffle(0, 1, "x")


        application_call.add_auxiliary_variable(
            masked_root_mean_square(final_embeddings, words_mask),
            name=call_name + '_merged_input_rootmean2')

        return final_embeddings

