"""
Baseline SNLI model

Inspired by https://github.com/Smerity/keras_snli

TODO: Refactor SNLI baseline so that it takes embedded words (this will factor out dict lookup kwargs nicely)
TODO: Add bn before LSTM
TODO: Recurrent dropout
TODO: Pass translate intelligently
TODO: Debug simlex?
"""
import theano
import theano.tensor as T
from theano import tensor

import logging
logger = logging.getLogger(__file__)

from blocks.bricks import Initializable, Linear, MLP
from blocks.bricks import Softmax, Rectifier
from blocks.bricks.bn import BatchNormalization
from blocks.bricks.recurrent import LSTM
from blocks.bricks.base import application, lazy
from blocks.bricks.lookup import LookupTable
from blocks.initialization import IsotropicGaussian, Constant, NdarrayInitialization, Uniform

from dictlearn.inits import GlorotUniform
from dictlearn.lookup import MeanPoolCombiner, LSTMReadDefinitions, MeanPoolReadDefinitions
from dictlearn.theano_util import apply_dropout

class NLISimple(Initializable):
    """
    Simple model based on https://github.com/Smerity/keras_snl
    """

    def __init__(self, mlp_dim, translate_dim, emb_dim, vocab, num_input_words=-1,
            num_input_def_words=-1, dropout=0.2, encoder="sum",
            n_layers=3, translate_after_emb=True,

            # Dict lookup kwargs
            retrieval=None, reader_type="rnn", compose_type="sum", def_dim=300,
            disregard_word_embeddings=False, combiner_dropout=1.0, combiner_bn=False,
            combiner_dropout_type="regular", share_def_lookup=False, exclude_top_k=-1,
            combiner_reader_translate=True, def_vocab=None, def_emb_dim=-1,
            combiner_gating="none",
            combiner_shortcut=False,
            # Others
            **kwargs):

        if def_emb_dim <= 0:
            logger.info("Assuming def_emb_dim=emb_dim")
            def_emb_dim = emb_dim

        self._vocab = vocab
        self._encoder = encoder
        self._dropout = dropout
        self._retrieval = retrieval
        self._only_def = disregard_word_embeddings
        self._num_input_def_words = num_input_def_words
        self._translate_after_emb = translate_after_emb

        if reader_type not in {"rnn", "mean"}:
            raise NotImplementedError("Not implemented " + reader_type)

        if num_input_words > 0:
            logger.info("Restricting vocab to " + str(num_input_words))
            self._num_input_words = num_input_words
        else:
            logger.info("Vocab set to " + str(vocab.size()))
            self._num_input_words = vocab.size()

        children = []

        if not disregard_word_embeddings:
            self._lookup = LookupTable(self._num_input_words, emb_dim, weights_init=GlorotUniform())
            children.append(self._lookup)

        if retrieval:
            if share_def_lookup:
                def_lookup = self._lookup
            else:
                def_lookup = None

            if reader_type== "rnn":
                self._def_reader = LSTMReadDefinitions(num_input_words=self._num_input_def_words,
                    weights_init=Uniform(width=0.1), translate=combiner_reader_translate,
                    biases_init=Constant(0.), dim=def_dim, emb_dim=def_emb_dim, vocab=def_vocab, lookup=def_lookup)
            elif reader_type == "mean":
                if combiner_reader_translate:
                    logger.warning("Translate in MeanPoolReadDefinitions is redundant")
                self._def_reader = MeanPoolReadDefinitions(num_input_words=self._num_input_def_words,
                    translate=combiner_reader_translate,
                    weights_init=Uniform(width=0.1), lookup=def_lookup, dim=def_emb_dim,
                    biases_init=Constant(0.), emb_dim=def_emb_dim, vocab=def_vocab)

            self._combiner = MeanPoolCombiner(dim=def_dim, emb_dim=def_emb_dim,
                dropout=combiner_dropout, dropout_type=combiner_dropout_type,
                def_word_gating=combiner_gating,
                shortcut_unk_and_excluded=combiner_shortcut, num_input_words=num_input_words, exclude_top_k=exclude_top_k, vocab=vocab,
                compose_type=compose_type)
            children.extend([self._def_reader, self._combiner])

            if self._encoder == "rnn":
                self._rnn_fork = Linear(input_dim=def_emb_dim, output_dim=4 * translate_dim,
                    weights_init=GlorotUniform(), biases_init=Constant(0))
                # TODO(kudkudak): Better LSTM weight init
                self._rnn_encoder = LSTM(dim=translate_dim, name='LSTM_encoder', weights_init=Uniform(width=0.1))
                children.append(self._rnn_fork)
                children.append(self._rnn_encoder)
            elif self._encoder == "sum":
                pass
            else:
                raise NotImplementedError("Not implemented encoder")

            if def_emb_dim != emb_dim:
                self._translate_pre_def = Linear(input_dim=emb_dim, output_dim=def_emb_dim,
                    weights_init=GlorotUniform(), biases_init=Constant(0))
                children.append(self._translate_pre_def)
            else:
                self._translate_pre_def = None

            if self._translate_after_emb:
                self._translation = Linear(input_dim=def_dim, output_dim=translate_dim,
                    weights_init=GlorotUniform(), biases_init=Constant(0))
                self._translation_act = Rectifier()
                children.append(self._translation)
                children.append(self._translation_act)
        else:
            if self._encoder == "rnn":
                self._translation = Linear(input_dim=emb_dim, output_dim=4 * translate_dim,
                    weights_init=GlorotUniform(), biases_init=Constant(0))
                self._rnn_fork = self._translation
                self._rnn_encoder = LSTM(dim=translate_dim, name='LSTM_encoder', weights_init=Uniform(width=0.01))
                children.append(self._rnn_encoder)
                children.append(self._translation)
            elif self._encoder == "sum":
                self._translation = Linear(input_dim=emb_dim, output_dim=translate_dim,
                    weights_init=GlorotUniform(), biases_init=Constant(0))
                self._translation_act = Rectifier()
                children.append(self._translation)
                children.append(self._translation_act)
            else:
                raise NotImplementedError("Not implemented encoder")



        self._hyp_bn = BatchNormalization(input_dim=translate_dim, name="hyp_bn", conserve_memory=False)
        self._prem_bn = BatchNormalization(input_dim=translate_dim, name="prem_bn", conserve_memory=False)
        children += [self._hyp_bn, self._prem_bn]

        self._mlp = []
        current_dim = 2 * translate_dim  # Joint
        for i in range(n_layers):
            rect = Rectifier()
            dense = Linear(input_dim=current_dim, output_dim=mlp_dim,
                name="MLP_layer_" + str(i), \
                weights_init=GlorotUniform(), \
                biases_init=Constant(0))
            current_dim = mlp_dim
            bn = BatchNormalization(input_dim=current_dim, name="BN_" + str(i), conserve_memory=False)
            children += [dense, rect, bn] #TODO: Strange place to put ReLU
            self._mlp.append([dense, rect, bn])

        self._pred = MLP([Softmax()], [current_dim, 3], \
            weights_init=GlorotUniform(), \
            biases_init=Constant(0))
        children.append(self._pred)

        super(NLISimple, self).__init__(children=children, **kwargs)

    def get_embeddings_lookups(self):
        return [self._lookup]

    def set_embeddings(self, embeddings):
        self._lookup.parameters[0].set_value(embeddings.astype(theano.config.floatX))

    @application
    def apply(self, application_call,
            s1_preunk, s1_mask, s2_preunk, s2_mask, def_mask=None, defs=None, s1_def_map=None,
            s2_def_map=None, train_phase=True):

        # Shortlist words (sometimes we want smaller vocab, especially when dict is small)
        s1 = (tensor.lt(s1_preunk, self._num_input_words) * s1_preunk
              + tensor.ge(s1_preunk, self._num_input_words) * self._vocab.unk)
        s2 = (tensor.lt(s2_preunk, self._num_input_words) * s2_preunk
              + tensor.ge(s2_preunk, self._num_input_words) * self._vocab.unk)

        # Embeddings
        s1_emb = self._lookup.apply(s1)
        s2_emb = self._lookup.apply(s2)

        if self._retrieval is not None:
            assert defs is not None

            if self._translate_pre_def:
                logger.info("Translate pre def")
                s1_emb = s1_emb.reshape((s1_emb.shape[0] * s1_emb.shape[1], s1_emb.shape[2]))
                s2_emb = s2_emb.reshape((s2_emb.shape[0] * s2_emb.shape[1], s2_emb.shape[2]))
                s1_emb = self._translate_pre_def.apply(s1_emb)
                s2_emb = self._translate_pre_def.apply(s2_emb)
                s1_emb = s1_emb.reshape((s1_preunk.shape[0], s1_preunk.shape[1], -1))
                s2_emb = s2_emb.reshape((s2_preunk.shape[0], s2_preunk.shape[1], -1))


            def_embs = self._def_reader.apply(defs, def_mask)

            s1_transl = self._combiner.apply(
                s1_emb, s1_mask,
                def_embs, s1_def_map, word_ids=s1, train_phase=train_phase, call_name="s1")

            s2_transl = self._combiner.apply(
                s2_emb, s2_mask,
                def_embs, s2_def_map, word_ids=s2, train_phase=train_phase, call_name="s2")

            if self._translate_after_emb:
                # Note: for some reader/combiner it can be redundant, but let's keep it
                # often it will not be redundant (ex: LSTMDefReader)
                s1_transl = s1_transl.reshape((s1_transl.shape[0] * s1_transl.shape[1], s1_transl.shape[2]))
                s2_transl = s2_transl.reshape((s2_transl.shape[0] * s2_transl.shape[1], s2_transl.shape[2]))
                s1_transl = self._translation.apply(s1_transl)
                s2_transl = self._translation.apply(s2_transl)
                s1_transl = self._translation_act.apply(s1_transl)
                s2_transl = self._translation_act.apply(s2_transl)
                s1_transl = s1_transl.reshape((s1_emb.shape[0], s1_emb.shape[1], -1))
                s2_transl = s2_transl.reshape((s2_emb.shape[0], s2_emb.shape[1], -1))
        else:
            application_call.add_auxiliary_variable(
                1*s1_emb,
                name='s1_word_embeddings')

            # Translate. Crucial for recovering useful information from embeddings
            s1_emb_flatten = s1_emb.reshape((s1_emb.shape[0] * s1_emb.shape[1], s1_emb.shape[2]))
            s2_emb_flatten = s2_emb.reshape((s2_emb.shape[0] * s2_emb.shape[1], s2_emb.shape[2]))

            if self._translate_after_emb:
                # Note: redundant for trainable (from scratch) embeddings
                s1_transl = self._translation.apply(s1_emb_flatten)
                s2_transl = self._translation.apply(s2_emb_flatten)
                s1_transl = self._translation_act.apply(s1_transl)
                s2_transl = self._translation_act.apply(s2_transl)
                s1_transl = s1_transl.reshape((s1_emb.shape[0], s1_emb.shape[1], -1))
                s2_transl = s2_transl.reshape((s2_emb.shape[0], s2_emb.shape[1], -1))
                application_call.add_auxiliary_variable(
                    1*s1_transl,
                    name='s1_translated_word_embeddings')
                assert s1_transl.ndim == 3
            else:
                s1_transl = s1_emb_flatten
                s2_transl = s2_emb_flatten

        if self._encoder == "rnn":
            s1_transl = self._rnn_fork.apply(s1_transl)
            s2_transl = self._rnn_fork.apply(s2_transl)

        assert s1_transl.ndim == s2_transl.ndim == 3

        # Construct entailment embedding
        if self._encoder == "sum":
            s1_emb_mask = s1_mask.dimshuffle((0, 1, "x"))
            s2_emb_mask = s2_mask.dimshuffle((0, 1, "x"))

            # TODO: This should be mean, might make learning harder otherwise
            prem = (s1_emb_mask * s1_transl).sum(axis=1)
            hyp = (s2_emb_mask * s2_transl).sum(axis=1)
        else:
            prem = self._rnn_encoder.apply(s1_transl.transpose(1, 0, 2), mask=s1_mask.transpose(1, 0))[0][-1]
            hyp = self._rnn_encoder.apply(s2_transl.transpose(1, 0, 2), mask=s2_mask.transpose(1, 0))[0][-1]

        prem = self._prem_bn.apply(prem)
        hyp = self._hyp_bn.apply(hyp)

        joint = T.concatenate([prem, hyp], axis=1)
        joint.name = "MLP_input"

        if train_phase:
            joint = apply_dropout(joint, drop_prob=self._dropout)

        # MLP
        for block in self._mlp:
            dense, relu, bn = block
            joint = dense.apply(joint)
            joint = relu.apply(joint)

            if train_phase:
                joint = apply_dropout(joint, drop_prob=self._dropout)

            joint = bn.apply(joint)

        return self._pred.apply(joint)