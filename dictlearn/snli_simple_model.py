"""
Baseline SNLI model

Inspired by https://github.com/Smerity/keras_snli

TODO: Refactor SNLI baseline so that it takes embedded words (this will factor out dict lookup kwargs nicely)
TODO: Add bn before LSTM
TODO: Recurrent dropout
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
from dictlearn.lookup import DictEnchancedLookup

class SNLISimple(Initializable):
    """
    Simple model based on https://github.com/Smerity/keras_snl
    """

    def __init__(self, mlp_dim, translate_dim, emb_dim, vocab, num_input_words=-1, dropout=0.2, encoder="sum", n_layers=3,
            # Dict lookup kwargs
            retrieval=None, compose_type="sum", disregard_word_embeddings=False, multimod_drop=1.0,
            # Others
            **kwargs):

        self._vocab = vocab
        self._encoder = encoder
        self._dropout = dropout
        self._retrieval = retrieval

        if num_input_words > 0:
            logger.info("Restricting vocab to " + str(num_input_words))
            self._num_input_words = num_input_words
        else:
            self._num_input_words = vocab.size()

        children = []

        if retrieval:
            self._lookup = DictEnchancedLookup(emb_dim=emb_dim, retrieval=retrieval, vocab=vocab,
                dim=translate_dim, disregard_word_embeddings=disregard_word_embeddings, multimod_drop=multimod_drop,
                num_input_words=num_input_words, compose_type=compose_type)
            if self._encoder == "rnn":
                # Translation serves as a "fork" to LSTM
                self._rnn_fork = Linear(input_dim=translate_dim, output_dim=4 * translate_dim,
                    weights_init=GlorotUniform(), biases_init=Constant(0))
                # TODO(kudkudak): Better LSTM weight init
                self._rnn_encoder = LSTM(dim=translate_dim, name='LSTM_encoder', weights_init=Uniform(width=0.1))
                children.append(self._rnn_fork)
                children.append(self._rnn_encoder)
            elif self._encoder == "sum":
                pass
            else:
                raise NotImplementedError("Not implemented encoder")
        else:
            self._lookup = LookupTable(self._num_input_words, emb_dim, weights_init=GlorotUniform())
            if self._encoder == "rnn":
                # Translation serves as a "fork" to LSTM
                self._translation = Linear(input_dim=emb_dim, output_dim=4 * translate_dim,
                    weights_init=GlorotUniform(), biases_init=Constant(0))
                # TODO(kudkudak): Activation?
                # TODO(kudkudak): Better LSTM weight init
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
        children.append(self._lookup)


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

        super(SNLISimple, self).__init__(children=children, **kwargs)

    def get_embeddings_lookups(self):
        if isinstance(self._lookup, LookupTable):
            return [self._lookup]
        elif isinstance(self._lookup, DictEnchancedLookup):
            return [self._lookup._base_lookup, self._lookup._def_lookup]
        else:
            raise NotImplementedError()

    def set_embeddings(self, embeddings):
        if isinstance(self._lookup, LookupTable):
            self._lookup.parameters[0].set_value(embeddings.astype(theano.config.floatX))
        elif isinstance(self._lookup, DictEnchancedLookup):
            self._lookup.set_embeddings(embeddings)
        else:
            raise NotImplementedError()

    def embeddings_var(self):
        if isinstance(self._lookup, LookupTable):
            return self._lookup.parameters[0]
        elif isinstance(self._lookup, DictEnchancedLookup):
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    def get_cg_transforms(self):
        cg = self._cg_transforms
        if isinstance(self._lookup, DictEnchancedLookup):
            cg += self._lookup.get_cg_transforms()
        elif isinstance(self._lookup, LookupTable):
            pass
        else:
            raise NotImplementedError()
        print(cg)
        return cg

    @application
    def apply(self, application_call,
            s1, s1_mask, s2, s2_mask, def_mask=None, defs=None, s1_def_map=None, s2_def_map=None):

        if isinstance(self._lookup, LookupTable):

            assert defs is None

            # Shortlist words (sometimes we want smaller vocab, especially when dict is small)
            s1 = (tensor.lt(s1, self._num_input_words) * s1
                              + tensor.ge(s1, self._num_input_words) * self._vocab.unk)
            s2 = (tensor.lt(s2, self._num_input_words) * s2
                               + tensor.ge(s2, self._num_input_words) * self._vocab.unk)

            # Embeddings
            s1_emb = self._lookup.apply(s1)
            s2_emb = self._lookup.apply(s2)

            # Translate. Crucial for recovering useful information from embeddings
            s1_emb_flatten = s1_emb.reshape((s1_emb.shape[0] * s1_emb.shape[1], s1_emb.shape[2]))
            s2_emb_flatten = s2_emb.reshape((s2_emb.shape[0] * s2_emb.shape[1], s2_emb.shape[2]))
            s1_transl = self._translation.apply(s1_emb_flatten)
            s2_transl = self._translation.apply(s2_emb_flatten)
            s1_transl = self._translation_act.apply(s1_transl)
            s2_transl = self._translation_act.apply(s2_transl)
            s1_transl = s1_transl.reshape((s1_emb.shape[0], s1_emb.shape[1], -1))
            s2_transl = s2_transl.reshape((s2_emb.shape[0], s2_emb.shape[1], -1))
            assert s1_transl.ndim == 3
        elif isinstance(self._lookup, DictEnchancedLookup):

            assert defs is not None

            print("DictEnchancedLookup")
            # This is hidden in DictEnchancedLookup then
            s1_transl = self._lookup.apply_with_given_defs(
                s1, s1_mask,
                defs, def_mask, s1_def_map)
            s2_transl = self._lookup.apply_with_given_defs(
                s2, s2_mask,
                defs, def_mask, s2_def_map)

            if self._encoder == "rnn":
                s1_transl = self._rnn_fork.apply(s1_transl)
                s2_transl = self._rnn_fork.apply(s2_transl)
        else:
            raise NotImplementedError()

        assert s1_transl.ndim == s2_transl.ndim == 3

        # Construct entailment embedding
        if self._encoder == "sum":
            s1_emb_mask = s1_mask.dimshuffle((0, 1, "x"))
            s2_emb_mask = s2_mask.dimshuffle((0, 1, "x"))
            prem = (s1_emb_mask * s1_transl).sum(axis=1)
            hyp = (s2_emb_mask * s2_transl).sum(axis=1)
            # prem = ( s1_transl).sum(axis=1)
            # hyp = (s2_transl).sum(axis=1)
        else:
            prem = self._rnn_encoder.apply(s1_transl.transpose(1, 0, 2), mask=s1_mask.transpose(1, 0))[0][-1]
            hyp = self._rnn_encoder.apply(s2_transl.transpose(1, 0, 2), mask=s2_mask.transpose(1, 0))[0][-1]

        prem = self._prem_bn.apply(prem)
        hyp = self._hyp_bn.apply(hyp)

        joint = T.concatenate([prem, hyp], axis=1)
        joint.name = "MLP_input"
        self._cg_transforms = [['dropout', self._dropout, joint]]

        # MLP
        for block in self._mlp:
            dense, relu, bn = block
            joint = dense.apply(joint)
            joint = relu.apply(joint)
            self._cg_transforms.append(['dropout', self._dropout, joint])
            joint = bn.apply(joint)

        return self._pred.apply(joint)
