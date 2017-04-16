"""
Baseline SNLI model

Inspired by https://github.com/Smerity/keras_snli
"""
import theano
import theano.tensor as T
from blocks.bricks import Initializable, Linear, MLP
from blocks.bricks import Softmax
from blocks.bricks.bn import BatchNormalization
from blocks.bricks.recurrent import LSTM
from blocks.bricks.base import application, lazy
from blocks.bricks.lookup import LookupTable
from blocks.initialization import IsotropicGaussian, Constant, NdarrayInitialization, Uniform
from theano import tensor

import numpy as np

from keras.backend import theano_backend

class GlorotUniform(NdarrayInitialization):
    """Initialize parameters from an isotropic Gaussian distribution.

    Parameters
    ----------
    std : float, optional
        The standard deviation of the Gaussian distribution. Defaults to 1.
    mean : float, optional
        The mean of the Gaussian distribution. Defaults to 0

    Notes
    -----
    Be careful: the standard deviation goes first and the mean goes
    second!

    """
    def __init__(self):
        pass

    def generate(self, rng, shape):
        if not len(shape) == 2:
            raise NotImplementedError()

        fan_in, fan_out = shape[0], shape[1]
        s = np.sqrt(6. / (fan_in + fan_out))
        return rng.uniform(size=shape, low=-s, high=s).astype(theano.config.floatX)

    def __repr__(self):
        return "GlorotUniform"


class SNLIBaseline(Initializable):
    """

    Simple model based on https://github.com/Smerity/keras_snl
    """

    def __init__(self, translate_dim, emb_dim, vocab, dropout=0.2,
            encoder="sum",
            n_layers=3, **kwargs):

        self._vocab = vocab
        self._encoder = encoder
        self._dropout = dropout
        self._num_input_words = vocab.size()


        children = []
        self._lookup = LookupTable(self._num_input_words, emb_dim, weights_init=GlorotUniform())

        if self._encoder == "rnn":
            # Translation serves as fork to LSTM
            self._translation = Linear(input_dim=emb_dim, output_dim=4*translate_dim,
                weights_init=GlorotUniform(), biases_init=Constant(0))
            self._rnn_encoder = LSTM(dim=translate_dim, name='LSTM_encoder', weights_init=Uniform(width=0.1))
            children.append(self._rnn_encoder)
        elif self._encoder == "sum":
            self._translation = Linear(input_dim=emb_dim, output_dim=translate_dim,
                weights_init=GlorotUniform(), biases_init=Constant(0))
        else:
            raise NotImplementedError("Not implemented encoder")

        self._hyp_bn = BatchNormalization(input_dim=translate_dim, name="hyp_bn")
        self._prem_bn = BatchNormalization(input_dim=translate_dim, name="prem_bn")

        self._mlp = []
        current_dim = 2 * translate_dim  # Joint
        for i in range(n_layers):
            dense = Linear(input_dim=current_dim, output_dim=2 * translate_dim,
                name="MLP_layer_" + str(i), \
                weights_init=GlorotUniform(), \
                biases_init=Constant(0))
            bn = BatchNormalization(input_dim=2 * translate_dim, name="BN_" + str(i))
            children += [dense, bn]
            self._mlp.append([dense, bn])
            cur_dim = 2 * translate_dim

        children += [self._lookup, self._translation]
        children += [self._hyp_bn, self._prem_bn]

        self._pred = MLP([Softmax()], [cur_dim, 3], \
            weights_init=GlorotUniform(), \
            biases_init=Constant(0))

        children.append(self._pred)

        super(SNLIBaseline, self).__init__(children=children, **kwargs)

    def set_embeddings(self, embeddings):
        self._lookup.parameters[0].set_value(embeddings.astype(theano.config.floatX))

    def embeddings_var(self):
        return self._lookup.parameters[0]

    def get_cg_transforms(self):
        # Note: dropout is not part of model specification in blocks convention

        # Uniform dropout for now
        return self._cg_transforms

    @application
    def apply(self, application_call,
            s1, s1_mask, s2, s2_mask):
        s1_emb = self._lookup.apply(s1)
        s2_emb = self._lookup.apply(s2)

        # Translate. Crucial for recovering useful information from embeddings
        s1_emb_flatten = s1_emb.reshape((s1_emb.shape[0] * s1_emb.shape[1], s1_emb.shape[2]))
        s2_emb_flatten = s2_emb.reshape((s2_emb.shape[0] * s2_emb.shape[1], s2_emb.shape[2]))
        s1_transl = self._translation.apply(s1_emb_flatten)
        s2_transl = self._translation.apply(s2_emb_flatten)
        s1_transl = s1_transl.reshape((s1_emb.shape[0], s1_emb.shape[1], -1))
        s2_transl = s2_transl.reshape((s2_emb.shape[0], s2_emb.shape[1], -1))

        assert s1_transl.ndim == 3
        assert s2_emb.ndim == s1_emb.ndim == 3

        # Construct entailment embedding
        if self._encoder == "sum":
            s1_emb_mask = s1_mask.dimshuffle((0, 1, "x"))
            s2_emb_mask = s2_mask.dimshuffle((0, 1, "x"))
            prem = (s1_emb_mask * s1_transl).sum(axis=1)
            hyp = (s2_emb_mask * s2_transl).sum(axis=1)
        else:
            prem = self._rnn_encoder.apply(s1_transl.transpose(1, 0, 2), mask=s1_mask.transpose(1, 0))[0][-1]
            hyp = self._rnn_encoder.apply(s2_transl.transpose(1, 0, 2), mask=s2_mask.transpose(1, 0))[0][-1]

        prem = self._prem_bn.apply(prem)
        hyp = self._hyp_bn.apply(hyp)

        joint = T.concatenate([prem, hyp], axis=1)
        self._cg_transforms = []

        # MLP
        for block in self._mlp:
            dense, bn = block
            joint = dense.apply(joint)
            self._cg_transforms.append(['dropout', self._dropout, joint])
            joint = bn.apply(joint)

        return self._pred.apply(joint)
