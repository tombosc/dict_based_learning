"""
Baseline SNLI model

Inspired by https://github.com/Smerity/keras_snli
"""
import theano
from theano import tensor

from blocks.bricks import Initializable, Linear, NDimensionalSoftmax, MLP, Tanh, Rectifier
from blocks.bricks.base import application
from blocks.bricks.recurrent import LSTM
from blocks.bricks.lookup import LookupTable

from dictlearn.ops import WordToIdOp, RetrievalOp
from dictlearn.aggregation_schemes import Perplexity
from dictlearn.stuff import DebugLSTM
from dictlearn.util import masked_root_mean_square

import sys
from dictlearn import ops, vocab
import dictlearn
from dictlearn.ops import WordToIdOp
from blocks.algorithms import GradientDescent, Scale
from blocks.bricks import MLP, Tanh, Softmax
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.initialization import IsotropicGaussian, Constant
from fuel.streams import DataStream
from fuel.transformers import Flatten
from fuel.datasets import MNIST
from fuel.schemes import SequentialScheme
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.main_loop import MainLoop
from blocks.roles import WEIGHT
import theano.tensor as T
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.bricks.lookup import LookupTable
from blocks.bricks import Initializable, Linear, NDimensionalSoftmax, MLP, Tanh, Rectifier
from blocks.bricks.base import application
from blocks.bricks.recurrent import LSTM
from blocks.bricks.recurrent.misc import Bidirectional
from blocks.bricks.lookup import LookupTable
from blocks.graph import apply_dropout, apply_batch_normalization


class TimeDistributedDense(Linear):
    r"""

    Applies Wx + b at each step of sequence

    """

    @lazy(allocation=['input_dim', 'output_dim'])
    def __init__(self, input_dim, output_dim, **kwargs):
        super(TimeDistributedDense, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Apply the linear transformation.

        Parameters
        ----------
        input_ : :class:`~tensor.TensorVariable`
            The input on which to apply the transformation

        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            The transformed input plus optional bias

        """

        assert input_.ndim == 3, "Works on sequences"

        input_.reshape((input_.shape[0] * input_.shape[1], input_.shape[2]))
        output = T.dot(input_, self.W)
        if getattr(self, 'use_bias', True):
            output += self.b
        return output.reshape((input_.shape[0], input_.shape[1], output.shape[2]))


class SNLIBaseline(Initializable):
    """

    Simple model based on https://github.com/Smerity/keras_snli

    TODO(kudkudak): Better inits (why there is no sensible init in blocks :P ??)
    """

    def __init__(self, translate_dim, emb_dim, vocab, dropout=0.2, encoder="sum",
            n_layers=3, retrieval=None, **kwargs):

        self._vocab = vocab
        self._encoder = encoder
        self._dropout = dropout
        self._num_input_words = vocab.size()

        if encoder != "sum":
            raise NotImplementedError()

        children = []
        self._lookup = LookupTable(self._num_input_words, emb_dim, weights_init=IsotropicGaussian(0.01))

        self._translation = TimeDistributedDense(input_dim=emb_dim, output_dim=translate_dim, \
            weights_init=IsotropicGaussian(0.01), \
            biases_init=Constant(0))

        self._hyp_bn = BatchNormalization(input_dim=translate_dim, name="hyp_bn")
        self._prem_bn = BatchNormalization(input_dim=translate_dim, name="prem_bn")

        self._mlp = []
        current_dim = 2 * translate_dim  # Joint
        for i in range(n_layers):
            dense = Linear(input_dim=current_dim, output_dim=2 * translate_dim,
                name="MLP_layer_" + str(i), \
                weights_init=IsotropicGaussian(0.01), \
                biases_init=Constant(0))
            bn = BatchNormalization(input_dim=2 * translate_dim, name="BN_" + str(i))
            children += [dense, bn]
            self._mlp.append([dense, bn])
            cur_dim = 2 * translate_dim

        children += [self._lookup, self._translation]
        children += [self._hyp_bn, self._prem_bn]

        self._pred = MLP([Softmax()], [cur_dim, 3], \
            weights_init=IsotropicGaussian(0.01), \
            biases_init=Constant(0))

        children.append(self._pred)

        super(SNLIBaseline, self).__init__(children=children, **kwargs)

    def set_embeddings(self, embeddings):
        self._lookup.parameters[0].set_value(embeddings.astype(theano.config.floatX))

    def embeddings_var(self):
        return self._lookup.parameters[0]

    # TODO: How to nicely code dropout in blocks?
    def get_cg_transforms(self):
        # Note: dropout is not part of model specification in blocks convention

        # Uniform dropout for now
        return self._cg_transforms

    @application
    def apply(self, application_call,
            s1, s1_mask, s2, s2_mask):
        s1_emb = self._lookup.apply(s1)
        s2_emb = self._lookup.apply(s2)

        s1_transl = self._translation.apply(s1_emb)
        s2_transl = self._translation.apply(s2_emb)

        assert s1_transl.ndim == 3

        s1_emb_mask = s1_mask.dimshuffle((0, 1, "x"))
        s2_emb_mask = s2_mask.dimshuffle((0, 1, "x"))

        assert s2_emb.ndim == s1_emb.ndim == 3

        # Construct entailment embedding
        prem = (s1_emb_mask * s1_transl).sum(axis=1)
        hyp = (s2_emb_mask * s2_transl).sum(axis=1)

        prem = self._prem_bn.apply(prem)
        hyp = self._hyp_bn.apply(hyp)

        joint = T.concatenate([prem, hyp], axis=1)
        self._cg_transforms = []

        # MLP
        for block in self._mlp:
            dense, bn = block
            joint = dense.apply(joint)
            # TODO: Is this is the absolutely cleanest way to do it? I am afraid so!
            self._cg_transforms.append(['dropout', self._dropout, joint])
            print("Apply " + str(bn))
            joint = bn.apply(joint)

        pred = self._pred.apply(joint)
        return pred