"""
ESIM NLI model
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
from blocks.bricks.recurrent.misc import Bidirectional
from blocks.initialization import IsotropicGaussian, Constant, NdarrayInitialization, Uniform

from dictlearn.inits import GlorotUniform
from dictlearn.lookup import MeanPoolCombiner, LSTMReadDefinitions, MeanPoolReadDefinitions
from dictlearn.util import apply_dropout

class ESIM(Initializable):
    """
    ESIM model based on https://github.com/NYU-MLL/multiNLI/blob/master/python/models/esim.py
    """

    def __init__(self, dim, mlp_dim, translate_dim, emb_dim, vocab,
            def_reader=None, def_combiner=None, dropout=0.5,
            # Others
            **kwargs):

        self._dropout = dropout
        self._vocab = vocab
        self._retrieval = retrieval
        self._only_def = disregard_word_embeddings

        if reader_type not in {"rnn", "mean"}:
            raise NotImplementedError("Not implemented " + reader_type)

        if num_input_words > 0:
            logger.info("Restricting vocab to " + str(num_input_words))
            self._num_input_words = num_input_words
        else:
            self._num_input_words = vocab.size()

        children = []

        if def_reader:
            self._def_reader = def_reader
            self._combiner = def_combiner
            children.extend([self._def_reader, self._combiner])

        # BiLSTM
        self._bidir_fork = Linear(emb_dim, 4 * dim, name='bidir_fork')
        self._bidir = Bidirectional(LSTM(dim), name='bidir')
        children.extend([self._bidir_fork, self._bidir])

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

        super(ESIM, self).__init__(children=children, **kwargs)

    def get_embeddings_lookups(self):
        if not self._retrieval:
            return [self._lookup]
        elif self._retrieval and not self._only_def:
            return [self._lookup, self._def_reader._def_lookup]
        elif self._retrieval and self._only_def:
            return [self._def_reader._def_lookup]
        else:
            raise NotImplementedError()

    def set_embeddings(self, embeddings):
        if not self._retrieval:
            self._lookup.parameters[0].set_value(embeddings.astype(theano.config.floatX))
        elif self._retrieval and not self._only_def:
            self._lookup.parameters[0].set_value(embeddings.astype(theano.config.floatX))
            self._def_reader._def_lookup.parameters[0].set_value(embeddings.astype(theano.config.floatX))
        else:
            raise NotImplementedError()

    def embeddings_var(self):
        if not self._retrieval:
            return [self._lookup.parameters[0]]
        elif self._retrieval and not self._only_def:
            return [self._lookup.parameters[0], self._def_reader._def_lookup.parameters[0]]
        else:
            raise NotImplementedError()

    @application
    def apply(self, application_call,
            s1_preunk, s1_mask, s2_preunk, s2_mask, def_mask=None,
            defs=None, s1_def_map=None, s2_def_map=None, train_phase=True):

        ### First biLSTM layer ###

        # Shortlist words (sometimes we want smaller vocab, especially when dict is small)
        s1 = (tensor.lt(s1_preunk, self._num_input_words) * s1_preunk
              + tensor.ge(s1_preunk, self._num_input_words) * self._vocab.unk)
        s2 = (tensor.lt(s2_preunk, self._num_input_words) * s2_preunk
              + tensor.ge(s2_preunk, self._num_input_words) * self._vocab.unk)

        ### Embed ###
        s1_emb = self._lookup.apply(s1)
        s2_emb = self._lookup.apply(s2)

        if self._def_reader:
            assert defs is not None

            def_embs = self._def_reader.apply(defs, def_mask)

            s1_emb = self._combiner.apply(
                s1_emb, s1_mask,
                def_embs, s1_def_map, word_ids=s1, train_phase=train_phase, call_name="s1")

            s2_emb = self._combiner.apply(
                s2_emb, s2_mask,
                def_embs, s2_def_map, word_ids=s2, train_phase=train_phase, call_name="s2")
        else:
            if train_phase and self._dropout > 0:
                s1_emb = apply_dropout(s1_emb, drop_prob=self._dropout)
                s2_emb = apply_dropout(s2_emb, drop_prob=self._dropout)

        ### biLSTM ###

        premise_outs, c1 = blocks.biLSTM(premise_in, dim=self.dim, seq_len=prem_seq_lengths, name='premise')
        hypothesis_outs, c2 = blocks.biLSTM(hypothesis_in, dim=self.dim, seq_len=hyp_seq_lengths, name='hypothesis')

        premise_bi = tf.concat(premise_outs, axis=2)
        hypothesis_bi = tf.concat(hypothesis_outs, axis=2)

        premise_list = tf.unstack(premise_bi, axis=1)
        hypothesis_list = tf.unstack(hypothesis_bi, axis=1)

        ### Attention ###

        scores_all = []
        premise_attn = []
        alphas = []

        for i in range(self.sequence_length):

            scores_i_list = []
            for j in range(self.sequence_length):
                score_ij = tf.reduce_sum(tf.multiply(premise_list[i], hypothesis_list[j]), 1, keep_dims=True)
                scores_i_list.append(score_ij)

            scores_i = tf.stack(scores_i_list, axis=1)
            alpha_i = blocks.masked_softmax(scores_i, mask_hyp)
            a_tilde_i = tf.reduce_sum(tf.multiply(alpha_i, hypothesis_bi), 1)
            premise_attn.append(a_tilde_i)

            scores_all.append(scores_i)
            alphas.append(alpha_i)

        scores_stack = tf.stack(scores_all, axis=2)
        scores_list = tf.unstack(scores_stack, axis=1)

        hypothesis_attn = []
        betas = []
        for j in range(self.sequence_length):
            scores_j = scores_list[j]
            beta_j = blocks.masked_softmax(scores_j, mask_prem)
            b_tilde_j = tf.reduce_sum(tf.multiply(beta_j, premise_bi), 1)
            hypothesis_attn.append(b_tilde_j)

            betas.append(beta_j)

        # Make attention-weighted sentence representations into one tensor,
        premise_attns = tf.stack(premise_attn, axis=1)
        hypothesis_attns = tf.stack(hypothesis_attn, axis=1)

        # For making attention plots,
        self.alpha_s = tf.stack(alphas, axis=2)
        self.beta_s = tf.stack(betas, axis=2)

        ### Subcomponent Inference ###

        prem_diff = tf.subtract(premise_bi, premise_attns)
        prem_mul = tf.multiply(premise_bi, premise_attns)
        hyp_diff = tf.subtract(hypothesis_bi, hypothesis_attns)
        hyp_mul = tf.multiply(hypothesis_bi, hypothesis_attns)

        m_a = tf.concat([premise_bi, premise_attns, prem_diff, prem_mul], 2)
        m_b = tf.concat([hypothesis_bi, hypothesis_attns, hyp_diff, hyp_mul], 2)

        ### Inference Composition ###

        v1_outs, c3 = blocks.biLSTM(m_a, dim=self.dim, seq_len=prem_seq_lengths, name='v1')
        v2_outs, c4 = blocks.biLSTM(m_b, dim=self.dim, seq_len=hyp_seq_lengths, name='v2')

        v1_bi = tf.concat(v1_outs, axis=2)
        v2_bi = tf.concat(v2_outs, axis=2)

        ### Pooling Layer ###

        v_1_sum = tf.reduce_sum(v1_bi, 1)
        v_1_ave = tf.div(v_1_sum, tf.expand_dims(tf.cast(prem_seq_lengths, tf.float32), -1))

        v_2_sum = tf.reduce_sum(v2_bi, 1)
        v_2_ave = tf.div(v_2_sum, tf.expand_dims(tf.cast(hyp_seq_lengths, tf.float32), -1))

        v_1_max = tf.reduce_max(v1_bi, 1)
        v_2_max = tf.reduce_max(v2_bi, 1)

        v = tf.concat([v_1_ave, v_2_ave, v_1_max, v_2_max], 1)

        # MLP layer
        h_mlp = tf.nn.tanh(tf.matmul(v, self.W_mlp) + self.b_mlp)

        # Dropout applied to classifier
        h_drop = tf.nn.dropout(h_mlp, self.keep_rate_ph)

        # Get prediction
        self.logits = tf.matmul(h_drop, self.W_cl) + self.b_cl



