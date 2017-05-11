"""
ESIM NLI model

TODO: Remove inits
"""
import sys
# 100x larger recursion limit is needed for ESIM
sys.setrecursionlimit(100000)

import theano
import theano.tensor as T
from theano import tensor
import logging
logger = logging.getLogger(__file__)

from blocks.bricks import Initializable, Linear, MLP, BatchNormalizedMLP
from blocks.bricks import Softmax, Rectifier, Sequence, Tanh
from blocks.bricks.bn import BatchNormalization
from blocks.bricks.recurrent import LSTM
from blocks.bricks.base import application, lazy
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent.misc import Bidirectional
from blocks.initialization import IsotropicGaussian, Constant, NdarrayInitialization, Uniform

from dictlearn.inits import GlorotUniform
from dictlearn.theano_util import apply_dropout

class ESIM(Initializable):
    """
    ESIM model based on https://github.com/NYU-MLL/multiNLI/blob/master/python/models/esim.py
    """

    # seq_length, emb_dim, hidden_dim
    def __init__(self, dim, emb_dim, vocab, def_dim=-1, encoder='bilstm', bn=True,
            def_reader=None, def_combiner=None, dropout=0.5, num_input_words=-1,
            # Others
            **kwargs):

        self._dropout = dropout
        self._vocab = vocab
        self._emb_dim = emb_dim
        self._def_reader = def_reader
        self._def_combiner = def_combiner

        if encoder != 'bilstm':
            raise NotImplementedError()

        if def_dim < 0:
            self._def_dim = emb_dim
        else:
            self._def_dim = def_dim

        if num_input_words > 0:
            logger.info("Restricting vocab to " + str(num_input_words))
            self._num_input_words = num_input_words
        else:
            self._num_input_words = vocab.size()

        children = []

        ## Embedding
        self._lookup = LookupTable(self._num_input_words, emb_dim, weights_init=GlorotUniform())
        children.append(self._lookup)

        if def_reader:
            self._final_emb_dim = self._def_dim
            self._def_reader = def_reader
            self._def_combiner = def_combiner
            children.extend([self._def_reader, self._def_combiner])
        else:
            self._final_emb_dim = self._emb_dim

        ## BiLSTM
        _hyp_bidir_fork = Linear(emb_dim, 4 * dim, name='hyp_bidir_fork')
        _hyp_bidir = Bidirectional(LSTM(dim), name='hyp_bidir')
        _prem_bidir_fork = Linear(emb_dim, 4 * dim, name='prem_bidir_fork')
        _prem_bidir = Bidirectional(LSTM(dim), name='prem_bidir')

        self._prem_bilstm = Sequence([_prem_bidir_fork, _prem_bidir], name="prem_bilstm")
        self._hyp_bilstm = Sequence([_hyp_bidir_fork, _hyp_bidir], name="hyp_bilstm")
        children.extend([_hyp_bidir_fork, _hyp_bidir])
        children.extend([_prem_bidir, _prem_bidir_fork])
        children.extend([self._prem_bilstm, self._hyp_bilstm])

        ## BiLSTM no. 2 (encoded attentioned embeddings)
        _hyp_bidir_fork = Linear(8 * dim, 4 * dim, name='hyp_bidir_fork2')
        _hyp_bidir = Bidirectional(LSTM(dim), name='hyp_bidir2')
        _prem_bidir_fork = Linear(8 * dim, 4 * dim, name='prem_bidir_fork2')
        _prem_bidir = Bidirectional(LSTM(dim), name='prem_bidir2')
        self._prem_bilstm2 = Sequence([_prem_bidir_fork, _prem_bidir], name="prem_bilstm2")
        self._hyp_bilstm2 = Sequence([_hyp_bidir_fork, _hyp_bidir], name="hyp_bilstm2")
        children.extend([_hyp_bidir_fork, _hyp_bidir])
        children.extend([_prem_bidir, _prem_bidir_fork])
        children.extend([self._prem_bilstm2, self._hyp_bilstm2])

        self._rnns = [self._prem_bilstm2, self._hyp_bilstm2, self._prem_bilstm, self._hyp_bilstm]

        ## MLP
        if bn:
            self._mlp = BatchNormalizedMLP([Tanh()], [8 * dim, dim], conserve_memory=False, name="mlp")
            self._pred = BatchNormalizedMLP([Softmax()], [dim, 3],  conserve_memory=False, name="pred_mlp")
        else:
            self._mlp = MLP([Tanh()], [8 * dim, dim], name="mlp")
            self._pred = MLP([Softmax()], [dim, 3], name="pred_mlp")

        children.append(self._mlp)
        children.append(self._pred)

        super(ESIM, self).__init__(children=children, **kwargs)

    def get_embeddings_lookups(self):
        return [self._lookup]

    def set_embeddings(self, embeddings):
        self._lookup.parameters[0].set_value(embeddings.astype(theano.config.floatX))

    @application
    def apply(self, application_call,
            s1_preunk, s1_mask, s2_preunk, s2_mask, def_mask=None,
            defs=None, s1_def_map=None, s2_def_map=None, train_phase=True):
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

            s1_emb = self._def_combiner.apply(
                s1_emb, s1_mask,
                def_embs, s1_def_map, word_ids=s1, train_phase=train_phase, call_name="s1")

            s2_emb = self._def_combiner.apply(
                s2_emb, s2_mask,
                def_embs, s2_def_map, word_ids=s2, train_phase=train_phase, call_name="s2")
        else:
            application_call.add_auxiliary_variable(
                1*s1_emb,
                name='s1_word_embeddings')
            if train_phase and self._dropout > 0:
                s1_emb = apply_dropout(s1_emb, drop_prob=self._dropout)
                s2_emb = apply_dropout(s2_emb, drop_prob=self._dropout)

        ### Encode ###

        # TODO: Share this bilstm?
        s1_bilstm, _ = self._prem_bilstm.apply(s1_emb) # (batch_size, n_seq, 2 * dim)
        s2_bilstm, _ = self._hyp_bilstm.apply(s2_emb) # (batch_size, n_seq, 2 * dim)

        ### Attention ###

        # Compute E matrix (eq. 11)
        # E_ij = <s1[i], s2[j]>
        # each call computes E[i, :]
        def compute_e_row(s2_i, s1_bilstm, s1_mask):
            b_size = s1_bilstm.shape[0]
            # s2_i is (batch_size, emb_dim)
            # s1_bilstm is (batch_size, seq_len, emb_dim)
            # s1_mask is (batch_size, seq_len)
            s2_i = s2_i.reshape((s2_i.shape[0], s2_i.shape[1], 1))
            s2_i = T.repeat(s2_i, 2, axis=2)
            # s2_i is (batch_size, emb_dim, 1)
            score = T.batched_dot(s1_bilstm, s2_i) # (batch_size, seq_len, 1)
            score = score[:, :, 0].reshape((b_size, -1)) # (batch_size, seq_len)

            score = theano.tensor.nnet.softmax(s1_mask * score)
            return score # E[i, :]

        # NOTE: No point in masking here
        E, _ = theano.scan(compute_e_row, sequences=[s1_bilstm.transpose(1, 0, 2)],
            non_sequences=[s2_bilstm, s2_mask])
        # (seq_len, batch_size, seq_len)
        E = E.dimshuffle(1, 0, 2)
        assert E.ndim == 3
        # (batch_size, seq_len, seq_len)

        ### Compute tilde vectors (eq. 12 and 13) ###

        def compute_tilde_vector(e_i, s, s_mask):
            # e_i is (batch_size, seq_len)
            # s_tilde_i = \sum e_ij b_j, (batch_size, seq_len)
            s_tilde_i = (e_i.dimshuffle(0, 1, "x") * (s * s_mask.dimshuffle(0, 1, "x"))).sum(axis=1)
            return s_tilde_i

        # (batch_size, seq_len, def_dim)
        s1_tilde, _ = theano.scan(compute_tilde_vector,
            sequences=[E.dimshuffle(1, 0, 2)], non_sequences=[s2_bilstm, s2_mask])
        s1_tilde = s1_tilde.dimshuffle(1, 0, 2)
        s2_tilde, _ = theano.scan(compute_tilde_vector,
            sequences=[E.dimshuffle(2, 0, 1)], non_sequences=[s1_bilstm, s1_mask])
        s2_tilde = s2_tilde.dimshuffle(1, 0, 2)

        ### Compose (eq. 14 and 15) ###

        # (batch_size, seq_len, 8 * dim)
        s1_comp = T.concatenate([s1_bilstm, s1_tilde, s1_bilstm - s1_tilde, s1_bilstm * s1_tilde], axis=2)
        s2_comp = T.concatenate([s2_bilstm, s2_tilde, s2_bilstm - s2_tilde, s2_bilstm * s2_tilde], axis=2)

        ### Encode (eq. 16 and 17) ###

        # (batch_size, seq_len, 8 * dim)
        # TODO: Share this bilstm?
        s1_comp_bilstm, _ = self._prem_bilstm2.apply(s1_comp)  # (batch_size, n_seq, 2 * dim)
        s2_comp_bilstm, _ = self._hyp_bilstm2.apply(s2_comp)  # (batch_size, n_seq, 2 * dim)

        ### Pooling Layer ###

        s1_comp_bilstm_ave = T.mean((s1_mask.dimshuffle(0, 1, "x") * s1_comp_bilstm), axis=1)
        s1_comp_bilstm_max = T.max((s1_mask.dimshuffle(0, 1, "x") * s1_comp_bilstm), axis=1)

        s2_comp_bilstm_ave = T.mean((s2_mask.dimshuffle(0, 1, "x") * s2_comp_bilstm), axis=1)
        # (batch_size, dim)
        s2_comp_bilstm_max = T.max((s2_mask.dimshuffle(0, 1, "x") * s2_comp_bilstm), axis=1)

        ### Final classifier ###

        # MLP layer
        # (batch_size, 8 * dim)
        m = T.concatenate([s1_comp_bilstm_ave, s1_comp_bilstm_max, s2_comp_bilstm_ave, s2_comp_bilstm_max], axis=1)
        pre_logits = self._mlp.apply(m)

        if train_phase:
            pre_logits = apply_dropout(pre_logits, drop_prob=self._dropout)

        # Get prediction
        self.logits = self._pred.apply(pre_logits)

        return self.logits


