"""A seq2seq model"""
import theano
from theano import tensor

from blocks.bricks import (Initializable, Linear, NDimensionalSoftmax, MLP,
                           Tanh, Rectifier)
from blocks.bricks.base import application
from blocks.bricks.recurrent import LSTM
from blocks.bricks.lookup import LookupTable
from blocks.initialization import Constant

from dictlearn.ops import WordToIdOp
from dictlearn.aggregation_schemes import Perplexity
from dictlearn.util import masked_root_mean_square

class Seq2Seq(Initializable):
    """ seq2seq model

    Parameters
    ----------
    emb_dim: int
        The dimension of word embeddings (including for def model if standalone)
    dim : int
        The dimension of the RNNs states (including for def model if standalone)
    num_input_words : int
        The size of the LM's input vocabulary.
    num_output_words : int
        The size of the LM's output vocabulary.
    vocab
        The vocabulary object.
    """
    def __init__(self, emb_dim, dim, num_input_words, 
                 num_output_words, vocab, 
                 **kwargs):
        if emb_dim == 0:
            emb_dim = dim
        if num_input_words == 0:
            num_input_words = vocab.size()
        if num_output_words == 0:
            num_output_words = vocab.size()

        self._num_input_words = num_input_words
        self._num_output_words = num_output_words
        self._vocab = vocab

        self._word_to_id = WordToIdOp(self._vocab)

        children = []

        self._main_lookup = LookupTable(self._num_input_words, emb_dim, name='main_lookup')
        self._encoder_fork = Linear(emb_dim, 4 * dim, name='encoder_fork')
        self._encoder_rnn = LSTM(dim, name='encoder_rnn')
        self._decoder_fork = Linear(emb_dim, 4 * dim, name='decoder_fork')
        self._decoder_rnn = LSTM(dim, name='decoder_rnn')
        children.extend([self._main_lookup,
                         self._encoder_fork, self._encoder_rnn,
                         self._decoder_fork, self._decoder_rnn])
        self._pre_softmax = Linear(dim, self._num_output_words)
        self._softmax = NDimensionalSoftmax()
        children.extend([self._pre_softmax, self._softmax])

        super(LanguageModel, self).__init__(children=children, **kwargs)

    def set_def_embeddings(self, embeddings):
        self._main_lookup.parameters[0].set_value(embeddings.astype(theano.config.floatX))

    def get_def_embeddings_params(self):
        return self._main_lookup.parameters[0]

    def add_perplexity_measure(self, application_call, minus_logs, mask, name):
        costs = (minus_logs * mask).sum(axis=0)
        perplexity = tensor.exp(costs.sum() / mask.sum())
        perplexity.tag.aggregation_scheme = Perplexity(
            costs.sum(), mask.sum())
        application_call.add_auxiliary_variable(perplexity, name=name)
        return costs

    @application
    def apply(self, application_call, words, mask):
        """Compute the log-likelihood for a batch of sequences.

        words
            An integer matrix of shape (B, T), where T is the number of time
            step, B is the batch size. Note that this order of the axis is
            different from what all RNN bricks consume, hence and the axis
            should be transposed at some point.
        mask
            A float32 matrix of shape (B, T). Zeros indicate the padding.

        """
        word_ids = self._word_to_id(words)

        # shortlisting
        input_word_ids = (tensor.lt(word_ids, self._num_input_words) * word_ids
                          + tensor.ge(word_ids, self._num_input_words) * self._vocab.unk)
        output_word_ids = (tensor.lt(word_ids, self._num_output_words) * word_ids
                          + tensor.ge(word_ids, self._num_output_words) * self._vocab.unk)

        application_call.add_auxiliary_variable(
            unk_ratio(input_word_ids, mask, self._vocab.unk),
            name='unk_ratio')

        # Run the main rnn with combined inputs
        rnn_inputs = self._main_lookup.apply(input_word_ids)

        encoder_rnn_states = self._encoder_rnn.apply(
            tensor.transpose(self._encoder_fork.apply(rnn_inputs), (1, 0, 2)),
            mask=mask.T)[0]

        # The first token is not predicted
        logits = self._pre_softmax.apply(main_rnn_states[:-1])
        targets = output_word_ids.T[1:]
        out_softmax = self._softmax.apply(logits, extra_ndim=1)
        application_call.add_auxiliary_variable(
                out_softmax.copy(), name="proba_out")
        minus_logs = self._softmax.categorical_cross_entropy(
            targets, logits, extra_ndim=1)

        targets_mask = mask.T[1:]
        costs = self.add_perplexity_measure(application_call, minus_logs,
                               targets_mask,
                               "perplexity")

        missing_embs = tensor.eq(input_word_ids, self._vocab.unk).astype('int32') # (bs, L)
        self.add_perplexity_measure(application_call, minus_logs,
                               targets_mask * missing_embs.T[:-1],
                               "perplexity_after_mis_word_embs")
        self.add_perplexity_measure(application_call, minus_logs,
                               targets_mask * (1-missing_embs.T[:-1]),
                               "perplexity_after_word_embs")

        word_counts = self._word_to_count(words)
        very_rare_masks = []
        for threshold in self._very_rare_threshold:
            very_rare_mask = tensor.lt(word_counts, threshold).astype('int32')
            very_rare_mask = targets_mask * (very_rare_mask.T[:-1])
            very_rare_masks.append(very_rare_mask)
            self.add_perplexity_measure(application_call, minus_logs,
                                   very_rare_mask,
                                   "perplexity_after_very_rare_" + str(threshold))

        if self._retrieval:
            has_def = tensor.zeros_like(output_word_ids)
            has_def = tensor.inc_subtensor(has_def[def_map[:,0], def_map[:,1]], 1)
            mask_targets_has_def = has_def.T[:-1] * targets_mask # (L-1, bs)
            self.add_perplexity_measure(application_call, minus_logs,
                                   mask_targets_has_def,
                                   "perplexity_after_def_embs")

            for thresh, very_rare_mask in zip(self._very_rare_threshold, very_rare_masks):
                self.add_perplexity_measure(application_call, minus_logs,
                                   very_rare_mask * mask_targets_has_def,
                                   "perplexity_after_def_very_rare_" + str(thresh))

            application_call.add_auxiliary_variable(
                    mask_targets_has_def.T, name='mask_def_emb')

        return costs, updates
