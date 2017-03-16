"""A dictionary-equipped language model."""
import theano
from theano import tensor

from blocks.bricks import Initializable, Linear, NDimensionalSoftmax
from blocks.bricks.base import application
from blocks.bricks.recurrent import LSTM
from blocks.bricks.lookup import LookupTable

from dictlearn.ops import WordToIdOp, RetrievalOp


class LanguageModel(Initializable):
    """The dictionary-equipped language model.

    Parameters
    ----------
    vocab
        The vocabulary object.
    dict_
        The dictionary object. If `None`, the language model
        does not use any dictionary.
    dim : int
        The default dimension for the components.

    """
    def __init__(self, dim, vocab, dict_=None, **kwargs):
        self._vocab = vocab
        self._dict = dict_
        self._word_to_id = WordToIdOp(self._vocab)

        if self._dict:
            self._retrieve = RetrievalOp(self._vocab, self._dict)

        # Dima: we can have slightly less copy-paste here if we
        # copy the RecurrentFromFork class from my other projects.
        children = []
        self._main_lookup = LookupTable(vocab.size(), dim, name='main_lookup')
        self._main_fork = Linear(dim, 4 * dim, name='main_fork')
        self._main_rnn = LSTM(dim, name='main_rnn')
        children.extend([self._main_lookup, self._main_fork, self._main_rnn])
        if self._dict:
            self._def_lookup = LookupTable(vocab.size(), dim, name='def_lookup')
            self._def_fork = Linear(dim, 4 * dim, name='def_fork')
            self._def_rnn = LSTM(dim, name='def_rnn')
            children.extend([self._def_lookup, self._def_fork, self._def_rnn])
        self._pre_softmax = Linear(dim, vocab.size())
        self._softmax = NDimensionalSoftmax()
        children.extend([self._pre_softmax, self._softmax])

        super(LanguageModel, self).__init__(children=children, **kwargs)

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
        if self._dict:
            defs, def_mask, def_map = self._retrieve(words)
            embedded_def_words = self._def_lookup.apply(defs)
            def_embeddings = self._def_rnn.apply(
                tensor.transpose(self._def_fork.apply(embedded_def_words), (1, 0, 2)),
                mask=def_mask.T)[0][-1]
            # Reorder and copy embeddings so that the embeddings of all the definitions
            # that correspond to a position in the text form a continuous span of a tensor
            def_embeddings = def_embeddings[def_map[2]]

            # Compute the spans corresponding to text positions
            num_defs = tensor.zeros((1 + words.shape[0] * words.shape[1],), dtype='int64')
            num_defs = tensor.inc_subtensor(
                num_defs[1 + def_map[:, 0] * words.shape[1] + def_map[:, 1]], 1)
            cum_sum_defs = tensor.cumsum(num_defs)
            def_spans = tensor.concatenate([cum_sum_defs[:-1, None], cum_sum_defs[1:, None]],
                                           axis=1)
            application_call.add_auxiliary_variable(def_spans, name='def_spans')

            # Mean-pooling of definitions
            def_sum = tensor.zeros((words.shape[0] * words.shape[1], def_embeddings.shape[1]))
            def_sum = tensor.inc_subtensor(def_sum[def_map[0] * words.shape[1] + def_map[1]],
                                        def_embeddings)
            def_lens = (def_spans[:, 1] - def_spans[:, 0]).astype(theano.config.floatX)
            def_mean = def_sum / def_lens[:, None]
            def_mean = def_mean.reshape((words.shape[0], words.shape[1], -1))

        # Run the main rnn with combined inputs
        word_ids = self._word_to_id(words)
        rnn_inputs = self._main_lookup.apply(word_ids)
        if self._dict:
            rnn_inputs += def_mean
        main_rnn_states = self._main_rnn.apply(
            tensor.transpose(self._main_fork.apply(rnn_inputs), (1, 0, 2)),
            mask=mask.T)[0]

        # The first token is not predicted
        logits = self._pre_softmax.apply(main_rnn_states[:-1])
        predictions = logits.argmax(axis=2)
        targets = word_ids.T[1:]
        targets_mask = mask.T[1:]
        minus_logs = self._softmax.categorical_cross_entropy(
            targets, logits, extra_ndim=1)
        costs = (minus_logs * targets_mask).sum(axis=0)

        # Analyze predictions
        correct = tensor.eq(predictions, targets)
        last_correct = correct[(targets_mask.sum(axis=0) - 1).astype('int64'),
                                tensor.arange(correct.shape[1])]
        application_call.add_auxiliary_variable(
            last_correct, name='last_correct')

        return costs


