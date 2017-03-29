"""A dictionary-equipped language model."""
import theano
from theano import tensor

from blocks.bricks import Initializable, Linear, NDimensionalSoftmax, MLP, Tanh, Rectifier
from blocks.bricks.base import application
from blocks.bricks.recurrent import LSTM
from blocks.bricks.lookup import LookupTable

from dictlearn.ops import WordToIdOp, RetrievalOp
from dictlearn.aggregation_schemes import Perplexity


def masked_root_mean_square(x, mask):
    """Masked room mean square for a 3D tensor"""
    return (((x * mask[:, :, None]) ** 2).sum() / x.shape[2] / mask.sum()) ** 0.5


class LanguageModel(Initializable):
    """The dictionary-equipped language model.

    Parameters
    ----------
    vocab
        The vocabulary object.
    retrieval
        The dictionary retrieval algorithm. If `None`, the language model
        does not use any dictionary.
    dim : int
        The default dimension for the components.
    standalone_def_rnn : bool
        If `True`, a standalone RNN with separate word embeddings is used
        to embed definition. If `False` the language model is reused.
    disregard_word_embeddings : bool
        If `True`, the word embeddings are not used, only the information
        from the definitions is used.
    compose_type : str
        If 'sum', the definition and word embeddings are averaged
        If 'fully_connected_linear', a learned perceptron compose the 2
        embeddings linearly
        If 'fully_connected_relu', ...
        If 'fully_connected_tanh', ...

    """
    def __init__(self, dim, vocab, retrieval=None,
                 standalone_def_rnn=True,
                 disregard_word_embeddings=False,
                 compose_type='sum',
                 **kwargs):
        self._vocab = vocab
        self._retrieval = retrieval
        self._disregard_word_embeddings = disregard_word_embeddings
        self._compose_type = compose_type

        self._word_to_id = WordToIdOp(self._vocab)

        if self._retrieval:
            self._retrieve = RetrievalOp(retrieval)

        # Dima: we can have slightly less copy-paste here if we
        # copy the RecurrentFromFork class from my other projects.
        children = []
        self._main_lookup = LookupTable(vocab.size(), dim, name='main_lookup')
        self._main_fork = Linear(dim, 4 * dim, name='main_fork')
        self._main_rnn = LSTM(dim, name='main_rnn')
        children.extend([self._main_lookup, self._main_fork, self._main_rnn])
        if self._retrieval:
            if standalone_def_rnn:
                self._def_lookup = LookupTable(vocab.size(), dim, name='def_lookup')
                self._def_fork = Linear(dim, 4 * dim, name='def_fork')
                self._def_rnn = LSTM(dim, name='def_rnn')
                children.extend([self._def_lookup, self._def_fork, self._def_rnn])
            else:
                self._def_lookup = self._main_lookup
                self._def_fork = self._main_fork
                self._def_rnn = self._main_rnn
        if compose_type == 'fully_connected_tanh':
            self._def_state_compose = MLP(activations=[Tanh(name="def_state_compose")], dims=[2*dim, dim])
            children.append(self._def_state_compose)
        elif compose_type == 'fully_connected_relu':
            self._def_state_compose = MLP(activations=[Rectifier(name="def_state_compose")],
                                          dims=[2*dim, dim])
            children.append(self._def_state_compose)
        elif compose_type == 'fully_connected_linear':
            self._def_state_compose = MLP(activations=[None],
                                          dims=[2*dim, dim])
            children.append(self._def_state_compose)
        elif compose_type == 'sum':
            pass
        elif not disregard_word_embeddings:
            raise Exception("Error: composition of embeddings and def not understood")

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
        if self._retrieval:
            defs, def_mask, def_map = self._retrieve(words)
            embedded_def_words = self._def_lookup.apply(defs)
            def_embeddings = self._def_rnn.apply(
                tensor.transpose(self._def_fork.apply(embedded_def_words), (1, 0, 2)),
                mask=def_mask.T)[0][-1]
            # Reorder and copy embeddings so that the embeddings of all the definitions
            # that correspond to a position in the text form a continuous span of a tensor
            def_embeddings = def_embeddings[def_map[:, 2]]

            # Compute the spans corresponding to text positions
            num_defs = tensor.zeros((1 + words.shape[0] * words.shape[1],), dtype='int32')
            num_defs = tensor.inc_subtensor(
                num_defs[1 + def_map[:, 0] * words.shape[1] + def_map[:, 1]], 1)
            cum_sum_defs = tensor.cumsum(num_defs)
            def_spans = tensor.concatenate([cum_sum_defs[:-1, None], cum_sum_defs[1:, None]],
                                           axis=1)
            application_call.add_auxiliary_variable(def_spans, name='def_spans')

            # Mean-pooling of definitions
            def_sum = tensor.zeros((words.shape[0] * words.shape[1], def_embeddings.shape[1]))
            def_sum = tensor.inc_subtensor(
                def_sum[def_map[:, 0] * words.shape[1] + def_map[:, 1]],
                def_embeddings)
            def_lens = (def_spans[:, 1] - def_spans[:, 0]).astype(theano.config.floatX)
            def_mean = def_sum / tensor.maximum(def_lens[:, None], 1)
            def_mean = def_mean.reshape((words.shape[0], words.shape[1], -1))

            # Auxililary variable for debugging
            application_call.add_auxiliary_variable(
                defs.shape[0], name="num_definitions")
            application_call.add_auxiliary_variable(
                defs.shape[1], name="max_definition_length")
            application_call.add_auxiliary_variable(
                masked_root_mean_square(def_mean, mask), name='def_mean_rootmean2')

        # Run the main rnn with combined inputs
        word_ids = self._word_to_id(words)
        rnn_inputs = self._main_lookup.apply(word_ids)
        application_call.add_auxiliary_variable(
            masked_root_mean_square(rnn_inputs, mask), name='rnn_input_rootmean2')
        if self._retrieval:
            if self._compose_type == 'sum':
                rnn_inputs += def_mean
            elif self._compose_type.startswith('fully_connected'):
                concat = tensor.concatenate([rnn_inputs, def_mean], axis=2)
                rnn_inputs = self._def_state_compose.apply(concat)
            else:
                assert False
        if self._disregard_word_embeddings:
            rnn_inputs = def_mean
        main_rnn_states = self._main_rnn.apply(
            tensor.transpose(self._main_fork.apply(rnn_inputs), (1, 0, 2)),
            mask=mask.T)[0]

        # The first token is not predicted
        logits = self._pre_softmax.apply(main_rnn_states[:-1])
        targets = word_ids.T[1:]
        targets_mask = mask.T[1:]
        minus_logs = self._softmax.categorical_cross_entropy(
            targets, logits, extra_ndim=1)
        costs = (minus_logs * targets_mask).sum(axis=0)
        perplexity = tensor.exp(costs.sum() / targets_mask.sum())
        perplexity.tag.aggregation_scheme = Perplexity(
            costs.sum(), targets_mask.sum())
        application_call.add_auxiliary_variable(
            perplexity, name='perplexity')

        # Analyze predictions
        last_indices = (targets_mask.sum(axis=0) - 1).astype('int64')
        batch_indices = tensor.arange(logits.shape[1])
        last_logits = logits[last_indices, batch_indices]
        last_predictions = last_logits.argmax(axis=1)
        last_targets = targets[last_indices, batch_indices]
        last_correct = tensor.eq(last_predictions, last_targets).astype('int64')
        application_call.add_auxiliary_variable(
            last_correct, name='last_correct')

        return costs
