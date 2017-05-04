"""A dictionary-equipped language model."""
import theano
from theano import tensor

from blocks.bricks import (Initializable, Linear, NDimensionalSoftmax, MLP,
                           Tanh, Rectifier)
from blocks.bricks.base import application
from blocks.bricks.recurrent import LSTM
from blocks.bricks.lookup import LookupTable

from dictlearn.ops import WordToIdOp, RetrievalOp
from dictlearn.aggregation_schemes import Perplexity
from dictlearn.stuff import DebugLSTM
from dictlearn.util import masked_root_mean_square
from dictlearn.lookup import (LSTMReadDefinitions, MeanPoolReadDefinitions,
                              MeanPoolCombiner)



class LanguageModel(Initializable):
    """The dictionary-equipped language model.

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
    retrieval
        The dictionary retrieval algorithm. If `None`, the language model
        does not use any dictionary.
    def_reader: either 'LSTM' or 'mean'
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
    def __init__(self, emb_dim, dim, num_input_words, num_output_words,
                 vocab, retrieval=None,
                 def_reader='LSTM',
                 standalone_def_lookup=True,
                 standalone_def_rnn=True,
                 disregard_word_embeddings=False,
                 compose_type='sum',
                 **kwargs):
        self._num_input_words = num_input_words
        self._num_output_words = num_output_words
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
        self._main_lookup = LookupTable(self._num_input_words, emb_dim, name='main_lookup')
        self._main_fork = Linear(dim, 4 * dim, name='main_fork')
        self._main_rnn = DebugLSTM(dim, name='main_rnn')
        children.extend([self._main_lookup, self._main_fork, self._main_rnn])
        if self._retrieval:
            if standalone_def_lookup:
                lookup = None
            else:
                lookup = self._main_lookup

            if def_reader == 'LSTM':
                if standalone_def_rnn:
                    fork_and_rnn = None
                else:
                    fork_and_rnn = (self._main_fork, self._main_rnn)
                self._def_reader = LSTMReadDefinitions(num_input_words, emb_dim,
                                                       dim, vocab, lookup,
                                                       fork_and_rnn)
            
            elif def_reader == 'mean':
                self._def_reader = MeanPoolReadDefinitions(num_input_words, emb_dim,
                                                           dim, vocab, lookup, 
                                                           translate=True, 
                                                           normalize=False)
            else:
                raise Exception("def reader not understood")

            # TODO: implem FC tanh, FC relu?

            self._combiner = MeanPoolCombiner(
                dim=dim, emb_dim=emb_dim, compose_type=compose_type)

            children.extend([self._def_reader, self._combiner])

        self._pre_softmax = Linear(dim, self._num_output_words)
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
            def_embeddings = self._def_reader.apply(defs, def_mask)

            # TODO(tombosc): are the following ~15 lines useful?
            # Reorder and copy embeddings so that the embeddings of all the definitions
            # that correspond to a position in the text form a continuous span of a tensor
            #def_embeddings = def_embeddings[def_map[:, 2]]

            # Compute the spans corresponding to text positions
            #num_defs = tensor.zeros((1 + words.shape[0] * words.shape[1],), dtype='int32')
            #num_defs = tensor.inc_subtensor(
            #    num_defs[1 + def_map[:, 0] * words.shape[1] + def_map[:, 1]], 1)
            #cum_sum_defs = tensor.cumsum(num_defs)
            #def_spans = tensor.concatenate([cum_sum_defs[:-1, None], cum_sum_defs[1:, None]],
            #                               axis=1)
            #application_call.add_auxiliary_variable(def_spans, name='def_spans')

            ## Mean-pooling of definitions
            #def_sum = tensor.zeros((words.shape[0] * words.shape[1], def_embeddings.shape[1]))
            #def_sum = tensor.inc_subtensor(
            #    def_sum[def_map[:, 0] * words.shape[1] + def_map[:, 1]],
            #    def_embeddings)
            #def_lens = (def_spans[:, 1] - def_spans[:, 0]).astype(theano.config.floatX)
            #def_mean = def_sum / tensor.maximum(def_lens[:, None], 1)
            #def_mean = def_mean.reshape((words.shape[0], words.shape[1], -1))

            # Auxililary variable for debugging
            application_call.add_auxiliary_variable(
                def_embeddings.shape[0], name="num_definitions")
            application_call.add_auxiliary_variable(
                def_embeddings.shape[1], name="max_definition_length")
            #application_call.add_auxiliary_variable(
            #    masked_root_mean_square(def_mean, mask), name='def_mean_rootmean2')

        # shortlisting
        word_ids = self._word_to_id(words)
        input_word_ids = (tensor.lt(word_ids, self._num_input_words) * word_ids
                          + tensor.ge(word_ids, self._num_input_words) * self._vocab.unk)
        output_word_ids = (tensor.lt(word_ids, self._num_output_words) * word_ids
                          + tensor.ge(word_ids, self._num_output_words) * self._vocab.unk)

        # Run the main rnn with combined inputs
        # rnn_inputs should be word_embs TODO
        word_embs = self._main_lookup.apply(input_word_ids)
        application_call.add_auxiliary_variable(
            masked_root_mean_square(word_embs, mask), name='word_emb_RMS')

        if self._retrieval:
            rnn_inputs = self._combiner.apply(word_embs, mask, def_embeddings, def_map)
            #TODO: deal with other args of combiner.apply
            #if self._compose_type == 'sum':
            #    rnn_inputs += def_mean
            #elif self._compose_type.startswith('fully_connected'):
            #    concat = tensor.concatenate([rnn_inputs, def_mean], axis=2)
            #    rnn_inputs = self._def_state_compose.apply(concat)
            #elif self._compose_type.startswith('linear_and_sum'):
            #    rnn_inputs += self._def_state_transform.apply(def_mean)
            #else:
            #    assert False
            #application_call.add_auxiliary_variable(
            #    masked_root_mean_square(rnn_inputs, mask),
            #    name='merged_input_rootmean2')
        else:
            rnn_inputs = word_embs

        application_call.add_auxiliary_variable(
            masked_root_mean_square(word_embs, mask), name='main_rnn_in_RMS')

        #if self._disregard_word_embeddings:
        #    rnn_inputs = def_mean
        main_rnn_states = self._main_rnn.apply(
            tensor.transpose(self._main_fork.apply(rnn_inputs), (1, 0, 2)),
            mask=mask.T)[0]

        # The first token is not predicted
        logits = self._pre_softmax.apply(main_rnn_states[:-1])
        targets = output_word_ids.T[1:]
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
        #last_indices = tensor.max(tensor.arange(targets_mask.shape[0]).reshape((-1,1))*targets_mask, axis=0).astype('int64')
        #batch_indices = tensor.arange(logits.shape[1])
        #last_logits = logits[last_indices, batch_indices]
        #last_predictions = last_logits.argmax(axis=1)
        #last_targets = targets[last_indices, batch_indices]
        #last_correct = tensor.eq(last_predictions, last_targets).astype('int64')
        #application_call.add_auxiliary_variable(
        #    last_correct, name='last_correct')

        return costs
