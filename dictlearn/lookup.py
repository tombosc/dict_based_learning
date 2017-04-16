"""
Methods of constructing word embeddings

TODO(kudkudak): Add unit test for it
TODO(kudkudak): Refactor (non-dict) lookup from snli_baseline_model to a class here (name for instance EnchancedLookup)
"""
from blocks.bricks import Initializable, Linear, MLP, Tanh, Rectifier
from blocks.bricks.base import application
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import LSTM
from blocks.initialization import Uniform

import theano
import theano.tensor as T

from dictlearn.util import masked_root_mean_square
from dictlearn.ops import RetrievalOp

# TODO: Implement
class LookupWithTranslation(Initializable):
    pass

class DictEnchancedLookup(Initializable):
    """
    Helper function for more powerful Lookup tables.
    
    The future of NLP!

    Parameters
    ----------
    translate_dim: int, default: -1
        Working with raw embeddings is hard, if set to >0 will transform any embedding going into
        model or out of lookup table to translate_dim

    num_input_words: int, default: -1
        If non zero will (a bit confusing name) restrict dynamically vocab. 
        WARNING: it assumes word ids are monotonical with frequency!

    emb_dim: int
        Dimensionality of word embeddings

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

    def __init__(self, emb_dim, vocab, retrieval, translate_dim=-1, disregard_word_embeddings=False,
            num_input_words=-1, compose_type="sum", **kwargs):

        # Probably we don't want these assers
        assert "weights_init" in kwargs
        assert "biases_init" in kwargs

        self._retrieval = retrieval
        
        if num_input_words > 0:
            self._num_input_words = num_input_words
        else:
            self._num_input_words = vocab.size()
            
        self._compose_type = compose_type
        self._emb_dim = emb_dim
        self._disregard_word_embeddings = disregard_word_embeddings
        self._translate_dim = translate_dim

        if self._retrieval:
            self._retrieve = RetrievalOp(retrieval)

        if self.translate_dim == -1:
            dim = emb_dim
        else:
            dim = translate_dim

        if self._disregard_word_embeddings:
            raise NotImplementedError()

        children = []

        self._base_lookup = LookupTable(self._num_input_words, emb_dim)
        children.append(self._base_lookup)

        if self.translate_dim > 0:
            self._translate = Linear(emb_dim, translate_dim, name="lookup_translate")

        self._def_lookup = LookupTable(self._num_input_words, dim, name='def_lookup')
        self._def_fork = Linear(dim, 4 * emb_dim, name='def_fork')
        # TODO(kudkudak): Better LSTM weight init
        self._def_rnn = LSTM(dim, name='def_rnn', weights_init=Uniform(width=0.1))
        children.extend([self._def_lookup, self._def_fork, self._def_rnn])

        if compose_type == 'fully_connected_tanh':
            self._def_state_compose = MLP(activations=[Tanh(name="def_state_compose")], dims=[2 * dim, dim])
            children.append(self._def_state_compose)
        elif compose_type == 'fully_connected_relu':
            self._def_state_compose = MLP(activations=[Rectifier(name="def_state_compose")],
                dims=[2 * dim, dim])
            children.append(self._def_state_compose)
        elif compose_type == 'fully_connected_linear':
            self._def_state_compose = MLP(activations=[None],
                dims=[2 * dim, dim])
            children.append(self._def_state_compose)
        elif compose_type == 'linear_and_sum':
            self._def_state_transform = Linear(dim, dim)
            children.append(self._def_state_transform)
        elif compose_type == 'sum':
            pass
        elif not disregard_word_embeddings:
            raise Exception("Error: composition of embeddings and def not understood")

        super(DictEnchancedLookup, self).__init__(children=children, **kwargs)

    def embeddings_var(self):
        return self._base_lookup.parameters[0]

    @application
    def apply(self, application_call, words, words_mask):
        """
        Returns vector per each word in sequence using the dictionary based lookup
        """
        defs, def_mask, def_map = self._retrieve(words)

        # Short listing
        defs = (T.lt(defs, self._num_input_words) * defs
                + T.ge(defs, self._num_input_words) * self._vocab.unk)

        embedded_def_words = self._def_lookup.apply(defs)
        def_embeddings = self._def_rnn.apply(
            T.transpose(self._def_fork.apply(embedded_def_words), (1, 0, 2)),
            mask=def_mask.T)[0][-1]
        # Reorder and copy embeddings so that the embeddings of all the definitions
        # that correspond to a position in the text form a continuous span of a T
        def_embeddings = def_embeddings[def_map[:, 2]]

        # Compute the spans corresponding to text positions
        num_defs = T.zeros((1 + words.shape[0] * words.shape[1],), dtype='int32')
        num_defs = T.inc_subtensor(
            num_defs[1 + def_map[:, 0] * words.shape[1] + def_map[:, 1]], 1)
        cum_sum_defs = T.cumsum(num_defs)
        def_spans = T.concatenate([cum_sum_defs[:-1, None], cum_sum_defs[1:, None]],
            axis=1)
        application_call.add_auxiliary_variable(def_spans, name='def_spans')

        # Mean-pooling of definitions
        def_sum = T.zeros((words.shape[0] * words.shape[1], def_embeddings.shape[1]))
        def_sum = T.inc_subtensor(
            def_sum[def_map[:, 0] * words.shape[1] + def_map[:, 1]],
            def_embeddings)
        def_lens = (def_spans[:, 1] - def_spans[:, 0]).astype(theano.config.floatX)
        def_mean = def_sum / T.maximum(def_lens[:, None], 1)
        def_mean = def_mean.reshape((words.shape[0], words.shape[1], -1))

        # Auxililary variable for debugging
        application_call.add_auxiliary_variable(
            defs.shape[0], name="num_definitions")
        application_call.add_auxiliary_variable(
            defs.shape[1], name="max_definition_length")
        application_call.add_auxiliary_variable(
            masked_root_mean_square(def_mean, words_mask), name='def_mean_rootmean2')

        # Shortlisting
        word_ids = self._word_to_id(words)
        input_word_ids = (T.lt(word_ids, self._num_input_words) * word_ids
                          + T.ge(word_ids, self._num_input_words) * self._vocab.unk)

        # Run the main rnn with combined inputs
        final_embeddings = self._base_lookup.apply(input_word_ids)
        application_call.add_auxiliary_variable(
            masked_root_mean_square(final_embeddings, words_mask), name='rnn_input_rootmean2')
        if self._retrieval:
            if self._compose_type == 'sum':
                final_embeddings += def_mean
            elif self._compose_type.startswith('fully_connected'):
                concat = T.concatenate([final_embeddings, def_mean], axis=2)
                final_embeddings = self._def_state_compose.apply(concat)
            elif self._compose_type.startswith('linear_and_sum'):
                final_embeddings += self._def_state_transform.apply(def_mean)
            else:
                assert False
            application_call.add_auxiliary_variable(
                masked_root_mean_square(final_embeddings, words_mask),
                name='merged_input_rootmean2')

        if self._disregard_word_embeddings:
            final_embeddings = def_mean

        return final_embeddings