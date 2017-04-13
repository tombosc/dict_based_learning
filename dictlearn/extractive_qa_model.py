"""A dictionary-equipped extractive QA model."""
import theano
from theano import tensor

from blocks.bricks import Initializable, Linear, NDimensionalSoftmax, MLP, Tanh, Rectifier
from blocks.bricks.base import application
from blocks.bricks.recurrent import LSTM
from blocks.bricks.recurrent.misc import Bidirectional
from blocks.bricks.lookup import LookupTable

from dictlearn.ops import WordToIdOp, RetrievalOp


class ExtractiveQAModel(Initializable):
    """The dictionary-equipped extractive QA model.

    Parameters
    ----------
    dim : int
        The default dimensionality for the components.
    emd_dim : int
        The dimensionality for the embeddings. If 0, `dim` is used.
    num_input_words : int
        The number of input words. If 0, `vocab.size()` is used.
    vocab
        The vocabulary object.
    retrieval
        The dictionary retrieval algorithm. If `None`, the language model
        does not use any dictionary.

    """
    def __init__(self, dim, emb_dim, num_input_words, vocab, retrieval=None, **kwargs):
        self._vocab = vocab
        if emb_dim == 0:
            emb_dim = dim
        if num_input_words == 0:
            num_input_words = vocab.size()
        self._num_input_words = num_input_words
        self._retrieval = retrieval

        self._word_to_id = WordToIdOp(self._vocab)

        if self._retrieval:
            self._retrieve = RetrievalOp(retrieval)

        # Dima: we can have slightly less copy-paste here if we
        # copy the RecurrentFromFork class from my other projects.
        children = []
        self._lookup = LookupTable(self._num_input_words, emb_dim)
        self._encoder_fork = Linear(emb_dim, 4 * dim, name='encoder_fork')
        self._encoder_rnn = LSTM(dim, name='encoder_rnn')
        self._question_transform = Linear(dim, dim, name='question_transform')
        self._bidir_fork = Linear(2 * dim, 4 * dim, name='bidir_fork')
        self._bidir = Bidirectional(LSTM(dim), name='bidir')
        children.extend([self._lookup,
                         self._encoder_fork, self._encoder_rnn,
                         self._question_transform,
                         self._bidir, self._bidir_fork])

        self._begin_readout = MLP([None], [2 * dim, 1], name='begin_readout')
        self._end_readout = MLP([None], [2 * dim, 1], name='end_readout')
        self._softmax = NDimensionalSoftmax()
        children.extend([self._begin_readout, self._end_readout, self._softmax])

        super(ExtractiveQAModel, self).__init__(children=children, **kwargs)

    def set_embeddings(self, embeddings):
        self._lookup.parameters[0].set_value(embeddings.astype(theano.config.floatX))

    def embeddings_var(self):
        return self._lookup.parameters[0]

    @application
    def apply(self, application_call,
              contexts, contexts_mask, questions, questions_mask,
              answer_begins, answer_ends):
        def flip01(x):
            return x.transpose((1, 0, 2))
        def flip12(x):
            return x.transpose((0, 2, 1))

        context_word_ids = self._word_to_id(contexts)
        context_word_ids = (
            tensor.lt(context_word_ids, self._num_input_words) * context_word_ids
            + tensor.ge(context_word_ids, self._num_input_words) * self._vocab.unk)
        question_word_ids = self._word_to_id(questions)
        question_word_ids = (
            tensor.lt(question_word_ids, self._num_input_words) * question_word_ids
            + tensor.ge(question_word_ids, self._num_input_words) * self._vocab.unk)
        context_embs = self._lookup.apply(context_word_ids)
        question_embs = self._lookup.apply(question_word_ids)

        context_enc = flip01(
            self._encoder_rnn.apply(self._encoder_fork.apply(
                flip01(context_embs)))[0])
        question_enc_pre = flip01(
            self._encoder_rnn.apply(self._encoder_fork.apply(
                flip01(question_embs)))[0])
        question_enc = tensor.tanh(self._question_transform.apply(question_enc_pre))

        # should be (batch size, context length, question_length)
        affinity = tensor.batched_dot(context_enc, flip12(question_enc))
        d2q_att_weights = self._softmax.apply(affinity, extra_ndim=1)
        d2q_att_weights *= questions_mask[:, None, :]
        q2d_att_weights = self._softmax.apply(flip12(affinity), extra_ndim=1)
        q2d_att_weights *= contexts_mask[:, None, :]

        # question encoding "in the view of the document"
        question_enc_informed = tensor.batched_dot(
            q2d_att_weights, context_enc)
        question_enc_concatenated = tensor.concatenate([question_enc, question_enc_informed], 2)
        document_enc_informed = tensor.batched_dot(
            d2q_att_weights, question_enc_concatenated)

        # note: forward and backward LSTMs share the
        # input weights in the current impl
        bidir_states = flip01(
            self._bidir.apply(self._bidir_fork.apply(
                flip01(document_enc_informed)))[0])

        begin_readouts = self._begin_readout.apply(bidir_states)[:, :, 0]
        begin_readouts = begin_readouts * contexts_mask - 1000.0 * (1 - contexts_mask)
        begin_costs = self._softmax.categorical_cross_entropy(
            answer_begins, begin_readouts)

        end_readouts = self._end_readout.apply(bidir_states)[:, :, 0]
        end_readouts = end_readouts * contexts_mask - 1000.0 * (1 - contexts_mask)
        end_costs = self._softmax.categorical_cross_entropy(
            answer_ends, end_readouts)

        predicted_begins = begin_readouts.argmax(axis=-1)
        predicted_ends = end_readouts.argmax(axis=-1)
        exact_match = (tensor.eq(predicted_begins, answer_begins) *
                       tensor.eq(predicted_ends, answer_ends))
        application_call.add_auxiliary_variable(
            exact_match, name='exact_match')

        return begin_costs + end_costs
