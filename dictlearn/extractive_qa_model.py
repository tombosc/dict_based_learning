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
    coattention : bool
        Use the coattention mechanism.
    num_input_words : int
        The number of input words. If 0, `vocab.size()` is used.
    vocab
        The vocabulary object.
    retrieval
        The dictionary retrieval algorithm. If `None`, the language model
        does not use any dictionary.

    """
    def __init__(self, dim, emb_dim, coattention, num_input_words, vocab, retrieval=None, **kwargs):
        self._vocab = vocab
        if emb_dim == 0:
            emb_dim = dim
        if num_input_words == 0:
            num_input_words = vocab.size()
        self._coattention = coattention
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
        self._bidir_fork = Linear(3 * dim if coattention else 2 * dim, 4 * dim, name='bidir_fork')
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

        # create default input variables
        self.contexts = tensor.lmatrix('contexts')
        self.context_mask = tensor.matrix('contexts_mask')
        self.questions = tensor.lmatrix('questions')
        self.question_mask = tensor.matrix('questions_mask')
        self.answer_begins = tensor.lvector('answer_begins')
        self.answer_ends = tensor.lvector('answer_ends')
        self.input_vars = [
            self.contexts, self.context_mask,
            self.questions, self.question_mask,
            self.answer_begins, self.answer_ends]

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

        contexts = (
            tensor.lt(contexts, self._num_input_words) * contexts
            + tensor.ge(contexts, self._num_input_words) * self._vocab.unk)
        application_call.add_auxiliary_variable(
            contexts, name='context_word_ids')
        questions = (
            tensor.lt(questions, self._num_input_words) * questions
            + tensor.ge(questions, self._num_input_words) * self._vocab.unk)
        context_embs = self._lookup.apply(contexts)
        question_embs = self._lookup.apply(questions)

        context_enc = flip01(
            self._encoder_rnn.apply(
                self._encoder_fork.apply(
                    flip01(context_embs)),
                mask=contexts_mask.T)[0])
        question_enc_pre = flip01(
            self._encoder_rnn.apply(
                self._encoder_fork.apply(
                    flip01(question_embs)),
                mask=questions_mask.T)[0])
        question_enc = tensor.tanh(self._question_transform.apply(question_enc_pre))

        # should be (batch size, context length, question_length)
        affinity = tensor.batched_dot(context_enc, flip12(question_enc))
        affinity_mask = contexts_mask[:, :, None] * questions_mask[:, None, :]
        affinity = affinity * affinity_mask - 1000 * (1 - affinity_mask)
        # soft-aligns every position in the context to positions in the question
        d2q_att_weights = self._softmax.apply(affinity, extra_ndim=1)
        application_call.add_auxiliary_variable(
            d2q_att_weights.copy(), name='d2q_att_weights')
        # soft-aligns every position in the question to positions in the document
        q2d_att_weights = self._softmax.apply(flip12(affinity), extra_ndim=1)
        application_call.add_auxiliary_variable(
            q2d_att_weights.copy(), name='q2d_att_weights')

        # question encoding "in the view of the document"
        question_enc_informed = tensor.batched_dot(
            q2d_att_weights, context_enc)
        question_enc_concatenated = tensor.concatenate(
            [question_enc, question_enc_informed], 2)
        # document encoding "in the view of the question"
        context_enc_informed = tensor.batched_dot(
            d2q_att_weights, question_enc_concatenated)

        if self._coattention:
            context_enc_concatenated = tensor.concatenate(
                [context_enc, context_enc_informed], 2)
        else:
            question_repr_repeated = tensor.repeat(
                question_enc[:, [-1], :], context_enc.shape[1], axis=1)
            context_enc_concatenated = tensor.concatenate(
                [context_enc, question_repr_repeated], 2)

        # note: forward and backward LSTMs share the
        # input weights in the current impl
        bidir_states = flip01(
            self._bidir.apply(
                self._bidir_fork.apply(
                    flip01(context_enc_concatenated)),
                mask=contexts_mask.T)[0])

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
            predicted_begins, name='predicted_begins')
        application_call.add_auxiliary_variable(
            predicted_ends, name='predicted_ends')
        application_call.add_auxiliary_variable(
            exact_match, name='exact_match')

        return begin_costs + end_costs

    def apply_with_default_vars(self):
        return self.apply(*self.input_vars)
