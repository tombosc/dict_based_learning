"""A dictionary-equipped extractive QA model."""
import theano
from theano import tensor
from theano.gradient import disconnected_grad

from collections import OrderedDict

from blocks.bricks import Initializable, Linear, NDimensionalSoftmax, MLP, Tanh, Rectifier
from blocks.bricks.base import application, Brick
from blocks.bricks.simple import Rectifier
from blocks.bricks.recurrent import LSTM
from blocks.bricks.recurrent.misc import Bidirectional
from blocks.bricks.lookup import LookupTable
from blocks.select import Selector

from dictlearn.ops import WordToIdOp, RetrievalOp
from dictlearn.lookup import (
    LSTMReadDefinitions, MeanPoolReadDefinitions,
    MeanPoolCombiner)
from dictlearn.theano_util import unk_ratio


def flip01(x):
    return x.transpose((1, 0, 2))


def flip12(x):
    return x.transpose((0, 2, 1))


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
        The vocabulary object.
    use_definitions : bool
        Triggers the use of definitions.
    reuse_word_embeddings : bool
    compose_type : str

    """
    def __init__(self, dim, emb_dim, readout_dims,
                 num_input_words, def_num_input_words, vocab,
                 use_definitions, def_word_gating, compose_type, coattention,
                 def_reader, reuse_word_embeddings, random_unk, **kwargs):
        self._vocab = vocab
        if emb_dim == 0:
            emb_dim = dim
        if num_input_words == 0:
            num_input_words = vocab.size()
        if def_num_input_words == 0:
            def_num_input_words = num_input_words

        self._coattention = coattention
        self._num_input_words = num_input_words
        self._use_definitions = use_definitions
        self._random_unk = random_unk
        self._reuse_word_embeddings = reuse_word_embeddings

        lookup_num_words = num_input_words
        if reuse_word_embeddings:
            lookup_num_words = max(num_input_words, def_num_input_words)
        if random_unk:
            lookup_num_words = vocab.size()

        # Dima: we can have slightly less copy-paste here if we
        # copy the RecurrentFromFork class from my other projects.
        children = []
        self._lookup = LookupTable(lookup_num_words, emb_dim)
        self._encoder_fork = Linear(emb_dim, 4 * dim, name='encoder_fork')
        self._encoder_rnn = LSTM(dim, name='encoder_rnn')
        self._question_transform = Linear(dim, dim, name='question_transform')
        self._bidir_fork = Linear(3 * dim if coattention else 2 * dim, 4 * dim, name='bidir_fork')
        self._bidir = Bidirectional(LSTM(dim), name='bidir')
        children.extend([self._lookup,
                         self._encoder_fork, self._encoder_rnn,
                         self._question_transform,
                         self._bidir, self._bidir_fork])

        activations = [Rectifier()] * len(readout_dims) + [None]
        readout_dims = [2 * dim] + readout_dims + [1]
        self._begin_readout = MLP(activations, readout_dims, name='begin_readout')
        self._end_readout = MLP(activations, readout_dims, name='end_readout')
        self._softmax = NDimensionalSoftmax()
        children.extend([self._begin_readout, self._end_readout, self._softmax])

        if self._use_definitions:
            # A potential bug here: we pass the same vocab to the def reader.
            # If a different token is reserved for UNK in text and in the definitions,
            # we can be screwed.
            def_reader_class = eval(def_reader)
            def_reader_kwargs = dict(
                num_input_words=def_num_input_words,
                dim=dim, emb_dim=emb_dim,
                vocab=vocab,
                lookup=self._lookup if reuse_word_embeddings else None)
            if def_reader_class == MeanPoolReadDefinitions:
                def_reader_kwargs.update(dict(normalize=True, translate=False))
            self._def_reader = def_reader_class(**def_reader_kwargs)
            self._combiner = MeanPoolCombiner(
                dim=dim, emb_dim=emb_dim,
                def_word_gating=def_word_gating, compose_type=compose_type)
            children.extend([self._def_reader, self._combiner])

        super(ExtractiveQAModel, self).__init__(children=children, **kwargs)

        # create default input variables
        self.contexts = tensor.lmatrix('contexts')
        self.context_mask = tensor.matrix('contexts_mask')
        self.questions = tensor.lmatrix('questions')
        self.question_mask = tensor.matrix('questions_mask')
        self.answer_begins = tensor.lvector('answer_begins')
        self.answer_ends = tensor.lvector('answer_ends')
        input_vars = [
            self.contexts, self.context_mask,
            self.questions, self.question_mask,
            self.answer_begins, self.answer_ends]
        if self._use_definitions:
            self.defs = tensor.lmatrix('defs')
            self.def_mask = tensor.matrix('def_mask')
            self.contexts_def_map = tensor.lmatrix('contexts_def_map')
            self.questions_def_map = tensor.lmatrix('questions_def_map')
            input_vars.extend([self.defs, self.def_mask,
                               self.contexts_def_map, self.questions_def_map])
        self.input_vars = OrderedDict([(var.name, var) for var in input_vars])

    def set_embeddings(self, embeddings):
        self._lookup.parameters[0].set_value(embeddings.astype(theano.config.floatX))

    def embeddings_var(self):
        return self._lookup.parameters[0]

    def def_reading_parameters(self):
        parameters = Selector(self._def_reader).get_parameters().values()
        parameters.extend(Selector(self._combiner).get_parameters().values())
        if self._reuse_word_embeddings:
            lookup_parameters = Selector(self._lookup).get_parameters().values()
            parameters = [p for p in parameters if p not in lookup_parameters]
        return parameters


    @application
    def _encode(self, application_call, text, mask, def_embs=None, def_map=None, text_name=None):
        if not self._random_unk:
            text = (
                tensor.lt(text, self._num_input_words) * text
                + tensor.ge(text, self._num_input_words) * self._vocab.unk)
        if text_name:
            application_call.add_auxiliary_variable(
                unk_ratio(text, mask, self._vocab.unk),
                name='{}_unk_ratio'.format(text_name))
        embs = self._lookup.apply(text)
        if self._random_unk:
            embs = (
                tensor.lt(text, self._num_input_words)[:, :, None] * embs
                + tensor.ge(text, self._num_input_words)[:, :, None] * disconnected_grad(embs))
        if def_embs:
            embs = self._combiner.apply(embs, mask, def_embs, def_map)
        encoded = flip01(
            self._encoder_rnn.apply(
                self._encoder_fork.apply(
                    flip01(embs)),
                mask=mask.T)[0])
        return encoded


    @application
    def apply(self, application_call,
              contexts, contexts_mask, questions, questions_mask,
              answer_begins, answer_ends,
              defs=None, def_mask=None, contexts_def_map=None, questions_def_map=None):
        def_embs = None
        if self._use_definitions:
            def_embs = self._def_reader.apply(defs, def_mask)

        context_enc = self._encode(contexts, contexts_mask,
                                   def_embs, contexts_def_map, 'context')
        question_enc_pre = self._encode(questions, questions_mask,
                                         def_embs, questions_def_map, 'question')
        question_enc = tensor.tanh(self._question_transform.apply(question_enc_pre))

        # should be (batch size, context length, question_length)
        affinity = tensor.batched_dot(context_enc, flip12(question_enc))
        affinity_mask = contexts_mask[:, :, None] * questions_mask[:, None, :]
        affinity = affinity * affinity_mask - 1000.0 * (1 - affinity_mask)
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
        return self.apply(*self.input_vars.values())
