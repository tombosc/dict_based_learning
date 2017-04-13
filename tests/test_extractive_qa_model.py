import numpy

import theano
from theano import tensor
from blocks.initialization import Uniform
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph

from dictlearn.util import str2vec
from dictlearn.vocab import Vocabulary
from dictlearn.retrieval import Retrieval, Dictionary
from dictlearn.extractive_qa_model import ExtractiveQAModel

from tests.util import (
    TEST_VOCAB, TEST_DICT_JSON, temporary_content_path)

def test_language_model():
    with temporary_content_path(TEST_VOCAB) as path:
        vocab = Vocabulary(path)
    with temporary_content_path(TEST_DICT_JSON) as path:
        dict_ = Dictionary(path)

    def make_data_and_mask(data):
        data = [[str2vec(s, 3) for s in row] for row in data]
        data = numpy.array(data)
        mask = numpy.ones((data.shape[0], data.shape[1]),
                        dtype=theano.config.floatX)
        return data, mask
    # create some dummy data
    contexts, context_mask = make_data_and_mask(
        [['a', 'a', 'a', 'b'], ['b', 'a', 'b', 'a'], ['a', 'b', 'b', 'b']])
    questions, question_mask = make_data_and_mask(
        [['a', 'a'], ['b', 'a'], ['a', 'b']])
    answer_begins = [0, 0, 1]
    answer_ends = [1, 2, 2]

    qam = ExtractiveQAModel(
        vocab=vocab, dim=10, num_input_words=10,
        weights_init=Uniform(width=0.1),
        biases_init=Uniform(width=0.1))
    qam.initialize()

    costs = qam.apply(tensor.as_tensor_variable(contexts), context_mask,
                      tensor.as_tensor_variable(questions), question_mask,
                      tensor.as_tensor_variable(answer_begins),
                      tensor.as_tensor_variable(answer_ends))
    assert costs.eval().shape == (3,)
