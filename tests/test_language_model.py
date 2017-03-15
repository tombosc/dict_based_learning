import numpy

import theano
from theano import tensor
from blocks.initialization import Uniform
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph

from dictlearn.util import str2vec
from dictlearn.vocab import Vocabulary
from dictlearn.retrieval import Dictionary
from dictlearn.language_model import LanguageModel

from tests.util import (
    TEST_VOCAB, TEST_DICT_JSON, temporary_content_path)

def test_language_model():
    with temporary_content_path(TEST_VOCAB) as path:
        vocab = Vocabulary(path)
    with temporary_content_path(TEST_DICT_JSON) as path:
        dict_ = Dictionary(path)

    data = [['a', 'a'], ['b', 'a'], ['a', 'b']]
    data = [[str2vec(s, 3) for s in row] for row in data]
    data = numpy.array(data)
    mask = numpy.ones((data.shape[0], data.shape[1]),
                      dtype=theano.floatX)

    # With the dictionary
    lm = LanguageModel(
        vocab=vocab, dict_=dict_, dim=10,
        weights_init=Uniform(width=0.1),
        biases_init=Uniform(width=0.1))
    lm.initialize()
    costs = lm.apply(tensor.as_tensor_variable(data), mask)
    cg = ComputationGraph(costs)
    def_spans, = VariableFilter(name='def_spans')(cg)
    f = theano.function([], [costs, def_spans])
    costs_value, def_spans_value = f()
    assert (def_spans_value.tolist() ==
            [[0, 2], [2, 4], [4, 5], [5, 7], [7, 9], [9, 10]])

    # Without the dictionary
    lm2 = LanguageModel(
        vocab=vocab, dim=10,
        weights_init=Uniform(width=0.1),
        biases_init=Uniform(width=0.1))
    costs2 = lm2.apply(tensor.as_tensor_variable(data), mask)
    costs2.eval()
