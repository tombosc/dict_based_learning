import numpy

import theano
from theano import tensor
from blocks.initialization import Uniform
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
import json

from dictlearn.util import str2vec
from dictlearn.vocab import Vocabulary
from dictlearn.retrieval import Dictionary
from dictlearn.language_model import LanguageModel
from dictlearn.generate_synthetic_data import FakeTextGenerator

from tests.util import (
    TEST_VOCAB, TEST_DICT_JSON, temporary_content_path)

def t_e_s_t_language_model():
    V = 50
    gen = FakeTextGenerator(V, 6, 6, 1.0, 0.2)
    n_sentences = 3
    len_sentences = 7
    data = [gen.sample_sentence(len_sentences) for i in range(n_sentences)]
    vocab_list = '\n'.join(list(set(gen.vocabulary)))
    dict_json = json.dumps(gen.dictionary)
    print "JSON dict:", dict_json

    with temporary_content_path(vocab_list) as path:
        vocab = Vocabulary(path)
    with temporary_content_path(dict_json) as path:
        dict_ = Dictionary(path)

    data = [[str2vec(s, generator.tok_len) for s in row] for row in data]
    data = numpy.array(data)
    print "Data:", data

    # With the dictionary
    lm = LanguageModel(
        vocab=vocab, dict_=dict_, dim=10,
        weights_init=Uniform(width=0.1),
        biases_init=Uniform(width=0.1))
    lm.initialize()
    costs = lm.apply(tensor.as_tensor_variable(data),
                     numpy.ones((data.shape[0], data.shape[1])))
    cg = ComputationGraph(costs)
    def_spans, = VariableFilter(name='def_spans')(cg)
    f = theano.function([], [costs, def_spans])
    costs_value, def_spans_value = f()
    assert def_spans_value.tolist() == [[0, 2], [2, 4], [4, 5], [5, 7]]

    # Without the dictionary
    lm2 = LanguageModel(
        vocab=vocab, dim=10,
        weights_init=Uniform(width=0.1),
        biases_init=Uniform(width=0.1))
    costs2 = lm2.apply(tensor.as_tensor_variable(data),
                     numpy.ones((data.shape[0], data.shape[1])))
    costs2.eval()
