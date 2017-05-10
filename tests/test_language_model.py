import numpy as np

import theano
from theano import tensor
from blocks.initialization import Uniform
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph

from dictlearn.util import str2vec
from dictlearn.vocab import Vocabulary
from dictlearn.retrieval import Retrieval, Dictionary
from dictlearn.language_model import LanguageModel

from tests.util import (
    TEST_VOCAB, TEST_DICT_JSON, temporary_content_path)

def test_language_model():
    with temporary_content_path(TEST_VOCAB) as path:
        vocab = Vocabulary(path)
    with temporary_content_path(TEST_DICT_JSON, suffix=".json") as path:
        dict_ = Dictionary(path)

    floatX = theano.config.floatX

    def make_data_and_mask(data):
        data = [[str2vec(s, 3) for s in row] for row in data]
        data = np.array(data)
        mask = np.ones((data.shape[0], data.shape[1]),
                        dtype=floatX)
        return data, mask
    words_val, mask_val = make_data_and_mask([['p', 'e', 'a', ], ['a', 'e', 'p',]])
    mask_val[1,2] = 0
    print "data:"
    print words_val
    print "mask:"
    print mask_val
    mask_def_emb_val = np.asarray([[0, 1], [0,0]])

    # With the dictionary
    retrieval = Retrieval(vocab, dict_, exclude_top_k=7)
    lm = LanguageModel(7, 5, vocab.size(), vocab.size(), 
        vocab=vocab, retrieval=retrieval,
        compose_type='transform_and_sum',
        weights_init=Uniform(width=0.1),
        biases_init=Uniform(width=0.1))
    lm.initialize()
    words = tensor.ltensor3('words')
    mask = tensor.matrix('mask', dtype=floatX)
    costs = lm.apply(words, mask)
    cg = ComputationGraph(costs)
    def_mean, = VariableFilter(name='_dict_word_embeddings')(cg)
    def_mean_f = theano.function([words], def_mean)

    perplexities = VariableFilter(name_regex='perplexity.*')(cg)
    mask_def_emb, = VariableFilter(name='mask_def_emb')(cg)
    perplexities_f = theano.function([words, mask], perplexities)
    perplexities_v = perplexities_f(words_val, mask_val)
    mask_emb_f = theano.function([words, mask], mask_def_emb)
    mask_def_v = mask_emb_f(words_val, mask_val)
    for v,p in zip(perplexities_v,perplexities):
        print p.name, ":", v
    assert(np.allclose(mask_def_v, mask_def_emb_val))
    
    #costs_value, def_spans_value = f()
    #assert (def_spans_value.tolist() ==
    #       [[0, 2], [2, 4], [4, 5], [5, 7], [7, 9], [9, 10]])

    # Without the dictionary
    #lm2 = LanguageModel(
    #    vocab=vocab, dim=10,
    #    num_input_words=vocab.size(),
    #    num_output_words=vocab.size(),
    #    weights_init=Uniform(width=0.1),
    #    biases_init=Uniform(width=0.1))
    #costs2 = lm2.apply(tensor.as_tensor_variable(data), mask)
    #costs2.eval()

    ## With the dictionary but only some definitions
    #data_no_def, mask_no_def  = make_data_and_mask([['a', 'p'], ['p', 'a']])
    #lm.apply(tensor.as_tensor_variable(data_no_def), mask_no_def).eval()

if __name__ == "__main__":
    test_language_model()
