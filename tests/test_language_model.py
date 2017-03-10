from theano import tensor

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

    data = [['a', 'a'], ['b', 'a']]
    data = [[str2vec(s, 3) for s in row] for row in data]

    lm = LanguageModel(vocab, dict_)
    lm.apply(tensor.as_tensor_variable(data))
