from theano import tensor

from dictlearn.vocab import Vocabulary
from dictlearn.ops import WordToIdOp

from tests.util import (
    TEST_VOCAB, temporary_content_path)

def test_vocab_op():
    with temporary_content_path(TEST_VOCAB) as path:
        vocab = Vocabulary(path)
    op = WordToIdOp(vocab)

    input_ = tensor.as_tensor_variable([ord('d'), ord(' '), ord('c'), 0, 0])
    assert op(input_).eval() == 0

    input_ = tensor.as_tensor_variable([ord('a')])
    assert op(input_).eval() == 5

    input_ = tensor.as_tensor_variable([[ord('a'), 0], [ord('b'), 0]])
    assert list(op(input_).eval()) == [5, 6]

    # test top_k
    with temporary_content_path(TEST_VOCAB) as path:
        vocab = Vocabulary(path, 6)
    assert vocab.word_to_id('a') == 5
    assert vocab.word_to_id('b') == 0
