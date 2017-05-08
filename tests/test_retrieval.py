from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
from numpy.testing import assert_allclose

from theano import tensor

from dictlearn.vocab import Vocabulary
from dictlearn.retrieval import (
    vec2str, Dictionary, Retrieval)
from dictlearn.ops import RetrievalOp

from tests.util import (
    TEST_VOCAB, TEST_DICT_JSON, temporary_content_path)


def test_vec2str():
    vector = map(ord, 'abc') + [0, 0]
    assert vec2str(vector) == 'abc'


def test_retrieval():
    with temporary_content_path(TEST_VOCAB, ".txt") as path:
        vocab = Vocabulary(path)
    with temporary_content_path(TEST_DICT_JSON, ".json") as path:
        dict_ = Dictionary(path)

    # check a super simple case
    batch = [['a']]
    defs, def_map = Retrieval(vocab, dict_).retrieve(batch)
    assert defs == [[3, 6, 7, 4], [3, 8, 9, 4]]
    assert def_map == [(0, 0, 0), (0, 0, 1)]

    # check that vectors are handled correctly
    batch = numpy.array([ord('d'), ord(' '), ord('c'), 0, 0])[None, None, :]
    defs, def_map = Retrieval(vocab, dict_).retrieve(batch)
    assert defs == [[3, 5, 6, 4]]
    assert def_map == [(0, 0, 0)]

    # check a complex case
    batch = [['a', 'b', 'b'], ['d c', 'a', 'b']]
    defs, def_map = Retrieval(vocab, dict_).retrieve(batch)
    assert defs == [[3, 6, 7, 4],
                    [3, 8, 9, 4],
                    [3, 9, 8, 4],
                    [3, 5, 6, 4]]
    assert def_map == [(0, 0, 0), (0, 0, 1),
                       (0, 1, 2),
                       (0, 2, 2),
                       (1, 0, 3),
                       (1, 1, 0), (1, 1, 1),
                       (1, 2, 2)]

    # check a complex case with exclude top k
    batch = [['a', 'b', 'c', 'd'], ['a', 'e', 'b']]
    exclude_top_k = 7 # should exclude 'a', 'b', 'c', 'd' and only define 'e'
    defs, def_map = Retrieval(vocab, dict_, exclude_top_k=exclude_top_k).retrieve(batch)
    assert defs == [[3, 6, 7, 8, 4]]
    assert def_map == [(1, 1, 0)]

    # check the op
    retrieval_op = RetrievalOp(Retrieval(vocab, dict_))
    batch = tensor.as_tensor_variable(
        [[[ord('d'), ord(' '), ord('c'), 0, 0],
          [ord('e'), 0, 0, 0, 0]]])
    defs_var, mask_var,  def_map_var = retrieval_op(batch)
    assert defs_var.eval().tolist() == [[3, 5, 6, 4, 0],
                                        [3, 6, 7, 8, 4]]
    assert_allclose(mask_var.eval(), [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    assert def_map_var.eval().tolist() == [[0, 0, 0], [0, 1, 1]]
