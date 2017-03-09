from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dictlearn.retrieval import (
    vec2str, Dictionary, Vocabulary, Retrieval)

from tests.test_util import (
    TEST_VOCAB, TEST_DICT_JSON, temporary_content_path)


def test_vec2str():
    vector = map(ord, 'abc') + [0, 0]
    assert vec2str(vector) == 'abc'


def test_retrieval():
    with temporary_content_path(TEST_VOCAB) as path:
        vocab = Vocabulary(path)
    with temporary_content_path(TEST_DICT_JSON) as path:
        dict_ = Dictionary(path)

    # First, a super simple case
    batch = [['A']]
    retrieval = Retrieval(dict_, vocab)
    defs, def_map = retrieval.retrieve(batch)
    assert defs == [[4, 5], [6, 7]]
    assert def_map == [(0, 0, 0), (0, 0, 1)]

