from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

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

    # check a super simple case
    batch = [['a']]
    defs, def_map = Retrieval(dict_, vocab).retrieve(batch)
    assert defs == [[4, 5], [6, 7]]
    assert def_map == [(0, 0, 0), (0, 0, 1)]

    # check that vectors are handled correctly
    batch = numpy.array([ord('d'), ord(' '), ord('c'), 0, 0])[None, None, :]
    defs, def_map = Retrieval(dict_, vocab).retrieve(batch)
    assert defs == [[3, 4]]
    assert def_map == [(0, 0, 0)]

    # check a complex case
    batch = [['a', 'b', 'b'], ['d c', 'a', 'b']]
    defs, def_map = Retrieval(dict_, vocab).retrieve(batch)
    assert defs == [[4, 5], [6, 7], [7, 6], [3, 4]]
    assert def_map == [(0, 0, 0), (0, 0, 1),
                        (0, 1, 2),
                        (0, 2, 2),
                        (1, 0, 3),
                        (1, 1, 0), (1, 1, 1),
                        (1, 2, 2)]





