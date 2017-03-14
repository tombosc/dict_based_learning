from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
from contextlib import contextmanager

TEST_DICT_JSON = (
"""
{
 "a" : [["b", "c"], ["d", "e"]],
 "b" : [["e", "d"]],
 "d c" : [["a", "b"]],
 "e" : [["b", "c", "d"]]
}
""")

TEST_VOCAB = (
"""<unk>
<bos>
<eos>
a
b
c
d
e"""
)

@contextmanager
def temporary_content_path(content):
    _, path = tempfile.mkstemp()
    with open(path, 'w') as dst:
        print(content, file=dst)
    yield path
    os.remove(path)
