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
"""<unk> 0
<bos> 0
<eos> 0
a 9
b 7
c 3
d 2
e 1"""
)

TEST_TEXT = (
"""abc abc def
def def xyz
xyz
abc def xyz
"""
)

@contextmanager
def temporary_content_path(content):
    _, path = tempfile.mkstemp()
    with open(path, 'w') as dst:
        print(content, file=dst)
    yield path
    os.remove(path)
