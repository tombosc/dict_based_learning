from dictlearn.util import vec2str, str2vec

def test_vec2str():
    vector = map(ord, 'abc') + [0, 0]
    assert vec2str(vector) == 'abc'

def test_str2vec():
    assert str2vec('def', 5).tolist() == [ord('d'), ord('e'), ord('f'), 0, 0]
    assert str2vec('abcdef', 3).tolist() == [ord('a'), ord('b'), ord('c')]

