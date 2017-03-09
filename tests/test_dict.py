from dictlearn.retrieval import Dictionary

from tests.test_util import temporary_content_path, TEST_DICT_JSON

def test_dict():
    with temporary_content_path(TEST_DICT_JSON) as path:
        dict_ = Dictionary(path)
    assert dict_.get_definitions('A') == [['B', 'C'], ['D', 'E']]
    assert dict_.get_definitions('D C') == [['A', 'B']]
