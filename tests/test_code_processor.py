import pytest
from nerdprompt import CodeProcesser, ANSI_CODES

@pytest.fixture
def config():
    return {
        'code_syntax_theme': 'dracula',
        'code_divider_choice': 1,
        'code_dividers': {1: '==='},
        'code_dividers_color': 'green'
    }

def test_extract_code_type_and_syntax():
    processor = CodeProcesser()
    input_str = "```python\nprint('hello')\n```"
    result = processor.extract_code_type_and_syntax(input_str)
    assert result['code_type'] == 'python'
    assert "print('hello')" in result['code_syntax']

def test_extract_code_type_and_syntax_no_match():
    processor = CodeProcesser()
    input_str = "no code here"
    result = processor.extract_code_type_and_syntax(input_str)
    assert result is None

def test_syntax_highlighter(config):
    processor = CodeProcesser()
    code_data = {
        'code_type': 'python',
        'code_syntax': "print('hello')"
    }
    result = processor.syntax_highlighter(config, code_data)
    assert 'highlighted_code' in result
    assert '\x1b[' in result['highlighted_code']  # ANSI codes present

def test_rebuild_code_type_and_syntax(config):
    processor = CodeProcesser()
    extracted_code = {
        'code_type': 'python',
        'highlighted_code': 'HIGHLIGHTED'
    }
    result = processor.rebuild_code_type_and_syntax(ANSI_CODES, config, extracted_code)
    assert 'Python Code:' in result
    assert 'HIGHLIGHTED' in result
    assert '===' in result
