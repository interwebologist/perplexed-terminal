import pytest
from unittest.mock import MagicMock, patch
from nerdprompt import PerplexityWrapper, ANSI_CODES

@pytest.fixture
def config():
    return {
        'system_content': {'default': 'system prompt'},
        'header_1': ['bold', 'magenta'],
        'header_2': ['bold', 'green'],
        'header_3': ['bold', 'cyan'],
        'header_4': ['bold', 'yellow'],
        'header_5': ['bold', 'blue'],
        'header_6': ['bold', 'red'],
        'ascii_divider_choice': 1,
        'ascii_dividers': {1: '---'},
        'dividers_color': 'magenta',
        'bullet_point_unicode': '*',
        'llm_url': 'http://test',
        'llm_model': 'test-model'
    }

def test_wrapper_init(config):
    wrapper = PerplexityWrapper(config, 'api-key')
    assert len(wrapper.messages) == 1
    assert wrapper.messages[0]['role'] == 'system'

def test_message_appender(config):
    wrapper = PerplexityWrapper(config, 'api-key')
    wrapper.message_appender('user', 'hello')
    assert len(wrapper.messages) == 2
    assert wrapper.messages[1]['content'] == 'hello'

def test_clear_history(config):
    wrapper = PerplexityWrapper(config, 'api-key')
    wrapper.message_appender('user', 'hello')
    wrapper.clear_history()
    assert len(wrapper.messages) == 1
    assert wrapper.messages[0]['role'] == 'system'

def test_remove_citations(config):
    wrapper = PerplexityWrapper(config, 'api-key')
    text = "Hello [1] world [23]."
    assert wrapper.remove_citations(text) == "Hello  world ."

def test_code_extractor(config):
    wrapper = PerplexityWrapper(config, 'api-key')
    text = "Here is code:\n```python\nprint(1)\n```\nAnd more."
    result = wrapper.code_extractor(text)
    assert '<CODE__REMOVED__0>' in result['text']
    assert len(result['code_blocks']) == 1
    assert "print(1)" in result['code_blocks'][0]

def test_code_injector(config):
    wrapper = PerplexityWrapper(config, 'api-key')
    data = {
        'ansi_converted_text': "Converted <CODE__REMOVED__0>",
        'code_blocks': ["```python\nprint(1)\n```"]
    }
    result = wrapper.code_injector(data)
    assert "Converted ```python\nprint(1)\n```" in result

def test_markdown_to_ansi_headers(config):
    wrapper = PerplexityWrapper(config, 'api-key')
    text = "# Header 1\n## Header 2"
    result = wrapper.markdown_to_ansi(ANSI_CODES, config, text)
    assert ANSI_CODES['magenta'] in result
    assert ANSI_CODES['green'] in result

def test_markdown_to_ansi_bold_italic(config):
    wrapper = PerplexityWrapper(config, 'api-key')
    text = "**bold** *italic*"
    result = wrapper.markdown_to_ansi(ANSI_CODES, config, text)
    assert ANSI_CODES['bold'] in result
    assert ANSI_CODES['italic'] in result

def test_markdown_to_ansi_dividers(config):
    wrapper = PerplexityWrapper(config, 'api-key')
    text = "---\ncontent\n---"
    result = wrapper.markdown_to_ansi(ANSI_CODES, config, text)
    assert ANSI_CODES['magenta'] in result  # divider color

def test_markdown_to_ansi_bullets(config):
    wrapper = PerplexityWrapper(config, 'api-key')
    text = "- item 1\n- item 2"
    result = wrapper.markdown_to_ansi(ANSI_CODES, config, text)
    assert '*' in result  # bullet_point_unicode

@patch('nerdprompt.OpenAI')
def test_ask(mock_openai, config):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content='response content'))]
    mock_client.chat.completions.create.return_value = mock_response

    wrapper = PerplexityWrapper(config, 'api-key')
    resp, api_time = wrapper.ask(config)
    
    assert resp.choices[0].message.content == 'response content'
    assert isinstance(api_time, float)
