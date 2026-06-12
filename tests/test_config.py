import pytest
from unittest.mock import patch, mock_open
from nerdprompt import ConfigEater, ANSI_CODES

@pytest.fixture
def valid_config():
    return {
        'llm_url': 'https://api.example.com',
        'llm_model': 'test-model',
        'remove_perplexity_citations': True,
        'dividers_color': 'magenta',
        'code_dividers_color': 'green',
        'code_syntax_theme': 'dracula',
        'system_content': {'default': 'test prompt'},
        'bullet_point_unicode': '\u2022',
        'header_1': ['bold', 'magenta'],
        'header_2': ['bold', 'green'],
        'header_3': ['bold', 'cyan'],
        'header_4': ['bold', 'yellow'],
        'header_5': ['bold', 'blue'],
        'header_6': ['bold', 'red'],
        'ascii_divider_position': 'center',
        'ascii_divider_choice': 1,
        'ascii_dividers': {1: '---'},
        'code_divider_choice': 1,
        'code_dividers': {1: '==='}
    }

def test_check_config_valid(valid_config):
    eater = ConfigEater()
    # Should not raise any exception
    eater.check_config(ANSI_CODES, valid_config)

def test_check_config_invalid(valid_config):
    eater = ConfigEater()
    valid_config['ascii_divider_position'] = 'invalid'
    with pytest.raises(ValueError, match="Config validation error"):
        eater.check_config(ANSI_CODES, valid_config)

def test_parse_config():
    eater = ConfigEater()
    mock_yaml = "llm_url: 'http://test'\nllm_model: 'model'"
    with patch("builtins.open", mock_open(read_data=mock_yaml)):
        with patch("yaml.safe_load", return_value={'llm_url': 'http://test', 'llm_model': 'model'}):
            config = eater.parse_config()
            assert config['llm_url'] == 'http://test'
            assert config['llm_model'] == 'model'
