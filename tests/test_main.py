import pytest
from unittest.mock import MagicMock, patch
import sys
from nerdprompt import initialize_app, process_and_print_response, get_next_input

@patch('nerdprompt.parse_arguments')
@patch('nerdprompt.ConfigEater')
@patch('nerdprompt.load_api_key')
@patch('nerdprompt.PerplexityWrapper')
def test_initialize_app(mock_wrapper, mock_load_key, mock_config_eater, mock_parse_args):
    mock_parse_args.return_value = MagicMock(model=None, prompt='default')
    mock_config_eater.return_value.parse_config.return_value = {'llm_model': 'old'}
    mock_load_key.return_value = 'key'
    
    config, args, client = initialize_app()
    
    assert config['llm_model'] == 'old'
    assert client == mock_wrapper.return_value

@patch('nerdprompt.formatted_output')
def test_process_and_print_response(mock_formatted, capsys):
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content='raw content'))]
    client.ask.return_value = (mock_response, 1.0)
    mock_formatted.return_value = 0.5
    
    config = {}
    args = MagicMock(raw=False, nothread=False)
    
    process_and_print_response(client, config, args, "user prompt")
    
    client.message_appender.assert_any_call("user", "user prompt")
    client.message_appender.assert_any_call("assistant", "raw content")
    mock_formatted.assert_called_once()

@patch('builtins.input')
@patch('nerdprompt.get_question')
def test_get_next_input_continue(mock_get_q, mock_input):
    mock_input.return_value = 'y'
    mock_get_q.return_value = 'next question'
    args = MagicMock(paste=False)
    client = MagicMock()
    
    question, cleared = get_next_input(args, client)
    
    assert question == 'next question'
    assert cleared is False

@patch('builtins.input')
@patch('nerdprompt.get_question')
def test_get_next_input_clear(mock_get_q, mock_input):
    mock_input.return_value = 'c'
    mock_get_q.return_value = 'new question'
    args = MagicMock(paste=False)
    client = MagicMock()
    
    question, cleared = get_next_input(args, client)
    
    assert question == 'new question'
    assert cleared is True
    client.clear_history.assert_called_once()
