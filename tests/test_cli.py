import pytest
import sys
from unittest.mock import patch, MagicMock
from nerdprompt import load_api_key, parse_arguments, get_question, formatted_output

def test_load_api_key_success():
    with patch("os.getenv", return_value="test-key"):
        assert load_api_key() == "test-key"

def test_load_api_key_missing():
    with patch("os.getenv", return_value=None):
        with patch("sys.exit") as mock_exit:
            load_api_key()
            mock_exit.assert_called_once_with(1)

def test_parse_arguments():
    with patch("sys.argv", ["nerdprompt.py", "test question", "--raw"]):
        args = parse_arguments()
        assert args.question == "test question"
        assert args.raw is True

def test_get_question_initial():
    assert get_question("initial question") == "initial question"

def test_get_question_interactive():
    with patch("builtins.input", return_value="user question"):
        assert get_question() == "user question"

def test_get_question_paste_mode():
    with patch("sys.stdin.isatty", return_value=True):
        # Mocking input() to raise EOFError to end paste mode
        with patch("builtins.input", side_effect=["line 1", "line 2", EOFError]):
            assert get_question(paste_mode=True) == "line 1\nline 2"

@patch('nerdprompt.PerplexityWrapper')
def test_formatted_output(mock_wrapper, capsys):
    client = mock_wrapper.return_value
    client.code_extractor.return_value = {
        'text': 'no code',
        'code_blocks': []
    }
    client.markdown_to_ansi.return_value = 'ansi text'
    client.code_injector.return_value = 'injected text'
    client.remove_citations.return_value = 'final output'
    
    config = {'code_syntax_theme': 'dracula'}
    formatted_output(client, config, 'raw content')
    
    captured = capsys.readouterr()
    assert 'final output' in captured.out
