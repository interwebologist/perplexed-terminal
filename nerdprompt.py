"""
AI-enhanced terminal chat client with fancy custom terminal formatting.
Refactored for PEP 8 compliance and Clean Code principles.
"""

import argparse
import logging
import os
import re
import sys
import time
import traceback
from pathlib import Path

import yaml
from cerberus import Validator
from dotenv import load_dotenv
from openai import OpenAI
from pygments import highlight
from pygments.formatters import Terminal256Formatter
from pygments.lexers import get_lexer_by_name, guess_lexer

ANSI_CODES = {
    'bold': '\033[1m',
    'italic': '\033[3m',
    'bold_italic': '\033[1;3m',
    'underline': '\033[4m',
    'green': '\033[32m',
    'blue': '\033[34m',
    'red': '\033[31m',
    'yellow': '\033[33m',
    'magenta': '\033[35m',
    'cyan': '\033[36m',
    'white': '\033[37m',
    'black': '\033[30m',
    'bg_red': '\033[41m',
    'bg_green': '\033[42m',
    'bg_yellow': '\033[43m',
    'bg_blue': '\033[44m',
    'bg_magenta': '\033[45m',
    'bg_cyan': '\033[46m',
    'bg_white': '\033[47m',
    'bg_black': '\033[40m',
    'strikethrough': '\033[9m',
    'reverse': '\033[7m',
    'conceal': '\033[8m',
    'reset': '\033[0m',
}


class PerplexityWrapper:
    """Wrapper for Perplexity/OpenAI API calls and markdown processing."""

    def __init__(self, config, api_key, prompt_type='default'):
        self.api_key = api_key
        system_prompt = config['system_content'][prompt_type]
        self.messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]

    def ask(self, config):
        """Perform a chat completion request."""
        client = OpenAI(api_key=self.api_key, base_url=f"{config['llm_url']}")
        start_time = time.perf_counter()
        response = client.chat.completions.create(
            model=f"{config['llm_model']}",
            messages=self.messages,
        )
        end_time = time.perf_counter()
        api_time = end_time - start_time
        return response, api_time

    def message_appender(self, role, content):
        """Add a message to the conversation history."""
        self.messages.append({
            "role": role,
            "content": content
        })
        return self.messages

    def clear_history(self):
        """Reset conversation while keeping system message."""
        system_msg = self.messages[0]
        self.messages = [system_msg]

    def markdown_to_ansi(self, ANSI_CODES, config, markdown_text):
        """Convert basic markdown elements to ANSI terminal sequences."""
        # Process headers
        if '#' in markdown_text:
            header_ansi = {}
            for level in range(1, 7):
                header_name = f"header_{level}"
                if config.get(header_name):
                    ansi_codes_list = config[header_name]
                    prefix = ''.join(ANSI_CODES[code] for code in ansi_codes_list)
                    suffix = ANSI_CODES['reset']
                    header_ansi[level] = (prefix, suffix)
                else:
                    header_ansi[level] = ('', '')

            def replace_header(match):
                hashes = match.group(1)
                text = match.group(2)
                level = len(hashes)
                if level in header_ansi:
                    prefix, suffix = header_ansi[level]
                    return f"{prefix}{text}{suffix}"
                return match.group(0)

            markdown_text = re.sub(r'(?m)^(#{1,6}) (.+)$', replace_header, markdown_text)

        # Process bold and italic
        if '*' in markdown_text:
            reset = ANSI_CODES['reset']
            bold_italic = f"{ANSI_CODES['bold_italic']}\\1{reset}"
            bold = f"{ANSI_CODES['bold']}\\1{reset}"
            italic = f"{ANSI_CODES['italic']}\\1{reset}"

            markdown_text = re.sub(r'\*\*\*(.+?)\*\*\*', bold_italic, markdown_text)
            markdown_text = re.sub(r'\*\*(.+?)\*\*', bold, markdown_text)
            markdown_text = re.sub(r'\*(.+?)\*', italic, markdown_text)

        # Process dividers
        divider_choice = config['ascii_divider_choice']
        divider_text = config["ascii_dividers"][divider_choice]

        if config.get("dividers_color"):
            color = ANSI_CODES[config['dividers_color']]
            divider_text = f"{color}{divider_text}{ANSI_CODES['reset']}"

        if '---' in markdown_text:
            markdown_text = re.sub(r'^---$', rf"{divider_text}", markdown_text, flags=re.MULTILINE)

        # Process bullet points
        if '\n-' in markdown_text or markdown_text.startswith('-'):
            bullet = config['bullet_point_unicode']
            markdown_text = re.sub(r'^\s*-\s+', f" {bullet} ", markdown_text, flags=re.MULTILINE)

        return markdown_text

    def remove_citations(self, text):
        """Remove citation markers like [1], [2] from text."""
        return re.sub(r'\[\d+\]', '', text)

    def code_extractor(self, text):
        """Extract code blocks and replace them with placeholders."""
        code_blocks = []
        if '```' not in text:
            return {"text": text, "code_blocks": code_blocks}

        pattern = r'```[\s\S]*?```'
        matches = list(re.finditer(pattern, text))

        for i, match in enumerate(reversed(matches)):
            code_blocks.insert(0, match.group(0))
            placeholder = f'<CODE__REMOVED__{len(matches) - 1 - i}>'
            text = text[:match.start()] + placeholder + text[match.end():]

        return {"text": text, "code_blocks": code_blocks}

    def code_injector(self, doc_data):
        """Inject processed code blocks back into the text."""
        text = doc_data['ansi_converted_text']
        for i, code_block in enumerate(doc_data['code_blocks']):
            text = text.replace(f'<CODE__REMOVED__{i}>', code_block)
        return text


class CodeProcesser:
    """Handles code extraction, syntax highlighting, and formatting."""

    def extract_code_type_and_syntax(self, input_string):
        """Extract language and content from a markdown code block."""
        pattern = r'```([A-Za-z]*)([\s\S.]*)```'
        match = re.search(pattern, input_string, re.DOTALL)
        if match:
            return {
                'code_type': match.group(1),
                'code_syntax': match.group(2)
            }
        return None

    def syntax_highlighter(self, config, code_data):
        """Apply syntax highlighting using Pygments."""
        try:
            lexer = get_lexer_by_name(code_data['code_type'])
        except Exception:
            lexer = guess_lexer(code_data['code_syntax'])
        finally:
            theme = config['code_syntax_theme']
            formatter = Terminal256Formatter(style=theme)

        highlighted = highlight(code_data['code_syntax'], lexer, formatter)
        code_data['highlighted_code'] = highlighted
        return code_data

    def rebuild_code_type_and_syntax(self, ANSI_CODES, config, code_data):
        """Rebuild highlighted code block with custom dividers."""
        code_type = code_data['code_type'].capitalize()
        code_syntax = code_data['highlighted_code']
        choice = config["code_divider_choice"]
        divider = config["code_dividers"][choice]

        if config.get("code_dividers_color"):
            color = ANSI_CODES[config['code_dividers_color']]
            reset = ANSI_CODES['reset']
            divider = f"{color}{divider}{reset}"

        return f"\n{divider}\n\n{code_type} Code:\n{code_syntax}\n{divider}\n"


class ConfigEater:
    """Handles configuration loading and validation."""

    def parse_config(self):
        """Load configuration from config.yaml."""
        script_path = Path(__file__).absolute().parent
        config_path = script_path / 'config.yaml'
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def check_config(self, ANSI_CODES, config_dict):
        """Validate configuration using Cerberus schema."""
        ansi_list = list(ANSI_CODES.keys())
        schema = {
            'llm_url': {'type': 'string', 'required': True},
            'llm_model': {'type': 'string', 'required': True},
            'remove_perplexity_citations': {'type': 'boolean', 'required': True},
            'dividers_color': {'type': 'string', 'required': True},
            'code_dividers_color': {'type': 'string', 'required': True},
            'code_syntax_theme': {'type': 'string', 'required': True},
            'system_content': {'type': 'dict', 'required': True},
            'bullet_point_unicode': {'type': 'string', 'required': True},
            'header_1': {'type': 'list', 'schema': {'type': 'string'}, 'required': True, 'allowed': ansi_list},
            'header_2': {'type': 'list', 'schema': {'type': 'string'}, 'required': True, 'allowed': ansi_list},
            'header_3': {'type': 'list', 'schema': {'type': 'string'}, 'required': True, 'allowed': ansi_list},
            'header_4': {'type': 'list', 'schema': {'type': 'string'}, 'required': True, 'allowed': ansi_list},
            'header_5': {'type': 'list', 'schema': {'type': 'string'}, 'required': True, 'allowed': ansi_list},
            'header_6': {'type': 'list', 'schema': {'type': 'string'}, 'required': True, 'allowed': ansi_list},
            'ascii_divider_position': {'type': 'string', 'allowed': ['left', 'center', 'right'], 'required': True},
            'ascii_divider_choice': {'type': 'integer', 'required': True},
            'ascii_dividers': {
                'type': 'dict',
                'keysrules': {'type': 'integer'},
                'valuesrules': {'type': 'string'},
                'required': True
            },
            'code_divider_choice': {'type': 'integer', 'required': True},
            'code_dividers': {
                'type': 'dict',
                'keysrules': {'type': 'integer'},
                'valuesrules': {'type': 'string'},
                'required': True
            },
        }

        validator = Validator(schema)
        if not validator.validate(config_dict):
            raise ValueError(f"Config validation error: {validator.errors}")


def test_256_term_colors():
    """Utility to display all 256 terminal colors."""
    for i in range(256):
        print(f"\033[48;5;{i}m {i:3d} \033[0m", end=' ')
        if (i + 1) % 16 == 0:
            print()
    print()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='AI-enhanced terminal chat client with fancy customer terminal formatting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Examples:
  nerdprompt.py "What is Docker?"                    # Formatted output for humans
  nerdprompt.py --raw "Explain Python decorators"   # Raw markdown for AI agents
  nerdprompt.py -p concise "tell me about the weather"  # Use concise system prompt
  nerdprompt.py --paste                             # Paste mode - read multiline input
  cat code.py | nerdprompt.py --paste --raw         # Pipe content in paste mode
  nerdprompt.py -m sonar-pro "complex query"        # Use sonar-pro model
  nerdprompt.py -m sonar --raw "simple query"       # Use sonar model with raw output

For AI agents: Use --raw flag. For multiline input: Use --paste flag.'''
    )
    parser.add_argument('question', nargs='?',
                        help='Question to ask the AI (optional, will prompt if not provided)')
    parser.add_argument('-r', '--raw', action='store_true',
                        help='output raw markdown (recommended for AI agents)')
    parser.add_argument('-p', '--prompt', default='default',
                        help='system prompt to use (default: default)')
    parser.add_argument('-m', '--model', default=None,
                        help='LLM model override (sonar, sonar-pro, sonar-reasoning, etc.)')
    parser.add_argument('-n', '--nothread', action='store_true',
                        help='no thread mode, return only the last answer')
    parser.add_argument('--paste', action='store_true',
                        help='paste mode - read multiline input until EOF (Ctrl+D)')
    return parser.parse_args()


def load_api_key():
    """Load API key from environment variables."""
    load_dotenv()
    api_key = os.getenv("API_KEY")
    if not api_key:
        logging.error("Error: API_KEY is missing from environment. Configure .env file.")
        sys.exit(1)
    return api_key


def get_question(initial_question=None, paste_mode=False):
    """Retrieve user question from arguments, stdin, or interactive prompt."""
    if initial_question:
        return initial_question

    if paste_mode:
        if not sys.stdin.isatty():
            content = sys.stdin.read()
            if not content.strip():
                print("No content provided via pipe.")
                sys.exit(1)
            try:
                tty_fd = os.open('/dev/tty', os.O_RDONLY)
                sys.stdin = os.fdopen(tty_fd, 'r')
            except (OSError, FileNotFoundError):
                sys.stdin.piped_input = True
            return content

        print("📋 Paste mode - enter your content (press Ctrl+D when done):")
        try:
            lines = []
            while True:
                lines.append(input())
        except EOFError:
            content = '\n'.join(lines)
            if not content.strip():
                print("No content provided.")
                return get_question(paste_mode=paste_mode)
            return content

    question = input("Please enter your question: ").strip()
    while not question:
        print("You must enter a question.")
        question = input("Please enter your question: ").strip()
    return question


def formatted_output(client, config, raw_content):
    """Process and print formatted LLM response."""
    start = time.perf_counter()

    doc_data = client.code_extractor(raw_content)
    ansi_text = client.markdown_to_ansi(ANSI_CODES, config, doc_data['text'])
    doc_data['ansi_converted_text'] = ansi_text

    if doc_data['code_blocks']:
        processor = CodeProcesser()
        formatted_blocks = []
        for code in doc_data['code_blocks']:
            info = processor.extract_code_type_and_syntax(code)
            highlighted = processor.syntax_highlighter(config, info)
            rebuilt = processor.rebuild_code_type_and_syntax(ANSI_CODES, config, highlighted)
            formatted_blocks.append(rebuilt)
        doc_data['code_blocks'] = formatted_blocks

    final_text = client.code_injector(doc_data)
    final_text = client.remove_citations(final_text)

    print(final_text)
    return time.perf_counter() - start


def initialize_app():
    """Initialize configuration, arguments, and LLM client."""
    args = parse_arguments()
    config_eater = ConfigEater()
    config = config_eater.parse_config()
    config_eater.check_config(ANSI_CODES, config)

    if args.model:
        config['llm_model'] = args.model

    try:
        api_key = load_api_key()
        client = PerplexityWrapper(config, api_key, args.prompt)
        return config, args, client
    except Exception as e:
        print(f"Initialization error: {e}")
        traceback.print_exc()
        sys.exit(1)


def process_and_print_response(client, config, args, content):
    """Handle a single prompt cycle: API call, formatting, and timing."""
    client.message_appender("user", content)
    print("🔍 Researching...", end='', flush=True)
    response, api_time = client.ask(config)
    print("\r🔍 Researching... done")

    raw_content = response.choices[0].message.content
    client.message_appender("assistant", raw_content)

    if args.raw:
        start = time.perf_counter()
        print(raw_content)
        proc_time = time.perf_counter() - start
    else:
        proc_time = formatted_output(client, config, raw_content)

    total_time = api_time + proc_time
    print(f"\nAPI: {api_time:.2f}s | Proc: {proc_time * 1000:.2f}ms | Total: {total_time:.2f}s")

    if args.nothread:
        sys.exit(0)


def get_next_input(args, client):
    """Determine the next user input or whether to exit/clear context."""
    try:
        choice = input("y=continue thread | n=exit | c=clear context: ").strip().lower()
    except EOFError:
        if hasattr(sys.stdin, 'piped_input'):
            print("\nPiped input processed. Cannot continue interactive session.")
        sys.exit(0)

    if choice == 'n':
        sys.exit(0)
    elif choice == 'c':
        client.clear_history()
        return get_question(paste_mode=args.paste), True
    elif choice == 'y':
        return get_question(paste_mode=args.paste), False
    else:
        return get_next_input(args, client)


def run_chat_loop(client, config, args):
    """Main application loop."""
    # First question handling
    content = get_question(args.question, args.paste)
    args.question = None  # Clear so subsequent calls use interactive input

    while True:
        try:
            process_and_print_response(client, config, args, content)
            content, _ = get_next_input(args, client)
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            break


def main():
    """Main entry point."""
    config, args, client = initialize_app()
    run_chat_loop(client, config, args)


if __name__ == "__main__":
    main()
