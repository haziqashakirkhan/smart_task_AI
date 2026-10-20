"""Command-line interface."""

import argparse
import os
import sys
from pathlib import Path

from .formatters import RENDERERS
from .llm import ConfigError

EXAMPLE = (
    "I need to finish my assignment tomorrow, buy groceries today, "
    "and prepare for my meeting next week."
)


def _read_input(args) -> str:
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    if args.text:
        return " ".join(args.text)
    if not sys.stdin.isatty():  # piped input
        return sys.stdin.read()
    print("Describe your tasks in plain language (press Enter to submit):")
    print(f'e.g. "{EXAMPLE}"\n')
    return input("> ")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="task_organizer",
        description="Turn a messy description of your tasks into a prioritized to-do list.",
    )
    p.add_argument("text", nargs="*", help="Task description (omit to be prompted, or pipe stdin)")
    p.add_argument("-f", "--file", help="Read the task description from a text file")
    p.add_argument(
        "--format", choices=sorted(RENDERERS), default="text", help="Output format (default: text)"
    )
    p.add_argument("-o", "--output", help="Also save the result to this file")
    p.add_argument("--model", help="Model name (overrides LLM_MODEL)")
    p.add_argument("--serve", action="store_true", help="Launch the cute web UI in your browser")
    p.add_argument("--port", type=int, default=8000, help="Port for --serve (default: 8000)")
    p.add_argument("--no-browser", action="store_true", help="With --serve: don't auto-open the browser")
    p.add_argument("--graph", action="store_true", help="Print the workflow as a Mermaid diagram and exit")
    return p


def main(argv=None) -> int:
    # Make emoji output safe on Windows consoles.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    args = build_parser().parse_args(argv)
    if args.model:
        os.environ["LLM_MODEL"] = args.model

    from .graph import build_graph, organize_tasks  # imported late so --help is fast

    if args.serve:
        from .web import serve

        serve(port=args.port, open_browser=not args.no_browser)
        return 0

    if args.graph:
        print(build_graph().get_graph().draw_mermaid())
        return 0

    try:
        user_input = _read_input(args)
        state = organize_tasks(user_input)
    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    except Exception as e:  # API/network failures
        print(f"Something went wrong while contacting the AI provider: {e}", file=sys.stderr)
        return 1

    output = RENDERERS[args.format](state.get("final_tasks", []))
    print(output)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
        print(f"\nSaved to {args.output}", file=sys.stderr)
    return 0
