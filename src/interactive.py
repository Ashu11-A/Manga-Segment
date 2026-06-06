"""Interactive, arrow-key driven CLI wizard for Manga-Segment.

Launched automatically when ``main.py`` is started with no arguments. Instead of
duplicating every flag, the wizard *introspects* the argparse parser built by
``main.build_parser`` so it always reflects the real CLI:

    1. pick a command          (arrow keys)
    2. tick the options to set  (space to toggle — everything else keeps its default)
    3. fill in each picked option (select / yes-no / text, all pre-filled)

It then prints the exact equivalent command line and, on confirmation, returns
the assembled ``argv`` so ``main`` runs it (for ``train`` that means training
starts right away).
"""

from __future__ import annotations

import argparse
import shlex
import sys


# ── questionary loader ────────────────────────────────────────────────────────


def _require_questionary():
	try:
		import questionary

		return questionary
	except ImportError:
		print(
			"❌ questionary is not installed.\n"
			"   Run: uv add questionary  (or pip install questionary)",
			file=sys.stderr,
		)
		sys.exit(1)


# ── parser introspection helpers ──────────────────────────────────────────────


def _subparsers_action(parser: argparse.ArgumentParser) -> argparse._SubParsersAction:
	for action in parser._actions:
		if isinstance(action, argparse._SubParsersAction):
			return action
	raise RuntimeError("CLI parser has no subcommands.")


def _command_help(sub: argparse._SubParsersAction) -> dict[str, str]:
	"""Map each command name to its one-line help string."""
	return {sa.dest: (sa.help or "") for sa in sub._get_subactions()}


def _flag(action: argparse.Action) -> str:
	"""The canonical (long, if available) option string for an action."""
	longs = [opt for opt in action.option_strings if opt.startswith("--")]
	return longs[-1] if longs else action.option_strings[0]


def _is_boolean(action: argparse.Action) -> bool:
	return isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction))


def _options(subparser: argparse.ArgumentParser) -> list[argparse.Action]:
	"""User-settable optional actions for a subcommand (excludes -h/--help)."""
	return [
		action
		for action in subparser._actions
		if action.option_strings and action.dest != "help"
	]


def _discover_algos() -> list[str]:
	"""List algorithm plug-ins by directory name (no heavy imports)."""
	try:
		import pkgutil

		import algorithms

		names = [
			module.name
			for module in pkgutil.iter_modules(algorithms.__path__)
			if module.ispkg and not module.name.startswith("_")
		]
		return sorted(names) or ["yolo", "unet"]
	except Exception:
		return ["yolo", "unet"]


# ── per-option prompting ──────────────────────────────────────────────────────

_CANCELLED = object()


def _numeric_validator(action: argparse.Action):
	caster = action.type
	if caster not in (int, float):
		return None

	def validate(text: str):
		text = text.strip()
		if text == "":
			return True
		try:
			caster(text)
			return True
		except (TypeError, ValueError):
			return f"Enter a valid {caster.__name__} (or leave blank to skip)."

	return validate


def _prompt_option(q, action: argparse.Action, algos: list[str]):
	"""Prompt for a single option. Returns its value or ``_CANCELLED``."""
	flag = _flag(action)
	help_text = (action.help or "").strip()
	label = f"{flag}" + (f" — {help_text}" if help_text else "")

	# Yes/No flags (store_true / store_false).
	if _is_boolean(action):
		answer = q.confirm(label + " ?", default=bool(action.default)).ask()
		return _CANCELLED if answer is None else ("bool", answer)

	# The algorithm picker gets the discovered plug-in list.
	if action.dest == "algo":
		default = action.default if action.default in algos else algos[0]
		answer = q.select(label, choices=algos, default=default).ask()
		return _CANCELLED if answer is None else ("value", answer)

	# Fixed-choice options (e.g. --mode, --device) become a select.
	if action.choices:
		choices = list(action.choices)
		default = action.default if action.default in choices else None
		answer = q.select(label, choices=choices, default=default).ask()
		return _CANCELLED if answer is None else ("value", answer)

	# Everything else is free text, pre-filled with the current default.
	default_text = "" if action.default is None else str(action.default)
	answer = q.text(
		label,
		default=default_text,
		validate=_numeric_validator(action),
	).ask()
	return _CANCELLED if answer is None else ("value", answer)


def _append_to_argv(argv: list[str], action: argparse.Action, kind: str, value) -> None:
	flag = _flag(action)
	if kind == "bool":
		# Emit the flag only when it actually changes the default behaviour.
		if value != action.default:
			argv.append(flag)
		return

	text = "" if value is None else str(value).strip()
	if text == "":
		return  # blank → leave the default in place
	if action.nargs in ("*", "+"):
		argv.append(flag)
		argv.extend(text.split())
	else:
		argv.extend([flag, text])


# ── main wizard ───────────────────────────────────────────────────────────────

# Commands that drive their own interactive prompts end-to-end. The generic
# introspection wizard hands control to them right after the command is picked.
_SELF_INTERACTIVE_COMMANDS = {"download"}


def run_wizard(build_parser) -> list[str] | None:
	"""Drive the interactive selection and return the assembled ``argv``.

	Returns ``None`` if the user cancels (Ctrl-C / Esc) or declines to run.
	``build_parser`` is injected to avoid a circular import with ``main``.
	"""
	if not sys.stdin.isatty():
		print(
			"❌ No command given and stdin is not a terminal, so the interactive\n"
			"   wizard can't run. Pass a command instead, e.g.:\n"
			"       uv run python main.py train --algo yolo\n"
			"   See 'uv run python main.py --help' for all commands.",
			file=sys.stderr,
		)
		return None

	q = _require_questionary()
	parser = build_parser()
	sub = _subparsers_action(parser)
	helps = _command_help(sub)
	algos = _discover_algos()

	print("\n🎛️  Manga-Segment — interactive CLI")
	print("    (↑/↓ to move · Enter to confirm · Ctrl-C to cancel)\n")

	# ── Step 1: choose the command ────────────────────────────────────────
	command = q.select(
		"What would you like to do?",
		choices=[
			q.Choice(
				title=f"{name:<10} {helps.get(name, '')}".rstrip(),
				value=name,
			)
			for name in sub.choices
		],
	).ask()
	if command is None:
		return None

	# Some commands run their own purpose-built interactive flow (e.g. the
	# Roboflow download wizard). For those we skip the generic option-toggling
	# step and dispatch straight to the command, which owns all of its prompts.
	if command in _SELF_INTERACTIVE_COMMANDS:
		print()  # blank line before the command's own output
		return [command]

	subparser = sub.choices[command]
	options = _options(subparser)

	# ── Step 2: pick which options to customise ───────────────────────────
	argv: list[str] = [command]
	if options:
		to_set = q.checkbox(
			"Select options to customise (space to toggle; the rest use defaults):",
			choices=[
				q.Choice(
					title=(
						f"{_flag(action):<22}"
						f" [default: {'off' if _is_boolean(action) else (action.default if action.default is not None else 'auto')}]"
					),
					value=action,
				)
				for action in options
			],
		).ask()
		if to_set is None:
			return None

		# ── Step 3: fill in each chosen option (in declaration order) ─────
		chosen = set(to_set)
		for action in options:
			if action not in chosen:
				continue
			result = _prompt_option(q, action, algos)
			if result is _CANCELLED:
				print("\n✋  Cancelled.")
				return None
			kind, value = result
			_append_to_argv(argv, action, kind, value)

	# ── Step 4: show the resulting command and confirm ────────────────────
	command_line = "uv run python main.py " + shlex.join(argv)
	print("\n🧩  Resulting command:\n")
	print(f"    {command_line}\n")

	run_now = q.confirm("Run this now?", default=True).ask()
	if not run_now:
		print("👍  Not running. You can copy the command above and run it yourself.")
		return None

	print()  # blank line before the command's own output
	return argv
