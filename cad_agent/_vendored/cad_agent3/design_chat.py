"""design_chat.py — interactive chat for CAD design.

Wraps the Orchestrator with a conversation loop. Each user turn:

1. Add the user message to the conversation history
2. Hand the latest message + history to the Orchestrator
3. After execution, render a preview if the part changed
4. Print a brief summary
5. Wait for the next message

This is the user-facing entry point. CLI today; can be wrapped by a
web UI later (the chat state is just a list, easy to serialize).

Run:
    python -m cad_agent3.design_chat

Inside the chat, special commands (typed alone on a line):
    /history     show operation history
    /summary     show current part summary
    /save NAME   emit standalone .py script + render preview
    /reset       discard current part, start over
    /context PATH load a STEP/FCStd as design context
    /help        show commands
    /quit        exit
"""

from __future__ import annotations

import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import List, Optional

from .orchestrator import Orchestrator
from .builder import Builder


# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------

@dataclass
class ChatTurn:
    role: str       # "user" | "assistant" | "system"
    text: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ChatState:
    turns: List[ChatTurn] = field(default_factory=list)
    out_dir: str = "design_out"

    def add_user(self, text: str):
        self.turns.append(ChatTurn(role="user", text=text))

    def add_assistant(self, text: str):
        self.turns.append(ChatTurn(role="assistant", text=text))

    def add_system(self, text: str):
        self.turns.append(ChatTurn(role="system", text=text))


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------

def _cmd_history(state: ChatState, orch: Orchestrator, arg: str) -> str:
    return orch.builder.history_summary()


def _cmd_summary(state: ChatState, orch: Orchestrator, arg: str) -> str:
    return orch.builder.summary()


def _cmd_save(state: ChatState, orch: Orchestrator, arg: str) -> str:
    name = arg.strip() or "design"
    if orch.builder.part is None:
        return "no part to save — describe a design first."
    os.makedirs(state.out_dir, exist_ok=True)
    py_path = os.path.join(state.out_dir, f"{name}.py")
    img_path = os.path.join(state.out_dir, f"{name}.png")
    orch.builder.emit(py_path)
    rendered = orch.builder.render(img_path)
    return f"saved {py_path}\nrendered {rendered}"


def _cmd_reset(state: ChatState, orch: Orchestrator, arg: str) -> str:
    orch.builder = Builder(verbose=orch.verbose)
    return "reset — current part discarded; context preserved if any was loaded"


def _cmd_context(state: ChatState, orch: Orchestrator, arg: str) -> str:
    path = arg.strip()
    if not path:
        return "usage: /context PATH"
    if not os.path.isfile(path):
        return f"file not found: {path}"
    try:
        orch.builder.start_from_scan(path)
        bb = orch.builder.context_bbox()
        msg = f"loaded context from {path}\n"
        if bb:
            sx = bb[3] - bb[0]; sy = bb[4] - bb[1]; sz = bb[5] - bb[2]
            msg += f"context bbox: {sx:.1f} × {sy:.1f} × {sz:.1f} mm"
        return msg
    except Exception as e:
        return f"failed to load context: {type(e).__name__}: {e}"


def _cmd_help(state: ChatState, orch: Orchestrator, arg: str) -> str:
    return (
        "Commands:\n"
        "  /history       — show operation history\n"
        "  /summary       — show current part summary\n"
        "  /save NAME     — emit .py + render preview to NAME.py / NAME.png\n"
        "  /reset         — discard current part\n"
        "  /context PATH  — load STEP/FCStd as design context\n"
        "  /help          — show this help\n"
        "  /quit          — exit\n"
        "\n"
        "Anything else is treated as a design request:\n"
        "  > a 50×30×5 plate with 4 corner M3 holes\n"
        "  > add a 5mm fillet to the vertical edges\n"
        "  > drill a 6mm hole in the center\n"
    )


SLASH_COMMANDS = {
    "/history": _cmd_history,
    "/summary": _cmd_summary,
    "/save": _cmd_save,
    "/reset": _cmd_reset,
    "/context": _cmd_context,
    "/help": _cmd_help,
}


# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------

def chat_session(out_dir: str = "design_out",
                  verbose: bool = True,
                  auto_render: bool = True) -> None:
    """Run an interactive design chat session on stdin/stdout."""
    state = ChatState(out_dir=out_dir)
    orch = Orchestrator(verbose=verbose)
    state.add_system("design_chat session started")

    print("=== cad_agent3 design chat ===")
    print("Type /help for commands, or describe what you want to design.")
    print("Type /quit (or Ctrl-D) to exit.")
    print()

    turn_idx = 0
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        # Slash command?
        if line.startswith("/"):
            head = line.split(None, 1)
            cmd, arg = head[0].lower(), (head[1] if len(head) > 1 else "")
            if cmd in ("/quit", "/exit", "/bye"):
                print("bye.")
                break
            handler = SLASH_COMMANDS.get(cmd)
            if handler is None:
                print(f"unknown command: {cmd}. /help for list.")
                continue
            response = handler(state, orch, arg)
            print(response)
            print()
            continue

        # Treat as a design request
        state.add_user(line)
        turn_idx += 1
        try:
            outcome = orch.handle(line)
        except Exception as e:
            print(f"error: {type(e).__name__}: {e}")
            state.add_assistant(f"(error: {e})")
            continue

        # Print results
        any_error = False
        for r in outcome["results"]:
            if isinstance(r, dict) and r.get("error"):
                print(f"  step error: {r['error']}")
                any_error = True
        print(outcome["current"])

        # Auto-render if part changed and we're not in error state
        if auto_render and not any_error and orch.builder.part is not None:
            os.makedirs(out_dir, exist_ok=True)
            preview_path = os.path.join(out_dir, f"turn_{turn_idx:03d}.png")
            try:
                rendered = orch.builder.render(preview_path)
                print(f"  preview: {rendered}")
            except Exception as e:
                print(f"  preview failed: {e}")

        state.add_assistant(outcome["current"])
        print()


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        description="Interactive design chat for cad_agent3.")
    p.add_argument("--out", default="design_out",
                   help="Output directory for previews and saved scripts")
    p.add_argument("--no-render", action="store_true",
                   help="Disable auto-render after each turn")
    p.add_argument("--quiet", action="store_true",
                   help="Less output from the builder/orchestrator")
    args = p.parse_args(argv)
    chat_session(out_dir=args.out,
                  verbose=not args.quiet,
                  auto_render=not args.no_render)


if __name__ == "__main__":
    main()
