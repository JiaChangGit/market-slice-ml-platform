#!/usr/bin/env python3
"""Reject execution-oriented API vocabulary from implementation source files."""

from __future__ import annotations

import ast
import re
from pathlib import Path

FORBIDDEN = {
    "placeOrder",
    "reqAccount",
    "reqAccountSummary",
    "reqPositions",
    "cancelOrder",
    "openOrders",
    "reqExecutions",
    "submit_order",
    "market_order",
    "limit_order",
    "position_size",
    "entry_signal",
    "exit_signal",
    "stop_loss",
    "take_profit",
}


def code_lines(path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return list(enumerate(text.splitlines(), 1))
    docstring_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    end = getattr(body[0], "end_lineno", body[0].lineno)
                    docstring_lines.update(range(body[0].lineno, end + 1))
    return [
        (number, line)
        for number, line in enumerate(text.splitlines(), 1)
        if number not in docstring_lines and not line.lstrip().startswith("#")
    ]


def scan(root: Path) -> list[str]:
    violations: list[str] = []
    if not root.exists():
        return violations
    patterns = {
        token: re.compile(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])")
        for token in FORBIDDEN
    }
    for path in root.rglob("*.py"):
        for line_number, line in code_lines(path):
            for token, pattern in patterns.items():
                if pattern.search(line):
                    violations.append(f"{path}:{line_number}: forbidden token {token}")
    return violations


def main() -> int:
    violations = scan(Path("src")) + scan(Path("tests"))
    if violations:
        print("No-trading guard failed:")
        print("\n".join(violations))
        return 1
    print("No-trading static guard passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
