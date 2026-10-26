#!/usr/bin/env python3
"""Script to verify that ``src/`` only imports Python standard library or approved ecosystem modules."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path
import sys
import sysconfig

_STDLIB_DIR = Path(sysconfig.get_paths()["stdlib"])
_DYNAMIC_STDLIB_NAMES: set[str] = {
    entry.stem.split(".")[0]
    for entry in _STDLIB_DIR.iterdir()
    if (entry.name.startswith("_") and not entry.name.startswith("__"))
    or entry.is_dir()
    or entry.suffix in (".py", ".so", ".pyd")
}

STD_LIB: frozenset[str] = frozenset(
    set(sys.builtin_module_names)
    | _DYNAMIC_STDLIB_NAMES
    | {
        "__future__",
        "abc",
        "argparse",
        "ast",
        "asyncio",
        "base64",
        "bisect",
        "builtins",
        "collections",
        "contextlib",
        "contextvars",
        "copy",
        "csv",
        "ctypes",
        "dataclasses",
        "datetime",
        "decimal",
        "difflib",
        "dis",
        "enum",
        "errno",
        "fcntl",
        "fnmatch",
        "fractions",
        "functools",
        "gc",
        "glob",
        "gzip",
        "hashlib",
        "heapq",
        "hmac",
        "html",
        "http",
        "importlib",
        "inspect",
        "io",
        "itertools",
        "json",
        "keyword",
        "linecache",
        "locale",
        "logging",
        "lzma",
        "math",
        "mmap",
        "multiprocessing",
        "numbers",
        "operator",
        "os",
        "pathlib",
        "pickle",
        "pkgutil",
        "platform",
        "plistlib",
        "pprint",
        "profile",
        "pstats",
        "pty",
        "pwd",
        "queue",
        "quopri",
        "random",
        "re",
        "reprlib",
        "resource",
        "runpy",
        "sched",
        "secrets",
        "select",
        "selectors",
        "shelve",
        "shlex",
        "shutil",
        "signal",
        "site",
        "socket",
        "sqlite3",
        "ssl",
        "stat",
        "statistics",
        "string",
        "stringprep",
        "struct",
        "subprocess",
        "sys",
        "sysconfig",
        "syslog",
        "tarfile",
        "tempfile",
        "termios",
        "textwrap",
        "threading",
        "time",
        "timeit",
        "tokenize",
        "token",
        "traceback",
        "tracemalloc",
        "tty",
        "turtle",
        "types",
        "typing",
        "typing_extensions",
        "unicodedata",
        "unittest",
        "urllib",
        "uuid",
        "venv",
        "warnings",
        "wave",
        "weakref",
        "webbrowser",
        "winreg",
        "winsound",
        "wsgiref",
        "xdrlib",
        "xml",
        "xmlrpc",
        "zipapp",
        "zipfile",
        "zipimport",
        "zlib",
        "zoneinfo",
        "cdd",
        "ml_ecosystem_snapshots",
        "ml_switcheroo_compiler",
        "ml_switcheroo_ir",
        "pydantic",
        "zero_chex",
        "zero_flax",
        "zero_grain",
        "zero_jax",
        "zero_optax",
        "zero_orbax",
    }
)


def check_file(filepath: Path) -> list[str]:
    """Check ``filepath`` for any disallowed non-stdlib imports.

    Args:
        filepath (Path): Python source file to inspect.

    Returns:
        list[str]: List of violation messages.
    """
    try:
        tree = ast.parse(filepath.read_text(encoding="utf-8"), filename=str(filepath))
    except SyntaxError as exc:
        return [f"{filepath}:{exc.lineno}: SyntaxError: {exc.msg}"]

    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_pkg = alias.name.split(".")[0]
                if top_pkg not in STD_LIB and not top_pkg.startswith("zero_"):
                    errors.append(f"{filepath}:{node.lineno}: Disallowed import '{alias.name}'")
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module is not None:
                top_pkg = node.module.split(".")[0]
                if top_pkg not in STD_LIB and not top_pkg.startswith("zero_"):
                    errors.append(f"{filepath}:{node.lineno}: Disallowed import from '{node.module}'")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    """Verify that all Python files in target directories only import stdlib modules.

    Args:
        argv (Sequence[str] | None): Optional CLI arguments specifying target paths.

    Returns:
        int: Zero if compliant, or 1 if any disallowed imports are found.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    targets = [Path(a) for a in args] if args else [Path("src")]
    exit_code = 0

    for target in targets:
        if not target.exists():
            print(f"{target}: Path does not exist", file=sys.stderr)
            exit_code = 1
            continue
        files = [target] if target.is_file() else sorted(target.rglob("*.py"))
        for filepath in files:
            for error in check_file(filepath):
                print(error, file=sys.stderr)
                exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
