#!/usr/bin/env python3
import ast
import os
import sys
import sysconfig
from pathlib import Path

# Standard library modules
std_lib_dir = sysconfig.get_paths()["stdlib"]
STD_LIB = (
    set(sys.builtin_module_names)
    | {p.stem for p in Path(std_lib_dir).glob("*.py")}
    | {p.name for p in Path(std_lib_dir).iterdir() if p.is_dir()}
)

# Allowed imports in non-test code
ALLOWED_IMPORTS = {
    "numpy",
    "pydantic",
    "ml_switcheroo",
    "ml_switcheroo_compiler",
    "cdd",
    "zero_jax",
}


def check_file(filepath: Path) -> list[str]:
    errors = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(filepath))
    except SyntaxError as e:
        return [f"{filepath}: Syntax error: {e}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base_module = alias.name.split(".")[0]
                if base_module not in STD_LIB and base_module not in ALLOWED_IMPORTS:
                    errors.append(
                        f"{filepath}:{node.lineno}: Disallowed 3rd-party import '{alias.name}'"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                continue  # Relative import
            if node.module:
                base_module = node.module.split(".")[0]
                if base_module not in STD_LIB and base_module not in ALLOWED_IMPORTS:
                    errors.append(
                        f"{filepath}:{node.lineno}: Disallowed 3rd-party import '{node.module}'"
                    )
    return errors


def main() -> int:
    exit_code = 0
    src_dir = Path("src")

    if not src_dir.exists():
        return 0

    for filepath in src_dir.rglob("*.py"):
        errors = check_file(filepath)
        for error in errors:
            print(error, file=sys.stderr)
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
