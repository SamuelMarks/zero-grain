"""Script to automatically add placeholder docstrings to Python files."""

import ast
import os


def add_docstrings(filepath: str) -> None:
    """Add placeholder docstrings to missing classes and functions in a file.

    Args:
        filepath: The path to the Python file to be modified.

    """
    with open(filepath, "r") as f:
        source = f.read()

    lines = source.split("\n")
    tree = ast.parse(source)

    inserts = []

    def get_docstring(node):
        """Retrieve docstring for an AST node.

        Args:
            node: The AST node to check.

        Returns:
            The docstring if it exists, None otherwise.

        """
        return ast.get_docstring(node)

    if not get_docstring(tree):
        inserts.append((1, '"""Module docstring."""'))

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not get_docstring(node):
                # find the line after the definition
                lineno = node.body[0].lineno
                indent = " " * node.body[0].col_offset
                inserts.append((lineno, f'{indent}"""Docstring for {node.name}."""'))

    inserts.sort(key=lambda x: x[0], reverse=True)

    for lineno, text in inserts:
        lines.insert(lineno - 1, text)

    with open(filepath, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    for root, _, files in os.walk("src"):
        for file in files:
            if file.endswith(".py"):
                add_docstrings(os.path.join(root, file))

    for root, _, files in os.walk("tests"):
        for file in files:
            if file.endswith(".py"):
                add_docstrings(os.path.join(root, file))
