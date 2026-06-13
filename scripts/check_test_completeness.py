import ast
import os
import sys


def main():
    expected_ops = set()
    with open("src/zero_grain/core.py") as f:
        tree = ast.parse(f.read())
        for node in ast.walk(tree):
            # find all classes imported that end with Operation
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name.endswith("Operation") and alias.name != "Operation":
                        expected_ops.add(alias.name)

    # check tests/
    tested_ops = set()
    for root, dirs, files in os.walk("tests"):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                with open(path) as f:
                    content = f.read()
                    for op in expected_ops:
                        if op in content:
                            tested_ops.add(op)

    missing = expected_ops - tested_ops
    if missing:
        print(f"Missing tests for operations: {missing}")
        sys.exit(1)
    else:
        print("All operations have tests.")
        sys.exit(0)


if __name__ == "__main__":
    main()
