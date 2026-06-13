import json
import ast
import sys

with open("snapshots/pygrain_v0.2.16.dev20260112.json") as f:
    data = json.load(f)

apis = data["categories"]["extras"]
expected = set([api["name"] for api in apis])

with open("src/zero_grain/__init__.py") as f:
    tree = ast.parse(f.read())

actual = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                if isinstance(node.value, ast.List):
                    actual = set(
                        getattr(elt, "value", getattr(elt, "s", None))
                        for elt in node.value.elts
                    )

missing = expected - actual
covered = expected.intersection(actual)

print(f"Total APIs: {len(expected)}")
print(f"Covered APIs: {len(covered)}")
print(f"Missing APIs: {missing}")
print(f"Compliance: {len(covered) / len(expected) * 100:.2f}%")

if missing:
    sys.exit(1)
