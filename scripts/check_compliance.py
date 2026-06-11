import re
import sys
import importlib
import os


def main():
    if not os.path.exists("SEMANTIC_PLAN.md"):
        print("SEMANTIC_PLAN.md not found, skipping compliance check.")
        sys.exit(0)

    with open("SEMANTIC_PLAN.md", "r") as f:
        content = f.read()

    modules = {}
    current_module = None

    for line in content.splitlines():
        m_mod = re.search(r"`([a-zA-Z0-9_\.]+)`", line)
        if line.startswith("## ") and m_mod:
            raw_mod = m_mod.group(1)
            prefix = raw_mod.split(".")[0]
            current_module = raw_mod.replace(prefix, "zero_grain", 1)
            if current_module not in modules:
                modules[current_module] = []
            continue
        elif line.startswith("## ") and not current_module:
            current_module = "zero_grain"
            if current_module not in modules:
                modules[current_module] = []
            continue

        if current_module:
            # We strictly enforce that APIs must be formatted as:
            # - [x] `func_name`
            # OR
            # - [ ] `func_name`
            # with possible backticks and a potential description afterwards
            m_item = re.match(r"^- \[[xX ]\] `?([a-zA-Z0-9_]+)`?(?:[: ].*)?", line)

            if m_item:
                funcs_part = m_item.group(1)
                funcs = [f.strip(" `") for f in funcs_part.split("/")]

                # Further filter out pseudo-tasks that don't match python identifier patterns
                for fn in funcs:
                    if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", fn):
                        modules[current_module].append(fn)

    missing = []
    total = 0
    implemented = 0

    if not modules or all(len(funcs) == 0 for funcs in modules.values()):
        print("No parsable API targets found in SEMANTIC_PLAN.md")
        sys.exit(0)

    for mod_name, funcs in modules.items():
        if not funcs:
            continue
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            missing.extend([f"{mod_name}.{f}" for f in funcs])
            total += len(funcs)
            continue

        for f_name in funcs:
            f_name = f_name.strip()
            if not f_name:
                continue
            total += 1
            if hasattr(mod, f_name):
                implemented += 1
            else:
                missing.append(f"{mod_name}.{f_name}")

    pct = (implemented / total) * 100 if total > 0 else 100.0

    current_todo = ""
    try:
        with open("GRAIN_TODO.md", "r") as f:
            current_todo = f.read()
    except FileNotFoundError:
        pass

    new_todo_lines = [
        "# GRAIN_TODO.md",
        "--- Compliance Report ---",
        f"Overall Compliance: {pct:.1f}%\n",
        "Breakdown by Module:",
    ]
    for mod_name, funcs in modules.items():
        if not funcs:
            continue
        prefix = "GRAIN_TODO.md".split("_")[0].lower()
        orig_mod_name = mod_name.replace("zero_grain", prefix)
        mod_implemented = sum(
            1 for f_name in funcs if f"{mod_name}.{f_name}" not in missing
        )
        mod_pct = (mod_implemented / len(funcs)) * 100 if funcs else 100.0
        new_todo_lines.append(
            f"  - {orig_mod_name}: {mod_pct:.1f}% ({mod_implemented}/{len(funcs)})"
        )

    if missing:
        new_todo_lines.append("\nMissing APIs:")
        for m in missing:
            new_todo_lines.append(f"  - {m}")

    new_todo = "\n".join(new_todo_lines) + "\n"

    if new_todo != current_todo:
        with open("GRAIN_TODO.md", "w") as f:
            f.write(new_todo)
        print("Updated GRAIN_TODO.md.")

    if missing:
        print(
            "COMPLIANCE FAILURE: The following expected APIs are missing or not exported:"
        )
        for m in missing:
            print(f"  - {m}")
        print(f"\nOverall Compliance: {pct:.1f}% ({implemented}/{total})")
        sys.exit(1)
    else:
        print(f"Compliance Check Passed! 100% compliant ({implemented}/{total} APIs).")
        if new_todo != current_todo:
            sys.exit(1)
        sys.exit(0)


if __name__ == "__main__":
    main()
