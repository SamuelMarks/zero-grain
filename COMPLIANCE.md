# API Compliance Declaration

`zero-grain` is a standalone, zero-runtime-dependency implementation of Google's **Grain (`grain` / `grain.python`)** API surface and execution semantics, grounded against `ml-switcheroo-ir` and `ml-ecosystem-snapshots` (`v0.0.4`).

- **100% Documentation Coverage**: Maintained across module, file, class, function, method, argument, and return annotations (`interrogate -vv -f 100` & `scripts/check_docstrings.py`).
- **100% Test Coverage**: Maintained across function, line, and branch metrics (`pytest --cov=src --cov=scripts --cov-branch --cov-fail-under=100`).
- **Strong Static Typing**: Maintained under `mypy --strict` with zero `Any`, `object`, `cast`, or dynamic reflection (`getattr`/`setattr`) in `src/` and `scripts/`.
- **Zero 3rd Party Runtime Dependencies**: Maintained (`requirements.txt` has zero runtime dependencies; verified by `scripts/check_deps.py`).
