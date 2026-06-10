# Zero Framework API Shell

> **Note:** This repository is an API-compatible shell. All underlying math, autodiff, and graph execution has been migrated to the [ml-switcheroo-compiler](https://github.com/SamuelMarks/ml-switcheroo-compiler) backend. This repository purely implements frontend routing and syntactic parity for the target framework.

# zero-grain

zero-grain is a 100% dependency-free (except for [numpy](https://numpy.org/)), API-compatible implementation of [Google's Grain](https://github.com/google/grain) data loader framework.

### Why does this exist?

In modern machine learning, deploying to edge devices, [WASM](https://webassembly.org/) runtimes, or strict containerized environments is heavily impeded by dependency bloat. The official [grain framework](https://github.com/google/grain) is incredibly powerful, but it transitively pulls in massive ecosystem dependencies (like heavy C++ file extensions for array_record, absl-py, and more) that make it completely impossible to run in environments like [pyodide](https://pyodide.org/) ([WebAssembly](https://webassembly.org/)) or lightweight mobile environments.

As part of the Abstract ML Compiler ([ml-switcheroo](https://github.com/SamuelMarks/ml-switcheroo)) ecosystem, zero-grain solves this by providing:

1. **Perfect API Parity:** zero-grain exports the exact same classes, functions, default arguments, and docstrings as the real [grain framework](https://github.com/google/grain) (specifically targeting version 0.2.16.dev20260112).
2. **Semantic Tracing Hooks:** Under the hood, the dataloaders act as tracers that emit Intermediate Representation (IR) graphs of your data pipeline, seamlessly tying into the [ml-switcheroo](https://github.com/SamuelMarks/ml-switcheroo)-compiler infrastructure.
3. **Zero Weight:** It relies solely on the Python Standard Library and [numpy](https://numpy.org/) (which is supported in [WASM](https://webassembly.org/)). File interfaces like [ArrayRecordDataSource](https://github.com/google/grain) gracefully fallback to mock behaviors when the proprietary C++ extensions aren't locally compiled.
4. **[PyTree](https://jax.readthedocs.io/en/latest/pytrees.html) & Distributed Mocking:** Integrates directly with [zero-jax](https://github.com/SamuelMarks/zero-jax) to handle complex [PyTree](https://jax.readthedocs.io/en/latest/pytrees.html) transformations ([tree_map](https://jax.readthedocs.io/en/latest/_autosummary/jax.tree_util.tree_map.html), tree_collate) and simulates distributed sharding logic (process_index()) locally.

This enables you to write your data pipelines *once* using the standard grain API, but execute them literally anywhere—including the browser.

---

[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/SamuelMarks/zero-grain/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/zero-grain/actions)
[![Test Coverage](https://img.shields.io/badge/test_coverage-100%25-brightgreen.svg)](#)
[![Doc Coverage](https://img.shields.io/badge/doc_coverage-100%25-brightgreen.svg)](#)


---

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
