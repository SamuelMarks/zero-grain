# zero-grain

zero-grain is a 100% dependency-free (except for numpy), API-compatible implementation of Google's Grain data loader framework.

### Why does this exist?

In modern machine learning, deploying to edge devices, WASM runtimes, or strict containerized environments is heavily impeded by dependency bloat. The official grain framework is incredibly powerful, but it transitively pulls in massive ecosystem dependencies (like heavy C++ file extensions for array_record, absl-py, and more) that make it completely impossible to run in environments like pyodide (WebAssembly) or lightweight mobile environments.

As part of the Abstract ML Compiler (ml-switcheroo) ecosystem, zero-grain solves this by providing:

1. **Perfect API Parity:** zero-grain exports the exact same classes, functions, default arguments, and docstrings as the real grain framework (specifically targeting version 0.2.16.dev20260112).
2. **Semantic Tracing Hooks:** Under the hood, the dataloaders act as tracers that emit Intermediate Representation (IR) graphs of your data pipeline, seamlessly tying into the ml-switcheroo-compiler infrastructure.
3. **Zero Weight:** It relies solely on the Python Standard Library and numpy (which is supported in WASM). File interfaces like ArrayRecordDataSource gracefully fallback to mock behaviors when the proprietary C++ extensions aren't locally compiled.
4. **PyTree & Distributed Mocking:** Integrates directly with zero-jax to handle complex PyTree transformations (tree_map, tree_collate) and simulates distributed sharding logic (process_index()) locally.

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
