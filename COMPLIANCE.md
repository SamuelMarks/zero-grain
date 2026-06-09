# Compliance Verification

This repository uses the `ml-switcheroo` ecosystem's automated tooling to ensure API parity and Intermediate Representation (IR) structural correctness.

As we burn down the TDD requirements outlined in `@ABSTRACT_ML_PLAN.md`, you can use these two checkers locally to verify progress.

## 1. API Snapshot Compliance

The `ml-framework-snapshots` tool verifies that `zero-grain` accurately replicates the public API signatures, parameters, and defaults of the real `grain` framework.

**To run the checker against the current codebase:**

```bash
ml_framework_snapshots check grain ~/repos/zero-grain/src \
    --reference-prefix grain \
    --target-prefix zero_grain
```

**What it does:**
- It statically analyzes all Python modules in `zero-grain/src`.
- It maps the target namespace (`zero_grain.*`) back to the reference framework (`grain.*`).
- It outputs a detailed markdown checklist of missing APIs and signature mismatches.
- A score of `100.0%` means perfect structural API parity.

## 2. IR & Frontend Compliance

The `ml-switcheroo-ir` tool verifies that the implementation correctly maps frontend primitives to the canonical ONNX-based Logical Graph Dialect.

**To run the checker:**

```bash
ml-switcheroo-ir compliance ~/repos/zero-grain/src
```

**What it does:**
- It scans the code for classes implementing the `GraphFrontend` protocol (from `ml_switcheroo_ir`).
- It verifies that the operations emitted into `LogicalNode` constructs strictly adhere to the registered ONNX dialect schemas (attributes, input/output counts, and dtypes).
- If your trace adapter attempts to emit an unregistered operation, it will fail compliance until a custom ops schema is registered.
