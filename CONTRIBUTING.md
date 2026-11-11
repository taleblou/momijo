# Contributing to Momijo

We welcome carefully scoped bug fixes, performance improvements, native Mojo tests and documentation changes for either engine.

1. Open an issue specifying the API or performance contract, relevant engine (`tensor` or `tensor_light`) and supported device.
2. Keep the engines independent. Do not introduce `tensor` ↔ `tensor_light` implementation imports, implicit conversions or `*_light` versions of the reserved application domains.
3. Add **only Mojo** programming-language source (`.mojo`); YAML, TOML and Markdown are allowed for CI, packaging and documentation. Do not add Python, Rust, shell-script files, JavaScript or other language implementations to Momijo.
4. Keep the MIT SPDX copyright/license header and accurate `# Module:`/`# Source:` lines on edited `.mojo` files.
5. Run `pixi run test-smoke` and the relevant `pixi run test-tensor`, `pixi run test-tensor-light` or `pixi run test-all` task, as well as `pixi run package` for packaging changes. If the toolchain is unavailable, state **NOT RUN** rather than implying a pass.
6. Include independently checked correctness, numerical edge cases, and if warranted a comparable benchmark with an explicit hardware/precision/timing methodology.
7. Document public API changes in [CHANGELOG.md](CHANGELOG.md), and significant architecture changes under `docs/architecture/` or an ADR.

See [development setup](docs/development/setup.md), [testing](docs/development/testing.md), and [code conventions](docs/development/contributing-code.md).
