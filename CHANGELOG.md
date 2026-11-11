# Changelog

All notable user-facing changes should be recorded in this file. Use an `Unreleased` section before creating an immutable release tag; follow Semantic Versioning for stable packages.

## [Unreleased]

### Logo filename standardization (2026-11-11)

- Renamed the shared SVG asset to `assets/momijo.svg` and updated the GitHub README and documentation index to reference the new path. The SVG artwork is byte-for-byte unchanged; no Mojo implementation or test files were modified.

### Sixth independent audit: fail-closed, fresh AOT output (2026-11-11)

- Reproduced a stale-artifact false-pass in the prior `pixi run package`: a successful mock compiler command that did not produce a new file left an existing `dist/momijo.mojoc` in place and returned success.
- Updated the packaging task to remove stale output, precompile to a private staging directory, require a nonempty newly generated payload, and move it to the final path only after validation. Failed/empty output now fails the task and leaves no stale final artifact.
- Clarified the AOT and testing documentation. All original Mojo source, tests, examples, benchmarks, ownership, and logo artwork are preserved; no real Mojo compilation is claimed.

### Repository branding (2026-11-11)

- Added the provided Momijo SVG artwork as a single shared branding asset, without modifying the supplied graphic.
- Displayed the shared logo in the repository README and documentation index using relative paths and accessible text descriptions. No Mojo source or test files were changed.

### Fifth independent audit: fail-closed entry-point inspection and source policy (2026-11-11)

- Fixed a reproducible false-pass: a `grep` read/inspection error while scanning a runnable Mojo test could previously be interpreted as a normal no-`main` file, allowing `test-all` to exit successfully after silently skipping a test. All three executing test tasks now distinguish normal no-match (status 1) from inspection failure (status 2 or other nonzero).
- Made reserved future-domain namespace checks fail when their `find` operation fails in CI, security or candidate-release workflows; additionally reject symlinks that had bypassed the regular-file allowlist.
- Preserved all original Mojo implementations, tests, examples, benchmarks and MIT attribution. Native compiler, GPU, Rattler and hosted GitHub verification are still required and were not reported as passed by local shell simulations.

### Fourth independent audit: end-to-end test gate and pipeline integrity (2026-11-11)

- Ensured all Pixi native test/listing tasks propagate discovery-pipeline errors through Bash `pipefail` rather than reporting partial success if `find` or `grep` fails. This preserves Mojo-only checked-in source files.
- Added the complete discovered Mojo regression suite as a mandatory prerequisite for GitHub pull-request CI and candidate-release artifacts, rather than requiring only two smoke tests.
- Added a minimal Rattler-build installed-payload check in the recipe, without claiming consumer import or numerical validation.
- Made documented Linux hash-generation examples fail on pipeline errors and synchronized setup, testing, and release-readiness explanations. No original `.mojo` files were changed.
- Static and mocked command validation can be executed in the audit environment; **real Mojo, Pixi, Rattler-build and hosted GitHub runs remain NOT RUN** until executed on a suitable system.

### Third independent review: verifiable candidate provenance (2026-11-11)

- Separated candidate `.mojoc` and checked-out Mojo source hashes into `ARTIFACT-SHA256SUMS` and `SOURCE-SHA256SUMS`. A downloaded GitHub artifact can now verify its included `.mojoc` independently; source verification is explicitly tied to the recorded Git commit.
- Updated architecture, reproducibility, and README instructions to match the revised artifact layout. Added pipeline failure propagation for source hashing; no Mojo implementation files or algorithms were changed.
- The new provenance commands were tested with simulated artifacts and tampering checks, **not** by running the unavailable Mojo compiler or GitHub-hosted jobs.

### Documentation completion and source-linked API navigation (2026-11-11)

- Added engine-specific API guidance and exhaustive package-root exported-name inventories, plus primary `Tensor` method-name indexes linked to their real source locations.
- Added declared-toolchain/platform compatibility, troubleshooting, and source-publication/release-readiness guidance.
- Improved documentation navigation and source-oriented examples; corrected a stale statement implying that the removed imported-name audit still runs in current CI.
- Preserved all 608 original Mojo source files and historical author/license notices; compilation and runtime verification remain unexecuted in this review.
- In a second repository QA pass, strengthened CI, scheduled security, and manual candidate-release source-policy checks to consistently reject foreign file formats and forbidden future-domain `*_light` variants; CI also checks source attribution and `# Module` / `# Source` metadata for every Mojo file. Clarified the difference between Mojo-only project source and external toolchain dependencies.

### Mojo-only GitHub source policy (2026-11-11)

- Removed four former Python tooling/test files and the explicit Python dependency. Every checked-in programming-language source file is now Mojo (`.mojo`). The previous Python scanner and its 27 tooling tests were **removed**, not converted into imaginary Mojo tests.
- Replaced Python-based Pixi smoke/test discovery with POSIX command orchestration of Mojo CLI entry points; updated GitHub CI, benchmark, security, and candidate-release workflows to avoid Python jobs. The full runtime tests still require a real Mojo compiler.
- Replaced the previous artifact-manifest utility with a Linux SHA-256 source/artifact inventory (originally `dist/SHA256SUMS`, subsequently split into the two separately verifiable manifests described above), commit/toolchain record files, and explicit nonempty-artifact verification. Updated active documentation and the source comment that named the removed utility.
### Historical audit notes (before the Mojo-only source cutover)

The entries below are historical records. Previously referenced Python tools and tests are **not** included in the current Mojo-only source tree.

- Ninth-pass independent audit: corrected relative Mojo executable invocation, launch errors, headers-only audit reporting and artifact provenance paths. Added four Python regression tests; no original Mojo changes.

- Eighth-pass audit: fixed a local imported-name checker bypass for function-scoped Mojo imports, while keeping nested names out of module-level exports. Hardened the AOT manifest against zero-byte output and destructive --output paths, and corrected the exact-smoke regression fixture. Four additional Python tests (19 total) passed; all 601 original Mojo files remain unchanged.

- Seventh-pass audit: hardened the CPU smoke gate to require the direct `smoke_cpu.mojo` entry in each engine; nested same-name files can no longer satisfy the baseline. Added three Python regression tests and a guard against unintended `*_light` application-domain forks. Original Mojo implementation files remain unchanged.

- Sixth-pass independent audit: verified all 5,947 local imported identifiers against source-declared names, added that assertion to the static checker and four regression tests (including escaped keywords, `comptime` exports and multiline imports); reconciled misleading `.mojopkg` wording against Mojo 1.1 release-note/API-documentation differences. No original Mojo source or algorithms were changed.


- Fifth-pass infrastructure audit: made the two-engine CPU smoke gate match only each engine's exact `smoke_cpu.mojo` filename (a similarly prefixed test can no longer bypass it), added a regression test, and added a full-package AOT precompile check to pull-request CI, and exactly pinned Mojo 1.1.0/MAX 26.6.0 to match the Rattler recipe. Engine source and algorithms were not changed.

- Fourth-pass audit: adopted Mojo 1.1 precompile and `.mojoc` in tooling, strengthened resolved relative-import and package-boundary checks, and required a runnable CPU smoke entry for **each** engine. Added three Python regression tests. All 601 original Mojo files remain untouched beyond the previously authorized header/import migration.
- Restricted the static Mojo scanner to repository-owned source trees so Pixi-installed dependency files are not misidentified as Momijo files. Added Python tooling regression tests and CodeQL scanning for Python tooling.
- Corrected the AOT manifest `--artifact` override semantics and added a smoke-test gate before publishing candidate CI build artifacts.

- Restored reserved `dataframe`, `learn`, `vision` and `visual` module boundaries and their docs/test/example/benchmark folders. These are scaffolds, not implemented libraries.
- Updated generated packaging integration to Mojo 1.1 `precompile` and compiler-matched `.mojoc`.

### Changed
- Consolidated the former `src/momijo/tensor/` and `src/momijo_light/tensor/` trees under `src/momijo/` as separately namespaced `tensor` and `tensor_light` packages.
- Reorganized example, test and benchmark directories by engine without discarding existing `.mojo` source content.
- Rewrote source and test import paths, synchronized SPDX/Module/Source headers, and added repository-level checks.

### Added
- Root package marker, Pixi manifest, conda recipe, documentation, contribution and security policies, GitHub workflows and standard library-building utilities.

### Verification status
- Source, local import-path and locally declared exported-name consistency checked statically (not typechecked).
- Compilation, binary package creation and workload benchmark results **not verified** during this migration.
- A resolved `pixi.lock` awaits execution of `pixi lock` on a network-enabled developer machine.

## [0.1.0] – pre-release baseline

Original full and FP32 Light Mojo tensor source snapshot, preserved as the basis for the above layout migration. This entry describes the baseline rather than asserting publication of a tagged release.

## Historical audit record — V10 provenance and release-asset audit (2026-10-24)

These lines record a prior tooling configuration, not capabilities included in the current ZIP.

- Prevent the AOT manifest from accepting a stale nonempty artifact when the Momijo source tree or a required engine/domain package marker has been deleted.
- Gracefully handle failures/timeouts when the optional `mojo --version` provenance probe cannot run, recording the compiler version as unknown rather than failing with an uncaught exception.
- Include the required MIT license in the manually reviewed GitHub upload artifact and label that output as a compiler-matched candidate, not a source archive.
- Add three regression tests for missing source package, missing engine boundary, and non-executable Mojo version-probe behavior. No original Mojo code changed.
