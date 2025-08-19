# momijo.enum

High-performance enum & pattern-matching toolkit for Mojo. Features include `Enum`/`IntEnum`/`StrEnum`, `Flag`, `EnumSet`/`EnumMap`, compact/tagged unions with **niche optimization**, and fast pattern matching via **sparse trees** and **jump tables**—bringing ergonomics from Python/Java along with low-level layout control akin to Rust/Swift.

---

## Highlights
- **Enum family**: `Enum`, `IntEnum`, `StrEnum` with ergonomic definition and display.
- **Bit flags**: `Flag` with safe bitwise ops.
- **Collections**: `EnumSet`, dynamic `EnumSetDyn`, and `EnumMap`.
- **Pattern matching**: Build-time sparse decision trees and jump tables (`Case`, `build_sparse_tree`, `match_sparse_tree`).
- **Unions with niche**: Payload-aware layout (`repr`, `payload`, `niche`) to eliminate tags when possible.
- **Serde**: String/JSON/CBOR adapters for easy I/O.
- **Diagnostics**: Exhaustiveness checks and friendly error messages.
- **Performance**: Benchmarks show O(1) jumps for dense tags and logarithmic sparse trees for wide enums.

## Install / Build
Add the repo `src/` to `-I` or your `MOJO_PATH`:
```bash
mojo -I /path/to/repo/src main.mojo
```

## Quickstart
```mojo
from enum import IntEnum, Flag, EnumSet, Case, build_sparse_tree, match_sparse_tree

@value
struct Color: IntEnum:
    var Red:   Int32 = 1
    var Green: Int32 = 2
    var Blue:  Int32 = 3

fn demo_match(tag: UInt64) -> UInt64:
    var tree = build_sparse_tree([
        Case(tag=1, arm=10),
        Case(tag=2, arm=20),
        Case(tag=3, arm=30)
    ])
    return match_sparse_tree(tree, tag, default_arm=0)
```

## API Surface (overview)
- **Core types**
  - `Enum`, `IntEnum`, `StrEnum`
  - `Flag`
  - `EnumSet[T]`, `EnumSetDyn`, `EnumMap[K, V]`
- **Matching**
  - `Case(tag: UInt64, arm: UInt64)`
  - `build_sparse_tree(cases: List[Case]) -> SparseTree`
  - `match_sparse_tree(tree: SparseTree, tag: UInt64, default_arm: UInt64) -> UInt64`
  - Dense tables for small consecutive tags (auto-selected)
- **Union/Representation**
  - `repr`, `payload`, `niche` helpers for compact layouts
- **Serde**
  - `serde_str`, `serde_json`, `serde_cbor`
- **Diagnostics & Utilities**
  - Exhaustive match checks, profiling hooks

> Exact module names may differ slightly depending on your snapshot; the public imports are re-exported by `momijo.enum.__init__`.

## Design Notes
- **No reserved-name conflicts**: internals use `matcher.mojo` instead of `match.mojo`, and public API is re-exported at package root.
- **Zero-cost abstractions**: Favor `@value` types and inlined helpers where possible.
- **Layout control**: Separate representation modules for clarity and future ABI stability.

## Performance
- Dense tag ranges: array-indexed jump → O(1).
- Sparse big enums: balanced decision tree (near log₂N depth).
- Benchmarks are included; integrate with your CI to track regressions.

## Compatibility
- Tested with Mojo 25.5+ toolchain snapshots.
- Pure Mojo—no external system deps.

## Examples
See `examples/enum/` (define enums, flags, pattern matching, serde).

## FAQ
**Q: Can I mix `EnumSet` with `Flag`?**  
Yes. `EnumSet` is for enum elements; `Flag` is bitfield-style. Choose based on your semantic needs.

**Q: How do I guarantee exhaustive matches?**  
Use diagnostics utilities; keep default arms minimal and prefer explicit tagging for evolving enums.

## Contributing
- Keep public symbols re-exported in `momijo/enum/__init__.mojo`.
- Add docs and tests with each feature.
- Follow SemVer; document breaking changes in the subpackage CHANGELOG.

## License
Add your license terms at repo root (e.g., `LICENSE`). You can also embed notices in source headers if required.
