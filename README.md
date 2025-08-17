# Momijo — Modern, Research-Backed Libraries (Mojo • Python • Rust • C/C++)

**Momijo** is a future-oriented effort to (re)implement essential libraries across languages using **modern software architecture** and **evidence-based design** (papers, benchmarks, best practices).
The initial focus is **Mojo**, while keeping APIs cohesive with **Python**, **Rust**, and **C/C++** where appropriate.

> **Project status:** actively evolving — growing and improving day by day.

---

## Why this project?

- **Research-backed decisions:** We follow peer-reviewed papers, established benchmarks, and canonical implementations.
- **Modern architecture:** Clean layering, small testable modules, and performance by design (data-oriented layouts, SIMD/parallelism when justified).
- **Cross-language consistency:** Similar semantics across languages to reduce cognitive load.
- **Community collaboration:** Contributions are welcome—help us shape a robust, future-minded library suite.

---

## Repository layout

```
src/
  momijo/
    <library>/
examples/
  <library>/
tests/
  <library>/
```

> We start from foundational “core” pieces, then expand outward. The set of libraries will grow over time.

---

## Evolving priorities

This repository evolves continuously. We begin with foundational building blocks that many others will use, then extend into broader functionality. Priorities adapt based on research findings, performance results, and community feedback.

---

## Architecture principles

- Separation of concerns (core vs. adapters vs. I/O)
- Stable, minimal, and testable APIs
- Performance by design (memory layout, SIMD/threads where it pays off)
- Robust error handling and documented invariants
- Portability and reproducibility (deterministic builds/tests where possible)

---

## Contributing

We’d love your help! This is a **future-minded** project, and contributions make us happy.

1. **Discuss** ideas and research references (open an issue).
2. **Design** small, composable APIs; include diagrams or rationale when helpful.
3. **Implement** according to the architecture principles.
4. **Test** with unit/property tests; add microbenchmarks for hot paths.
5. **Document** usage and design trade-offs.

**PR checklist**
- Tests pass (`tests/<library>`).
- Performance-critical paths include measurements.
- API changes documented (with migration notes if needed).
- Small, focused commits with clear messages.

---

## Getting started (Mojo example)

```bash
# structure
src/momijo/<library>/
examples/<library>/
tests/<library>/

# run tests (adapt to your runner)
mojo tests/<library>/run_tests.mojo
```

Add language/toolchain specifics for Python, Rust, or C/C++ as needed.

---

## License

**MIT License** — see `LICENSE` in this repository.

---

## Copyright

**© 2025 — Morteza Taleblou & Mitra Daneshmand**  
**Website:** taleblou.ir

---

## Contact & Community

- Use issues/discussions for bugs, features, and design proposals.
- Share papers/benchmarks that can guide architectural choices.
- Tell us where these libraries help—your feedback shapes the direction.

> Join us and help Momijo grow—this project is evolving and getting better every day.
