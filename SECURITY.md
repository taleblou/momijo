# Security Policy

## Supported versions

No stable release or long-term support policy is claimed by this source snapshot. Security fixes should target the active default development branch; the supported release list will be updated when a tagged release exists.

## Reporting a vulnerability

Avoid a public GitHub issue for unpatched vulnerabilities. Prefer a **private vulnerability report** using the repository's GitHub Security Advisories reporting facility, if enabled; otherwise contact the maintainers through an existing private repository channel. Include affected version or commit, reproducible steps, expected versus actual behavior, impact and whether a workaround exists. Do not share secrets or user data.

## Security-sensitive boundaries

Memory owners, alias analysis, unchecked indexing, device transfers, alignment, buffer lifetime, the AOT loader and externally supplied shape/stride metadata require extra scrutiny. See [safety design](docs/security/safety.md). The `security.yml` workflow checks repository tooling; it is **not** a whole-program security proof for Mojo kernels.
