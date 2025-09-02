# momijo.core

Minimal core utilities used across Momijo. This package intentionally avoids:
- exceptions (prefer `Result[T,E]` + `Error`)
- enums (some Mojo versions lack `enum`); use tagged-structs
- code in `__init__.mojo`

Files:
- `version.mojo`: simple version helpers
- `result.mojo`: `Result[T,E]`
- `option.mojo`: `Option[T]`
- `errors.mojo`: lightweight `Error`
- `asserts.mojo`: `require`/checks
- `log.mojo`: tiny logging
- `string_utils.mojo`: starts_with/ends_with

Usage example:

```
from momijo.core.result import Result
from momijo.core.errors import Error

fn double_if_even(x: Int) -> Result[Int, Error]:
    if x % 2 == 0:
        return Result[Int, Error].ok(x * 2)
    else:
        return Result[Int, Error].err(Error(1, "not even"))
```
