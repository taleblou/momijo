# momijo.csv
High-performance CSV for Mojo (SIMD-ready, DFA, zero-copy, parallel).

- SIMD-ready pre-scan + DFA (RFC 4180 & relaxed)
- Chunk-parallel with quote parity
- Zero-copy RowView; lazy unescape/decoding
- Serde-like row→struct mapping
- Columnar builder with post-hoc inference (Int64/Float64/String)
- Writer with QuotePolicy
- Optional: Python interop, delimiter auto-detect

## Quick Start
```
from momijo.csv import CsvReader, CsvReaderOptions, Dialect, ByteSource
```

## Columnar mode
```
from momijo.csv.columnar import ColumnarBuilder, ColumnarTable
var builder = ColumnarBuilder(Dialect())
var table = ColumnarTable()
# ingest rows with builder.add_row(...)
builder.finalize(table)
```
