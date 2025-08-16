from momijo.csv.dialect import Dialect

struct Chunk:
    var start: Int
    var end: Int
    fn __init__(start: Int, end: Int): self.start = start; self.end = end

fn make_chunks(n_bytes: Int, chunk_size: Int = 16 * 1024 * 1024) -> List[Chunk]:
    var chunks = List[Chunk]()
    var i = 0
    while i < n_bytes:
        let e = min(i + chunk_size, n_bytes)
        chunks.push_back(Chunk(i, e))
        i = e
    return chunks

fn compute_chunk_parity(bytes: List[UInt8], chunks: List[Chunk], dialect: Dialect) -> List[Bool]:
    var parity_per_chunk = List[Bool](chunks.size)
    for idx in range(0, chunks.size):
        var c = chunks[idx]
        var count = 0
        var i = c.start
        while i < c.end:
            if bytes[i] == dialect.quote: count += 1
            i += 1
        parity_per_chunk[idx] = (count % 2) == 1
    var prefix = List[Bool](chunks.size)
    var p = False
    for idx in range(0, chunks.size):
        prefix[idx] = p
        if parity_per_chunk[idx]: p = not p
    return prefix
