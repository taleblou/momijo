from momijo.csv.reader import ByteSource

fn bytes_to_source(py_bytes: List[UInt8]) -> ByteSource:
    return ByteSource(py_bytes)
