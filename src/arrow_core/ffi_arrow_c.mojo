# MIT License
# Project: momijo.arrow_core
# File: momijo/arrow_core/ffi_arrow_c.mojo

fn __module_name__() -> String:
    return String("momijo/arrow_core/ffi_arrow_c.mojo")

fn __self_test__() -> Bool:
    # Extend with real checks as needed
    return True

# --- Lightweight helpers (no external deps) ---
fn argmax_index(xs: List[Float64]) -> Int:
    if len(xs) == 0:
        return -1
    var best = xs[0]
    var idx = 0
    var i = 1
    while i < len(xs):
        if xs[i] > best:
            best = xs[i]
            idx = i
        i += 1
    return idx

fn argmin_index(xs: List[Float64]) -> Int:
    if len(xs) == 0:
        return -1
    var best = xs[0]
    var idx = 0
    var i = 1
    while i < len(xs):
        if xs[i] < best:
            best = xs[i]
            idx = i
        i += 1
    return idx

fn ensure_not_empty[T: Copyable & Movable](xs: List[T]) -> Bool:
    return len(xs) > 0

# -------------------------------------------------------------------
# Minimal Arrow C Data Interface mirror using plain integers for pointers.
# We intentionally avoid Pointer[...] and consts to keep this compilable.
# -------------------------------------------------------------------

struct ArrowArray(Copyable, Movable):
    var length: Int64
    var null_count: Int64
    var offset: Int64
    var n_buffers: Int64
    var n_children: Int64
    # Pseudo-pointer fields as raw addresses (UInt64)
    var buffers_addr: UInt64
    var children_addr: UInt64
    var dictionary_addr: UInt64
    var release_addr: UInt64
    var private_data_addr: UInt64

    fn __init__(out self):
        self.length = 0
        self.null_count = 0
        self.offset = 0
        self.n_buffers = 0
        self.n_children = 0
        self.buffers_addr = 0
        self.children_addr = 0
        self.dictionary_addr = 0
        self.release_addr = 0
        self.private_data_addr = 0

struct ArrowSchema(Copyable, Movable):
    # C strings as raw addresses
    var format_addr: UInt64
    var name_addr: UInt64
    var metadata_addr: UInt64
    var flags: Int64
    var n_children: Int64
    # Pseudo-pointer fields as raw addresses
    var children_addr: UInt64
    var dictionary_addr: UInt64
    var release_addr: UInt64
    var private_data_addr: UInt64

    fn __init__(out self):
        self.format_addr = 0
        self.name_addr = 0
        self.metadata_addr = 0
        self.flags = 0
        self.n_children = 0
        self.children_addr = 0
        self.dictionary_addr = 0
        self.release_addr = 0
        self.private_data_addr = 0

# ---------- Free functions ----------
fn arrow_array_length(arr: ArrowArray) -> Int64:
    return arr.length

fn arrow_array_null_count(arr: ArrowArray) -> Int64:
    return arr.null_count

fn arrow_schema_flag_nullable(schema: ArrowSchema) -> Bool:
    # ARROW_FLAG_NULLABLE = 2
    return (schema.flags & 2) != 0
