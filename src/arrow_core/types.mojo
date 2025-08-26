# MIT License
# Project: momijo.arrow_core
# File: momijo/arrow_core/types.mojo

# If you have helpers in momijo.enum, import as a namespace (not inherited).
import momijo.enum.enum_core as enum_core

# ---------------- Module meta ----------------
fn __module_name__() -> String:
    return String("momijo/arrow_core/types.mojo")

fn __self_test__() -> Bool:
    # Quick smoke tests
    var xs = List[Float64]()
    xs.append(1.0); xs.append(3.0); xs.append(2.0)
    if argmax_index(xs) != 1: return False
    if argmin_index(xs) != 0: return False
    if arrow_type_name(ArrowType_INT()) != String("Int"): return False
    if not arrow_type_is_numeric(ArrowType_FLOAT64()): return False
    return True

# ---------------- Small helpers ----------------
fn argmax_index(xs: List[Float64]) -> Int:
    if len(xs) == 0: return -1
    var best = xs[0]; var idx = 0; var i = 1
    while i < len(xs):
        if xs[i] > best:
            best = xs[i]; idx = i
        i += 1
    return idx

fn argmin_index(xs: List[Float64]) -> Int:
    if len(xs) == 0: return -1
    var best = xs[0]; var idx = 0; var i = 1
    while i < len(xs):
        if xs[i] < best:
            best = xs[i]; idx = i
        i += 1
    return idx

fn ensure_not_empty[T: ExplicitlyCopyable & Movable](xs: List[T]) -> Bool:
    return len(xs) > 0

# ---------------- ArrowType (enum-like struct) ----------------
struct ArrowType(ExplicitlyCopyable, Movable):
    var tag: Int32

    fn __init__(out self, tag: Int32):
        self.tag = tag

    fn __copyinit__(out self, other: Self):
        self.tag = other.tag

    # Simple “enum” API for interop
    fn to_int(self) -> Int32:
        return self.tag

    @staticmethod
    fn from_int(v: Int32) -> Self:
        return Self(v)

    # Named constructors (variants) — decorators must be on their own line
    @staticmethod
    fn INT() -> Self:
        return Self(1)

    @staticmethod
    fn FLOAT64() -> Self:
        return Self(2)

    @staticmethod
    fn STRING() -> Self:
        return Self(3)

    @staticmethod
    fn BOOL() -> Self:
        return Self(4)

    @staticmethod
    fn UNKNOWN() -> Self:
        return Self(0)

    # Optional helpers for listing names/tags
    @staticmethod
    fn variant_tags() -> List[Int32]:
        var xs = List[Int32]()
        xs.append(1); xs.append(2); xs.append(3); xs.append(4); xs.append(0)
        return xs

    @staticmethod
    fn variant_names() -> List[String]:
        var xs = List[String]()
        xs.append(String("Int"))
        xs.append(String("Float64"))
        xs.append(String("String"))
        xs.append(String("Bool"))
        xs.append(String("Unknown"))
        return xs

# Convenience top-level functions to get variants without calling static methods
# (useful if your call-sites prefer functions over ArrowType.*())
fn ArrowType_INT() -> ArrowType:      return ArrowType.INT()
fn ArrowType_FLOAT64() -> ArrowType:  return ArrowType.FLOAT64()
fn ArrowType_STRING() -> ArrowType:   return ArrowType.STRING()
fn ArrowType_BOOL() -> ArrowType:     return ArrowType.BOOL()
fn ArrowType_UNKNOWN() -> ArrowType:  return ArrowType.UNKNOWN()

# Human-readable name
fn arrow_type_name(t: ArrowType) -> String:
    var v = t.tag
    if v == 1: return String("Int")
    if v == 2: return String("Float64")
    if v == 3: return String("String")
    if v == 4: return String("Bool")
    return String("Unknown")

# Parse from string
fn parse_arrow_type(name: String) -> ArrowType:
    if name == String("Int"):      return ArrowType_INT()
    if name == String("Float64"):  return ArrowType_FLOAT64()
    if name == String("String"):   return ArrowType_STRING()
    if name == String("Bool"):     return ArrowType_BOOL()
    return ArrowType_UNKNOWN()

# Numeric predicate
fn arrow_type_is_numeric(t: ArrowType) -> Bool:
    var v = t.tag
    return (v == 1) or (v == 2)

# Default value (as String)
fn arrow_type_default(t: ArrowType) -> String:
    var v = t.tag
    if v == 1: return String("0")
    if v == 2: return String("0.0")
    if v == 3: return String("")
    if v == 4: return String("false")
    return String("")
