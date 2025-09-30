# Project:      Momijo
# Module:       dataframe.utils
# File:         utils.mojo
# Path:         dataframe/utils.mojo
#
# Description:  dataframe.utils — Utility helpers and shared routines.
#               Implements core data structures, algorithms, and convenience APIs for production use.
#               Designed as a stable, composable building block within the Momijo public API.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# License:      MIT License
# SPDX-License-Identifier: MIT
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand
#
# Notes:
#   - Structs: Stats, Ops, ExprKind, ConstVal, Pred
#   - Key functions: __init__, __copyinit__, is_digit, digit_value, is_sign, infer_dtype, sqrt64, parse_f64, compute_stats, pairs_append, percentile_f64, expanding_corr_last, between_i64, isin_string, df_where_qty_between, EQ, NE, GT
#   - Static methods present.
 
 
from momijo.dataframe.sorting import argsort_f64
from momijo.dataframe.series_bool import append
from momijo.dataframe.stats_core import corr_f64

from momijo.tensor.indexing import slice
from momijo.arrow_core.poly_column import get_string
from momijo.dataframe.api import col_str, df_make
from momijo.dataframe.frame import DataFrame, width
from momijo.dataframe.helpers import contains_string, find_col, parse_i64_or_zero

from momijo.dataframe.bitmap import Bitmap
from momijo.dataframe.frame import DataFrame
from momijo.dataframe.column import Column, ColumnTag, F64, I64
from momijo.dataframe.helpers import contains_string, find_col

struct Stats:
    var count: Int
    var mean: Float64
    var std: Float64
    var min: Float64
    var max: Float64

    fn __init__(out self):
        self.count = 0
        self.mean = 0.0
        self.std = 0.0
        self.min = 0.0
        self.max = 0.0

    fn __copyinit__(out self, other: Stats):
        self.count = other.count
        self.mean = other.mean
        self.std = other.std
        self.min = other.min
        self.max = other.max

fn is_digit(ch: String) -> Bool:
    return ch == "0" or ch == "1" or ch == "2" or ch == "3" or ch == "4" or                ch == "5" or ch == "6" or ch == "7" or ch == "8" or ch == "9"

fn digit_value(ch: String) -> Int:
    if ch == "0":
        return 0
    elif ch == "1":
        return 1
    elif ch == "2":
        return 2
    elif ch == "3":
        return 3
    elif ch == "4":
        return 4
    elif ch == "5":
        return 5
    elif ch == "6":
        return 6
    elif ch == "7":
        return 7
    elif ch == "8":
        return 8
    elif ch == "9":
        return 9
    else:
        return -1

fn is_sign(ch: String) -> Bool:
    return ch == "-" or ch == "+"

fn infer_dtype(col: List[String]) -> String:
    var has_float = False
    var has_int = True
    var i = 0
    while i < len(col):
        var s = col[i]
        if len(s) == 0:
            return "string"
        var ok_int = True
        var j = 0
        if len(s) > 0 and is_sign(String(s[0])):
            j = 1
        if j == len(s):
            ok_int = False
        while j < len(s):
            if not is_digit(String(s[j])):
                ok_int = False
                break
            j += 1
        if not ok_int:
            var ok_float = True
            var dot_seen = False
            j = 0
            if len(s) > 0 and is_sign(String(s[0])):
                j = 1
            if j == len(s):
                ok_float = False
            var k = j
            while k < len(s):
                var c = String(s[k])
                if is_digit(c):
                    # keep going
                    k += 1
                    continue
                elif c == "." and not dot_seen:
                    dot_seen = True
                    k += 1
                    continue
                else:
                    ok_float = False
                    break
            if ok_float:
                has_float = True
                has_int = False
            else:
                return "string"
        i += 1
    if has_float:
        return "float"
    if has_int:
        return "int"
    return "string"

fn sqrt64(x: Float64) -> Float64:
    if x <= 0.0:
        return 0.0
    var g = x
    var i = 0
    while i < 20:
        g = 0.5 * (g + x / g)
        i += 1
    return g

fn parse_f64(s: String, default: Float64 = 0.0) -> Float64:
    var sign: Float64 = 1.0
    var idx = 0
    if len(s) == 0:
        return default
    if s[0] == "-":
        sign = -1.0
        idx = 1
    elif s[0] == "+":
        idx = 1
    var val: Float64 = 0.0
    var frac: Float64 = 0.0
    var place: Float64 = 1.0
    var dot = False
    while idx < len(s):
        var c = String(s[idx])
        if c == ".":
            if dot:
                return default
            dot = True
            idx += 1
            continue
        var dv = digit_value(c)
        if dv < 0:
            return default
        var d = Float64(dv)
        if not dot:
            val = val * 10.0 + d
        else:
            place = place * 0.1
            frac = frac + d * place
        idx += 1
    return sign * (val + frac)

fn compute_stats(col: List[String]) -> Stats:
    var st = Stats()
    var n = len(col)
    if n == 0:
        return st
    var i = 0
    var sumv: Float64 = 0.0
    var minv: Float64 = 0.0
    var maxv: Float64 = 0.0
    var first = True
    while i < n:
        var v = parse_f64(col[i], 0.0)
        if first:
            minv = v
            maxv = v
            first = False
        else:
            if v < minv:
                minv = v
            if v > maxv:
                maxv = v
        sumv += v
        i += 1
    var mean = sumv / Float64(n)
    var varsum: Float64 = 0.0
    i = 0
    while i < n:
        var v2 = parse_f64(col[i], 0.0)
        var d = v2 - mean
        varsum += d * d
        i += 1
    var stdv = 0.0
    if n > 1:
        stdv = sqrt64(varsum / Float64(n - 1))
    st.count = n
    st.mean = mean
    st.std = stdv
    st.min = minv
    st.max = maxv
    return st

# --- Added for API completeness: pairs_append ---

fn pairs_append(mut pairs: List[Tuple[String, List[String]]], name: String, values: List[String]):
    pairs.append((name, values))

fn infer_dtype(col: Column) -> String:
    var tmp = List[String]()
    var i = 0
    while i < col.len():
        tmp.append(col[i])
        i += 1
    return infer_dtype(tmp)


fn compute_stats(col: Column) -> Stats:
    var tmp = List[String]()
    var i = 0
    while i < col.len():
        tmp.append(col[i])
        i += 1
    return compute_stats(tmp)

fn percentile_f64(xs: List[Float64], p: Float64) -> Float64
    if len(xs) == 0:
        return 0.0
    var idx = Int(Float64(len(xs) - 1) * p)
    return xs[argsort_f64(xs, True)[idx]]

fn expanding_corr_last(x: List[Float64], y: List[Float64]) -> List[Float64]
    var out = List[Float64]()
    var i = 1
    while i <= len(x):
        out.append(corr_f64(x.slice(0, i), y.slice(0, i)))
        i += 1
    return out


fn between_i64(x: Int64, a: Int64, b: Int64) -> Bool
    return x >= a and x <= b
fn isin_string(s: String, universe: List[String]) -> Bool
    return contains_string(universe, s)

# Filter rows where column 'qty' is between [a,b], inclusive. String-based materialization.
fn df_where_qty_between(df: DataFrame, a: Int64, b: Int64) -> DataFrame
    var i_qty = find_col(df, String("qty"))
    if i_qty < 0:
        return df_make(List[String](), List[Column]())
    var keep_idx = List[Int]()
    var r = 0
    while r < df.nrows():
        var q = parse_i64_or_zero(df.cols[i_qty][r])
        if between_i64(q, a, b):
            keep_idx.append(r)
        r += 1
# materialize filtered copy
    var col_names = df.col_names
    var cols = List[Column]()
    var c = 0
    while c < df.ncols():
        var acc = List[String]()
        var rr = 0
        while rr < len(keep_idx):
            acc.append(df.cols[c][keep_idx[rr]])
            rr += 1
        cols.append(col_str(df.col_names[c], acc))
        c += 1
    return df_make(col_names, cols)


struct Ops:
    @staticmethod
    fn EQ() -> Int:
            return 0

    @staticmethod
    fn NE() -> Int:
            return 1

    @staticmethod
    fn GT() -> Int:
            return 2

    @staticmethod
    fn GE() -> Int:
            return 3

    @staticmethod
    fn LT() -> Int:
            return 4

    @staticmethod
    fn LE() -> Int:
            return 5
    fn __moveinit__(out self, deinit other: Self):
            pass
struct ExprKind:
    @staticmethod
    fn SINGLE() -> Int:
        return 0

    @staticmethod
    fn AND() -> Int:
        return 1

    @staticmethod
    fn OR() -> Int:
        return 2
    fn __moveinit__(out self, deinit other: Self):
        pass

struct ConstVal:
    var tag: Int   # 1:f64 2:i64 3:bool 4:str
    var f64: Float64
    var i64: Int64
    var b: Bool
    var s: String
    fn __init__(out self):
            self.tag = 4
            self.f64 = 0.0
            self.i64 = 0
            self.b = False
            self.s = String("")
    fn __copyinit__(out self, other: Self):
            self.tag = other.tag
            self.f64 = other.f64
            self.i64 = other.i64
            self.b = other.b
            self.s = String(other.s)

    @staticmethod
    fn of_f64(x: Float64) -> ConstVal:
            var c = ConstVal()
            c.tag = 1
            c.f64 = x
            return c

    @staticmethod
    fn of_i64(x: Int64) -> ConstVal:
            var c = ConstVal()
            c.tag = 2
            c.i64 = x
            return c

    @staticmethod
    fn of_bool(x: Bool) -> ConstVal:
            var c = ConstVal()
            c.tag = 3
            c.b = x
            return c

    @staticmethod
    fn of_str(x: String) -> ConstVal:
            var c = ConstVal()
            c.tag = 4
            c.s = x
            return c
    fn __moveinit__(out self, deinit other: Self):
            self.tag = other.tag
            self.f64 = other.f64
            self.i64 = other.i64
            self.b = other.b
            self.s = other.s

struct Pred:
    var col: String
    var op: Int
    var value: ConstVal
    fn __init__(out self):
            self.col = String("")
            self.op = Ops.EQ()
            assert(self is not None, String("self is None"))
            self.value() = ConstVal()
    fn __init__(out self, col: String, op: Int, value: ConstVal):
            self.col = col
            self.op = op
            assert(self is not None, String("self is None"))
            self.value() = value
    fn __copyinit__(out self, other: Self):
            self.col = String(other.col)
            self.op = other.op
            assert(self is not None, String("self is None"))
            self.value() = other.value()
    fn __init__(out self):
            self.kind = ExprKind.SINGLE()
            self.p1 = Pred()
            self.p2 = Pred()
    fn __init__(out self, kind: Int, p1: Pred, p2: Pred):
            self.kind = kind
            self.p1 = p1
            self.p2 = p2
    fn __copyinit__(out self, other: Self):
            self.kind = other.kind
            self.p1 = other.p1
            self.p2 = other.p2
    fn single(p: Pred) -> Expr:
            return Expr(ExprKind.SINGLE(), p, Pred())

    @staticmethod
    fn both_and(a: Pred, b: Pred) -> Expr:
            return Expr(ExprKind.AND(), a, b)

    @staticmethod
    fn both_or(a: Pred, b: Pred) -> Expr:
            return Expr(ExprKind.OR(), a, b)
# Back-compat helpers
    fn expr_single(p: Pred) -> Expr: return Expr.single(p)
    fn expr_and(a: Pred, b: Pred) -> Expr: return Expr.both_and(a, b)
    fn expr_or(a: Pred, b: Pred) -> Expr: return Expr.both_or(a, b)

# Evaluators
fn eval_pred(df: DataFrame, p: Pred) -> Bitmap:
    var s = df.get_column(p.col)
    var n = df.nrows()
    var mask = Bitmap(n, False)
    var i = 0
    while i < n:
        if s.is_valid(i):
            var ok = False
# Numeric compare if predicate value is numeric
            assert(p is not None, String("p is None"))
            if p.value().tag == 1 or p.value().tag == 2:
                var lv = s.as_f64_or_nan(i)
                var rv = 0.0
                assert(p is not None, String("p is None"))
                if p.value().tag == 1:
                    rv = p.value().f64
                else:
                    assert(p is not None, String("p is None"))
                    rv = Float64(p.value().i64)
                if p.op == Ops.EQ():
                    ok = (lv == rv)
                elif p.op == Ops.NE():
                    ok = (lv != rv)
                elif p.op == Ops.GT():
                    ok = (lv > rv)
                elif p.op == Ops.GE():
                    ok = (lv >= rv)
                elif p.op == Ops.LT():
                    ok = (lv < rv)
                elif p.op == Ops.LE():
                    ok = (lv <= rv)
# Bool compare if predicate value is bool
            assert(p is not None, String("p is None"))
            elif p.value().tag == 3:
                var lvb = s.b.get(i)
                if p.op == Ops.EQ():
                    assert(p is not None, String("p is None"))
                    ok = (lvb == p.value().b)
                elif p.op == Ops.NE():
                    ok = (lvb != p.value().b)
# String compare otherwise
            else:
                assert(s is not None, String("s is None"))
                var lvs = s.value()_str(i)
                if p.op == Ops.EQ():
                    assert(p is not None, String("p is None"))
                    ok = (lvs == p.value().s)
                elif p.op == Ops.NE():
                    ok = (lvs != p.value().s)
            if ok:
                _ = mask.set(i, True)
        i += 1
    return mask
fn eval_expr(df: DataFrame, e: Expr) -> Bitmap:
    if e.kind == ExprKind.SINGLE():
        return eval_pred(df, e.p1)
    elif e.kind == ExprKind.AND():
        return mask_and(eval_pred(df, e.p1), eval_pred(df, e.p2))
    else:
        return mask_or(eval_pred(df, e.p1), eval_pred(df, e.p2))

# Parser utilities (kept here to avoid a second file)
fn is_digit_like(ch: String) -> Bool:
# crude: accepts [0-9] or '.' or '-'
    if ch == String("0") or ch == String("1") or ch == String("2") or ch == String("3") or ch == String("4"):
        return True
    if ch == String("5") or ch == String("6") or ch == String("7") or ch == String("8") or ch == String("9"):
        return True
    if ch == String(".") or ch == String("-"):
        return True
    return False
fn trim(s: String) -> String:
    var i = 0
    var j = len(s) - 1
    while i < len(s) and s[i] == ' ':
        i += 1
    while j >= 0 and s[j] == ' ':
        j -= 1
    if j < i:
        return String("")
    var out = String("")
    var k = i
    while k <= j:
        out = out + String(s[k])
        k += 1
    return out
fn parse_op(tok: String) -> Int:
    if tok == String("=="):
        return Ops.EQ()
    elif tok == String("!="):
        return Ops.NE()
    elif tok == String(">="):
        return Ops.GE()
    elif tok == String("<="):
        return Ops.LE()
    elif tok == String(">"):
        return Ops.GT()
    elif tok == String("<"):
        return Ops.LT()
    return Ops.EQ()

# --- tiny numeric parsing to avoid std parsing deps ---
fn digit_val(d: String) -> Int64:
    if d == String("0"): return 0
    elif d == String("1"): return 1
    elif d == String("2"): return 2
    elif d == String("3"): return 3
    elif d == String("4"): return 4
    elif d == String("5"): return 5
    elif d == String("6"): return 6
    elif d == String("7"): return 7
    elif d == String("8"): return 8
    elif d == String("9"): return 9
    return 0
fn parse_i64_str(t: String) -> Int64:
    var sgn: Int64 = 1
    var i: Int = 0
    if len(t) > 0 and t[0] == '-':
        sgn = -1
        i = 1
    var acc: Int64 = 0
    while i < len(t):
        var dv = digit_val(String(t[i]))
        acc = acc * 10 + dv
        i += 1
    return sgn * acc
fn parse_f64_str(t: String) -> Float64:
    var sgn: Float64 = 1.0
    var i: Int = 0
    if len(t) > 0 and t[0] == '-':
        sgn = -1.0
        i = 1
    var int_part: Int64 = 0
    var frac_part: Float64 = 0.0
    var frac_scale: Float64 = 1.0
    var in_frac = False
    while i < len(t):
        var ch = t[i]
        if ch == '.':
            in_frac = True
            i += 1
            continue
        var dv = Float64(digit_val(String(ch)))
        if not in_frac:
            int_part = int_part * 10 + Int64(dv)
        else:
            frac_scale = frac_scale * 0.1
            frac_part = frac_part + dv * frac_scale
        i += 1
    return sgn * (Float64(int_part) + frac_part)
fn parse_const(tok: String) -> ConstVal:
    var t = trim(tok)
    if t == String("true"):
        return ConstVal.of_bool(True)
    if t == String("false"):
        return ConstVal.of_bool(False)

    var n = len(t)
    if n >= 2 and ((t[0] == '"' and t[n-1] == '"') or (t[0] == '\'' and t[n-1] == '\'')):
        var inner = String("")
        var i = 1
        while i < n - 1:
            inner = inner + String(t[i])
            i += 1
        return ConstVal.of_str(inner)

# number? crude scan
    var numeric = True
    var i = 0
    var dot_count = 0
    while i < len(t):
        var chs = String(t[i])
        if chs == String("."):
            dot_count += 1
        if not is_digit_like(chs):
            numeric = False
            break
        i += 1
    if numeric:
        if dot_count > 0:
            return ConstVal.of_f64(parse_f64_str(t))
        else:
            return ConstVal.of_i64(parse_i64_str(t))

    return ConstVal.of_str(t)
fn split_tokens(expr: String) -> List[String]:
# Split by spaces but keep quoted strings intact and recognize two-char ops
    var out = List[String]()
    var cur = String("")
    var i = 0
    var in_s = False
    var in_double = False

    while i < len(expr):
        var ch = expr[i]
        if in_s:
            cur = cur + String(ch)
            if (in_double and ch == '"') or ((not in_double) and ch == '\''):
                in_s = False
                out.append(cur)
                cur = String("")
            i += 1
            continue
        if ch == '\'' or ch == '"':
            if len(cur) > 0:
                out.append(cur)
                cur = String("")
            in_s = True
            in_double = (ch == '"')
            cur = cur + String(ch)
            i += 1
            continue
        if ch == ' ':
            if len(cur) > 0:
                out.append(cur)
                cur = String("")
            i += 1
            continue
# two-char ops
        if i + 1 < len(expr):
            var duo = String("")
            duo = duo + String(ch) + String(expr[i+1])
            if duo == String("==") or duo == String("!=") or duo == String(">=") or duo == String("<="):
                if len(cur) > 0:
                    out.append(cur)
                    cur = String("")
                out.append(duo)
                i += 2
                continue
# single-char ops
        if ch == '>' or ch == '<':
            if len(cur) > 0:
                out.append(cur)
                cur = String("")
            out.append(String(ch))
            i += 1
            continue
        cur = cur + String(ch)
        i += 1
    if len(cur) > 0:
        out.append(cur)
    return out
fn parse_simple_expr(expr: String) -> Expr:
# Pattern: <col> <op> <lit> [and|or <col> <op> <lit>] ... (left-assoc)
    var toks = split_tokens(expr)
    var n = len(toks)
    if n < 3:
        return Expr.single(Pred())

    var col = toks[0]
    var op = parse_op(toks[1])
    var val = parse_const(toks[2])
    var cur = Expr.single(Pred(col, op, val))

    var i = 3
    while i + 2 < n:
        var gate = toks[i]
        if i + 3 >= n:
            break
        var col2 = toks[i+1]
        var op2 = parse_op(toks[i+2])
        var val2 = parse_const(toks[i+3])
        var p2 = Pred(col2, op2, val2)
        if gate == String("and") or gate == String("AND"):
            cur = Expr.both_and(cur.p1, p2)
        elif gate == String("or") or gate == String("OR"):
            cur = Expr.both_or(cur.p1, p2)
        i += 4
    return cur

# Convenience: evaluate a textual expression directly
fn eval_where(df: DataFrame, expr_text: String) -> Bitmap:
    var e = parse_simple_expr(expr_text)
    return eval_expr(df, e)
 
fn _pow10(n: Int) -> Float64:
    var r = 1.0
    var i = 0
    while i < n:
        r = r * 10.0
        i += 1
    return r

@always_inline
fn _is_digit(ss) -> Bool:
# Accept any slice-like single-character; compare by equality to avoid range ops.
    return (ss == "0") or (ss == "1") or (ss == "2") or (ss == "3") or (ss == "4") or            (ss == "5") or (ss == "6") or (ss == "7") or (ss == "8") or (ss == "9")

fn _to_f64(s: String) -> Float64:
    if len(s) == 0:
        return 0.0
    var neg = False
    var i = 0
    if s[0] == "-":
        neg = True
        i = 1
    var int_part: Int = 0
    var frac_part: Int = 0
    var frac_len: Int = 0
    var seen_dot = False
    while i < len(s):
        var ch = s[i]
        if ch == ".":
            if seen_dot:   # second dot -> stop parse
                break
            seen_dot = True
        elif _is_digit(ch):
# ch like "7": convert by table
            var d: Int = 0
            if ch == "1": d = 1
            elif ch == "2": d = 2
            elif ch == "3": d = 3
            elif ch == "4": d = 4
            elif ch == "5": d = 5
            elif ch == "6": d = 6
            elif ch == "7": d = 7
            elif ch == "8": d = 8
            elif ch == "9": d = 9
            else: d = 0
            if not seen_dot:
                int_part = int_part * 10 + d
            else:
                frac_part = frac_part * 10 + d
                frac_len += 1
        else:
            break
        i += 1
    var val = Float64(int_part) + Float64(frac_part) / _pow10(frac_len)
    if neg: val = -val
    return val

fn ge(col_vals: List[String], threshold: Float64) -> List[Bool]:
    var out = List[Bool]()
    for v in col_vals:
        out.append(_to_f64(v) >= threshold)
    return out

fn isin(col_vals: List[String], values: List[String]) -> List[Bool]:
    var out = List[Bool]()
    for v in col_vals:
        var ok = False
        for x in values:
            if v == x:
                ok = True
                break
        out.append(ok)
    return out

fn logical_and(a: List[Bool], b: List[Bool]) -> List[Bool]:
    var out = List[Bool]()
    var i = 0
    var n = len(a)
    while i < n:
        out.append(a[i] and b[i])
        i += 1
    return out

fn between(col_vals: List[String], low: Float64, high: Float64, inclusive: Bool = True) -> List[Bool]:
    var out = List[Bool]()
    for v in col_vals:
        var x = _to_f64(v)
        if inclusive:
            out.append(x >= low and x <= high)
        else:
            out.append(x > low and x < high)
    return out
fn nlargest_f64(xs: List[Float64], n: Int) -> List[Int]
    var idx = argsort_f64(xs, False)
    var k = n
    if k > len(idx):
        k = len(idx)
    var out = List[Int]()
    var i = 0
    while i < k:
        out.append(idx[i])
        i += 1
    return out
fn nsmallest_f64(xs: List[Float64], n: Int) -> List[Int]
    var idx = argsort_f64(xs, True)
    var k = n
    if k > len(idx):
        k = len(idx)
    var out = List[Int]()
    var i = 0
    while i < k:
        out.append(idx[i])
        i += 1
    return out
fn clip_f64(xs: List[Float64], lo: Float64, hi: Float64) -> List[Float64]
    var out = List[Float64]()
    var i = 0
    while i < len(xs):
        var v = xs[i]
        if v < lo:
            v = lo
        if v > hi:
            v = hi
        out.append(v)
        i += 1
    return out

# --- one pass of top-level rewrite ---
fn optimize_once(p: LogicalPlan) -> LogicalPlan:
# Rule 1: Project over Filter  -> swap
    if p.kind == PLAN_PROJECT and p.child.kind == PLAN_FILTER:
# p = Project(Filter(X, e), cols)  ==>  Filter(Project(X, cols), e)
        var filter_child = p.child.child
        var filter_expr = p.child.expr
        var new_child = LogicalPlan.project(filter_child, p.columns)
        return LogicalPlan.filter(new_child, filter_expr)

# Rule 2: Filter over Project -> swap
    if p.kind == PLAN_FILTER and p.child.kind == PLAN_PROJECT:
# p = Filter(Project(X, cols), e)  ==>  Project(Filter(X, e), cols)
        var proj_child = p.child.child
        var proj_cols = p.child.columns
        var new_child = LogicalPlan.filter(proj_child, p.expr)
        return LogicalPlan.project(new_child, proj_cols)

# Rule 3: Collapse double Project
    if p.kind == PLAN_PROJECT and p.child.kind == PLAN_PROJECT:
# Project(Project(X, cols1), cols2) -> Project(X, cols2)
        var inner_child = p.child.child
        return LogicalPlan.project(inner_child, p.columns)

# No change at top-level in this pass
    return p

# --- run a few passes to propagate rewrites ---
fn optimize(plan: LogicalPlan) -> LogicalPlan:
    var cur = plan
    var i = 0
# Run a small fixed number of passes. This helps push rules deeper
# without requiring full recursion/rebuild for all node kinds.
    while i < 6:
        var nxt = optimize_once(cur)
# If no change at top, further passes may still catch a new pattern
# exposed by the previous rewrite one level down, so we just continue.
        cur = nxt
        i += 1
    return cur



fn is_numeric_col(c: Column) -> Bool
    return c.tag() == ColumnTag.I64() or c.tag() == ColumnTag.F64()
fn select_numeric(df: DataFrame) -> DataFrame
    var col_names = List[String]()
    var cols = List[Column]()
    var i = 0
    while i < df.ncols():
        if is_numeric_col(df.cols[i]):
            col_names.append(df.col_names[i])
            cols.append(df.cols[i])
        i += 1
    return df_make(col_names, cols)
fn drop_duplicates_rows(df: DataFrame, subset: String) -> DataFrame
    var idx = find_col(df, subset)
    if idx < 0:
        return df_make(List[String](), List[Column]())
    var seen = List[String]()
    var keep = List[Int]()
    var r = 0
    while r < df.nrows():
        var v = df.cols[idx][r]
        if not contains_string(seen, v):
            seen.append(v)
            keep.append(r)
        r += 1
    var col_names = df.col_names
    var cols = List[Column]()
    var c = 0
    while c < df.ncols():
        var vals = List[String]()
        var i = 0
        while i < len(keep):
            vals.append(df.cols[c][keep[i]])
            i += 1
        cols.append(col_str(col_names[c], vals))
        c += 1
    return df_make(col_names, cols)