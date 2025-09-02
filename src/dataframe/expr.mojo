# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo

# =============================================================
# FILE 1/2 — src/momijo/dataframe/expr.mojo (library module)
# =============================================================

from momijo.dataframe.frame import DataFrame
from momijo.dataframe.bitmap import Bitmap
from momijo.dataframe.column import Column

# ------------------------------
# Op and Expr kinds (no global vars; static accessors)
# ------------------------------
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

# ------------------------------
# ConstVal — tiny tagged union for scalar literals
# ------------------------------
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

# ------------------------------
# Pred — single-column comparison
# ------------------------------
struct Pred:
    var col: String
    var op: Int
    var value: ConstVal

    fn __init__(out self):
        self.col = String("")
        self.op = Ops.EQ()
        self.value = ConstVal()

    fn __init__(out self, col: String, op: Int, value: ConstVal):
        self.col = col
        self.op = op
        self.value = value

    fn __copyinit__(out self, other: Self):
        self.col = String(other.col)
        self.op = other.op
        self.value = other.value

    fn __moveinit__(out self, owned other: Self):
        self.col = other.col
        self.op = other.op
        self.value = other.value

# ------------------------------
# Expr — either a single Pred or A AND/OR B (A,B as Pred; left-assoc builder in parser)
# ------------------------------
struct Expr:
    var kind: Int
    var p1: Pred
    var p2: Pred

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

    fn __moveinit__(out self, owned other: Self):
        self.kind = other.kind
        self.p1 = other.p1
        self.p2 = other.p2

    @staticmethod
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

# ------------------------------
# Evaluators
# ------------------------------
fn eval_pred(df: DataFrame, p: Pred) -> Bitmap:
    var s = df.get_column(p.col)
    var n = df.height()
    var mask = Bitmap(n, False)
    var i = 0
    while i < n:
        if s.is_valid(i):
            var ok = False
            # Numeric compare if predicate value is numeric
            if p.value.tag == 1 or p.value.tag == 2:
                var lv = s.as_f64_or_nan(i)
                var rv = 0.0
                if p.value.tag == 1:
                    rv = p.value.f64
                else:
                    rv = Float64(p.value.i64)
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
            elif p.value.tag == 3:
                var lvb = s.b.get(i)
                if p.op == Ops.EQ():
                    ok = (lvb == p.value.b)
                elif p.op == Ops.NE():
                    ok = (lvb != p.value.b)
            # String compare otherwise
            else:
                var lvs = s.value_str(i)
                if p.op == Ops.EQ():
                    ok = (lvs == p.value.s)
                elif p.op == Ops.NE():
                    ok = (lvs != p.value.s)
            if ok:
                _ = mask.set(i, True)
        i += 1
    return mask

fn mask_and(a: Bitmap, b: Bitmap) -> Bitmap:
    var n = len(a)
    var out = Bitmap(n, False)
    var i = 0
    while i < n:
        _ = out.set(i, a.get(i) and b.get(i))
        i += 1
    return out

fn mask_or(a: Bitmap, b: Bitmap) -> Bitmap:
    var n = len(a)
    var out = Bitmap(n, False)
    var i = 0
    while i < n:
        _ = out.set(i, a.get(i) or b.get(i))
        i += 1
    return out

fn eval_expr(df: DataFrame, e: Expr) -> Bitmap:
    if e.kind == ExprKind.SINGLE():
        return eval_pred(df, e.p1)
    elif e.kind == ExprKind.AND():
        return mask_and(eval_pred(df, e.p1), eval_pred(df, e.p2))
    else:
        return mask_or(eval_pred(df, e.p1), eval_pred(df, e.p2))

# ------------------------------
# Parser utilities (kept here to avoid a second file)
# ------------------------------
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

# ------------------------------
# Convenience: evaluate a textual expression directly
# ------------------------------
fn eval_where(df: DataFrame, expr_text: String) -> Bitmap:
    var e = parse_simple_expr(expr_text)
    return eval_expr(df, e)

