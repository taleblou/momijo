# Project:      Momijo
# Module:       tensor.core
# File:         tensor_core.mojo
# Path:         src/momijo/tensor/tensor_core.mojo
#
# Description:
#   Core generic Tensor[T] and TensorView[T] with fast constructors, shape/stride
#   utilities, copy routines, and materialization. Imports are explicit (no
#   wildcards). Hot loops use simple unrolling. No assertions; all code paths are
#   defensive and branch-light.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# License:      MIT License
# SPDX-License-Identifier: MIT
# Notes:
#   - var-only; no asserts.
#   - No wildcard imports.
#   - Small inline helpers for tight loops (copy/unroll).
#   - Fast materialization path for contiguous views with arbitrary offset.
# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       momijo.tensor.tensor
# File:         src/momijo/tensor/tensor.mojo
#
# Description:
#   Core generic Tensor[T] and TensorView[T] with fast constructors,
#   shape/stride utilities, copy routines, and materialization.
#   English-only comments. No `let`, no `assert`.

from collections.list import List

# Explicit helpers (adjust paths if your project uses different names)
from momijo.tensor.helpers import *
from momijo.tensor.nanops import nanmean,nansum,nanmin


from momijo.tensor.transform import reshape,transpose,unsqueeze,squeeze_all,repeat,squeeze_axis
from momijo.tensor.broadcast import broadcast_shapes,can_broadcast_shapes,clamp
from momijo.tensor.broadcast import matmul_core_vec,clamp_int

from momijo.tensor.broadcast import matmul as _matmul_free
from momijo.tensor.broadcast import tensordot as _tensordot_free
from momijo.tensor.math import mean as _mean_free

from momijo.tensor.math import *
from momijo.tensor.cast import *
from momijo.tensor.creation import scalar_f64,scalar_f32,scalar_int
from momijo.tensor.indexing import *
from momijo.tensor.transform import flatten,view ,permute,unbind,split_sizes,cat,chunk

from momijo.tensor.creation import empty_tensor_with


# ---------- helpers: digit & int parsing (no-throw) ----------
@always_inline
fn _digit_val(ch: String) -> (Bool, Int):
    if len(ch) != 1:
        return (False, 0)
    if ch == "0": return (True, 0)
    if ch == "1": return (True, 1)
    if ch == "2": return (True, 2)
    if ch == "3": return (True, 3)
    if ch == "4": return (True, 4)
    if ch == "5": return (True, 5)
    if ch == "6": return (True, 6)
    if ch == "7": return (True, 7)
    if ch == "8": return (True, 8)
    if ch == "9": return (True, 9)
    return (False, 0)

@always_inline
fn _parse_int_no_throw(s_in: String) -> (Bool, Int):
    var s = String(s_in.strip())
    if len(s) == 0:
        return (False, 0)

    var neg = False
    var i = 0
    if s[0] == "-":
        neg = True
        i = 1
    elif s[0] == "+":
        i = 1

    if i >= len(s):
        return (False, 0)

    var acc: Int = 0
    while i < len(s):
        var (ok, d) = _digit_val(String(s[i]))
        if ok == False:
            return (False, 0)
        acc = acc * 10 + d
        i += 1

    if neg:
        acc = -acc
    return (True, acc)


@always_inline
fn _parse_opt_int(s: String) -> Optional[Int]:
    var ss = String(s.strip())
    if len(ss) == 0:
        return None
    var (ok, v) = _parse_int_no_throw(ss)
    if ok:
        return Optional[Int](v)
    return None

# ---------- helper: stringify List[String] برای دیباگ ----------
@always_inline
fn _str_string_list(xs: List[String]) -> String:
    var out = String("[")
    var i = 0
    while i < len(xs):
        if i > 0: out += String(", ")
        out += String("\"") + xs[i] + String("\"")
        i += 1
    out += String("]")
    return out.copy()



@always_inline
fn _normalize_trip(n: Int, a_opt: Optional[Int], b_opt: Optional[Int], c_opt: Optional[Int]) -> (Int, Int, Int):
    var step: Int
    if c_opt is None:
        step = 1
    else:
        step = c_opt.value()
        if step == 0:
            step = 1

    var start: Int
    var stop:  Int

    if step > 0:
        # defaults
        if a_opt is None: start = 0        else: start = a_opt.value()
        if b_opt is None: stop  = n        else: stop  = b_opt.value()

        # wrap منفی‌ها
        if start < 0: start += n
        if stop  < 0: stop  += n

        if start < 0: start = 0
        if start > n: start = n
        if stop  < 0: stop  = 0
        if stop  > n: stop  = n
    else:
        # defaults برای گام منفی
        if a_opt is None: start = n - 1    else: start = a_opt.value()
        if b_opt is None: stop  = -1       else: stop  = b_opt.value()


        if start < 0: start += n
        if stop  < -1: stop += n

        # clamp: start ∈ [-1..n-1] ، stop ∈ [-1..n-1]
        if start < -1: start = -1
        if start >= n: start = n - 1
        if stop  < -1: stop  = -1
        if stop  >= n: stop  = n - 1

    return (start, stop, step)









@always_inline
fn _ls_index(i: Int) -> IndexSel:
    return IndexSel.index(i)

@always_inline
fn _ls_slice(a: Int, b: Int, st: Int) -> IndexSel:
    return IndexSel.slice(a, b, st)

@always_inline
fn _ls_str(s: IndexSel) -> String:
    if s.tag == 0:
        return "[tag=0] index i=" + String(s.i)
    else:
        return "[tag=1] slice " + String(s.start) + ":" + String(s.stop) + ":" + String(s.step)

@always_inline
fn _ls_list_str(v: List[IndexSel]) -> String:
    var out = String("[")
    var n = len(v)
    var k = 0
    while k < n:
        if k > 0:
            out += String(", ")
        out += _ls_str(v[k])
        k += 1
    out += String("]")
    return out.copy()

@always_inline
fn _index_sel_str(sel: IndexSel) -> String:
    var s = String("[tag=") + String(sel.tag) + "] "
    if sel.tag == 0:
        s += "index i=" + String(sel.i)
    elif sel.tag == 1:
        s += "slice " + String(sel.start) + ":" + String(sel.stop) + ":" + String(sel.step)
    else:
        s += "fancy idxs=" + _list_str(sel.idxs)
    return s

@always_inline
fn _sels_str(sels: List[IndexSel]) -> String:
    var parts = List[String]()
    var k = 0
    var n = len(sels)
    while k < n:
        var tmp = IndexSel.index(0)
        tmp.__copyinit__(sels[k])
        parts.append(_index_sel_str(tmp))
        k += 1
    var out = String("[")
    var i = 0
    var m = len(parts)
    while i < m:
        out += parts[i]
        if i + 1 < m: out += ", "
        i += 1
    out += "]"
    return out.copy()


# ----------------------------
# انتخاب هر محور: یا ایندکس تکی یا برش (start, stop, step)
# ----------------------------

struct _SelViewMeta(Copyable, Movable):
    var base_offset: Int
    var out_shape:   List[Int]
    var out_coefs:   List[Int]

    fn __init__(out self, base: Int, shp: List[Int], cof: List[Int]):
        self.base_offset = base
        self.out_shape   = shp.copy()     # deep copy (List غیر-implicit)
        self.out_coefs   = cof.copy()

    # چون List[Int] ImplicitlyCopyable نیست، کپی‌ساز دستی لازم است
    fn __copyinit__(out self, other: Self):
        self.base_offset = other.base_offset
        self.out_shape   = other.out_shape.copy()
        self.out_coefs   = other.out_coefs.copy()





fn _trip_len(a: Int, b: Int, st: Int) -> Int:
    # len(range(a,b,st)) برای step≠0 (فرض: ورودی نرمال‌شده است)
    if st > 0:
        if b <= a: return 0
        var diff = b - a
        return (diff + (st - 1)) // st
    else:
        if b >= a: return 0
        var diff2 = a - b
        return (diff2 + ((-st) - 1)) // (-st)


@always_inline
fn _flat_index_affine(base: Int, coefs: List[Int], idx: List[Int]) -> Int:
    var j = base
    var k = 0
    while k < len(idx):
        j += idx[k] * coefs[k]
        k += 1
    return j
# ===== کمک‌تابع‌ها برای کار با IndexSel =====
@always_inline
fn _sel_is_index(sel: IndexSel) -> Bool:
    var r = sel.kind == .Index            # ← تطبیق با تعریف خودت
    return r

@always_inline
fn _sel_get_index(sel: IndexSel) -> Int:
    return sel.i                           # ← index انتخاب‌شده

@always_inline
fn _sel_get_trip(sel: IndexSel) -> (Int, Int, Int):
    return sel.trip                        # ← (start, stop, step) نرمال‌شده






@always_inline
fn _opt_int(x: Optional[Int]) -> String:
    if x is None:
        return "None"
    else:
        return String(x.value())
@always_inline
fn _prod_shape(shp: List[Int]) -> Int:
    var n = 1
    var i = 0
    while i < len(shp):
        n = n * shp[i]
        i += 1
    return n

@always_inline
fn _shapes_equal(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b): return False
    var i = 0
    while i < len(a):
        if a[i] != b[i]: return False
        i += 1
    return True

@always_inline
fn _flat_index(offset: Int, strides: List[Int], idx: List[Int]) -> Int:
    var s = offset
    var d = 0
    while d < len(idx):
        s = s + idx[d] * strides[d]
        d += 1
    return s

# idxLike: شمارندهٔ چندبعدی روی شکل dst
@always_inline
fn _zero_indices(rank: Int) -> List[Int]:
    var idx = List[Int]()
    var i = 0
    while i < rank:
        idx.append(0)
        i += 1
    return idx.copy()

@always_inline
fn _bump_indices(mut idx: List[Int], shp: List[Int]) -> Bool:
    # ++idx; return False when overflow (end)
    var d = len(shp) - 1
    while d >= 0:
        idx[d] = idx[d] + 1
        if idx[d] < shp[d]:
            return True
        idx[d] = 0
        d -= 1
    return False

# ---------------- core: no external methods needed ----------------
# ---------------- Debug helpers ----------------
@always_inline
fn _list_str(xs: List[Int]) -> String:
    var s = String("[")
    var i = 0
    var n = len(xs)
    while i < n:
        s = s + String(xs[i])
        if i + 1 < n: s = s + ", "
        i += 1
    s = s + "]"
    return s

@always_inline
fn _dump_view[T: ImplicitlyCopyable & Copyable & Movable](label: String, x: Tensor[T]) -> None:
    # Prints shape/strides/offset and computed numel
    var numel = _prod_shape(x._shape)
    print(label + " shape=" + _list_str(x._shape) +
          " strides=" + _list_str(x._strides) +
          " offset=" + String(x._offset) +
          " numel=" + String(numel))


# ---------------- Core with debug prints ----------------
@always_inline
fn _assign_view_tensor[T: ImplicitlyCopyable & Copyable & Movable](mut dst: Tensor[T], rhs: Tensor[T]) -> None:

    var rhs_total = _prod_shape(rhs._shape)

    # ---------- treat any single-element RHS as scalar (rank-0 OR shape=[1] OR any numel==1 view) ----------
    if rhs_total == 1:
        # get that single element robustly (works for any strides/offset)
        var zeros = _zero_indices(len(rhs._shape))
        var j_rhs = _flat_index(rhs._offset, rhs._strides, zeros)
        var v = rhs._data[j_rhs]

        if len(dst._shape) == 0:
            dst._data[dst._offset] = v
            return

        var total = _prod_shape(dst._shape)
        if total == 0:
            return

        var idx = _zero_indices(len(dst._shape))
        var wrote = 0
        while True:
            var j = _flat_index(dst._offset, dst._strides, idx)
            dst._data[j] = v
            wrote += 1
            if wrote <= 4:
                print("[SETITEM] fill @j=" + String(j))
            if _bump_indices(idx, dst._shape) == False:
                break
        return

    # ---------- exact-shape copy (both non-contig OK) ----------
    if _shapes_equal(dst._shape, rhs._shape) == False:
        print("  dst.shape=" + _list_str(dst._shape) + " rhs.shape=" + _list_str(rhs._shape))
        return

    if len(dst._shape) == 0:
        var zeros_rhs = _zero_indices(len(rhs._shape))
        var j_rhs0 = _flat_index(rhs._offset, rhs._strides, zeros_rhs)
        dst._data[dst._offset] = rhs._data[j_rhs0]
        return

    var idx2 = _zero_indices(len(dst._shape))
    var copied = 0
    while True:
        var j_dst = _flat_index(dst._offset, dst._strides, idx2)
        var j_rhs = _flat_index(rhs._offset, rhs._strides, idx2)
        dst._data[j_dst] = rhs._data[j_rhs]
        copied += 1
        if copied <= 4:
            print("[SETITEM] copy j_dst=" + String(j_dst) + " <= j_rhs=" + String(j_rhs))
        if _bump_indices(idx2, dst._shape) == False:
            break



@always_inline
fn _rank(shp: List[Int]) -> Int:
    return len(shp)

@always_inline
fn _right_align_map(dst_shape: List[Int], src_shape: List[Int]) -> List[Int]:
    var r_dst = len(dst_shape)
    var r_src = len(src_shape)
    var m = List[Int]()
    m.reserve(r_dst)
    var i = 0
    while i < r_dst:
        m.append(-1)
        i += 1
    var d = r_dst - 1
    var s = r_src - 1
    while d >= 0 and s >= 0:
        m[d] = s
        d -= 1
        s -= 1
    return m

@always_inline
fn _can_broadcast(dst_shape: List[Int], src_shape: List[Int]) -> Bool:
    var r_dst = len(dst_shape)
    var r_src = len(src_shape)
    var d = r_dst - 1
    var s = r_src - 1
    while d >= 0:
        var dd = dst_shape[d]
        var ss = 1
        if s >= 0:
            ss = src_shape[s]
        if not (ss == 1 or ss == dd):
            return False
        d -= 1
        s -= 1
    return True

@always_inline
fn _advance_index(mut idx: List[Int], shape: List[Int]) -> Bool:
    var r = len(shape)
    if r == 0:
        return False
    var ax = r - 1
    while True:
        idx[ax] = idx[ax] + 1
        if idx[ax] < shape[ax]:
            return True
        idx[ax] = 0
        if ax == 0:
            return False
        ax -= 1

@always_inline
fn _offset_of(shape: List[Int], strides: List[Int], base_off: Int, idx: List[Int]) -> Int:
    var off = base_off
    var i = 0
    var r = len(shape)
    while i < r:
        off = off + idx[i] * strides[i]
        i += 1
    return off

# In-place fill over a (possibly strided) view.
@always_inline
fn _fill_view[T: ImplicitlyCopyable & Copyable & Movable](mut v: Tensor[T], s: T) -> None:
    var r = len(v._shape)
    if r == 0:
        v._data[v._offset] = s
        return
    var idx = List[Int]()
    idx.reserve(r)
    var i = 0
    while i < r:
        idx.append(0)
        i += 1
    while True:
        var off = _offset_of(v._shape, v._strides, v._offset, idx)
        v._data[off] = s
        if _advance_index(idx, v._shape) == False:
            break

# In-place copy with right-aligned broadcasting from src -> dst view.
@always_inline
fn _copy_view_broadcast[T: ImplicitlyCopyable & Copyable & Movable](mut dst: Tensor[T], src: Tensor[T]) -> None:
    # Fast path: exact shape match (including rank)
    if len(dst._shape) == len(src._shape):
        var same = True
        var k = 0
        while k < len(dst._shape):
            if dst._shape[k] != src._shape[k]:
                same = False
                break
            k += 1
        if same:
            # shape-equal strided copy
            var r = len(dst._shape)
            if r == 0:
                dst._data[dst._offset] = src._data[src._offset]
                return
            var idx = List[Int]()
            idx.reserve(r)
            var i = 0
            while i < r:
                idx.append(0)
                i += 1
            while True:
                var doff = _offset_of(dst._shape, dst._strides, dst._offset, idx)
                var soff = _offset_of(src._shape, src._strides, src._offset, idx)
                dst._data[doff] = src._data[soff]
                if _advance_index(idx, dst._shape) == False:
                    break
            return

    # General broadcast path
    assert(_can_broadcast(dst._shape, src._shape), "broadcast mismatch in __setitem__")
    var r = len(dst._shape)
    if r == 0:
        # dst scalar view
        var sval = src
        if len(src._shape) == 0:
            dst._data[dst._offset] = src._data[src._offset]
        else:
            # src must broadcast to scalar (OK): just take first element by row-major
            var first = src._data[src._offset]
            dst._data[dst._offset] = first
        return

    var idx_dst = List[Int]()
    idx_dst.reserve(r)
    var i = 0
    while i < r:
        idx_dst.append(0)
        i += 1

    # Build a right-aligned map from dst axes to src axes
    var map = _right_align_map(dst._shape, src._shape)

    # We reuse idx_dst's length for on-the-fly src indexing (projected)
    var idx_src = List[Int]()
    idx_src.reserve(r)
    i = 0
    while i < r:
        idx_src.append(0)
        i += 1

    while True:
        # Project dst index -> src index
        var a = 0
        while a < r:
            var s_ax = map[a]
            if s_ax < 0:
                idx_src[a] = 0
            else:
                if src._shape[s_ax] == 1:
                    idx_src[a] = 0
                else:
                    idx_src[a] = idx_dst[a]
            a += 1

        var doff = _offset_of(dst._shape, dst._strides, dst._offset, idx_dst)
        # Build a compact src idx along src rank to compute offset
        # We can compute in-place by reading only mapped coordinates.
        var so = src._offset
        var sr = len(src._shape)
        var ax = 0
        while ax < sr:
            # find the dst axis that maps to ax
            # because map is right-aligned + ordered, we can reconstruct:
            # dst axis d that has map[d] == ax
            var d = len(map) - 1
            var found = False
            while d >= 0:
                if map[d] == ax:
                    var step = idx_dst[d]
                    if src._shape[ax] == 1:
                        step = 0
                    so = so + step * src._strides[ax]
                    found = True
                    break
                d -= 1
            if found == False:
                # extra leading axis on dst → src has no axis → broadcast as 0
                # (so no addition needed)
                pass
            ax += 1

        dst._data[doff] = src._data[so]

        if _advance_index(idx_dst, dst._shape) == False:
            break


# ========================= slice primitives =========================
struct StrIndex:
    var spec: String

    @always_inline
    fn __init__(out self, spec: String):
        self.spec = spec

@always_inline
fn S(spec: String) -> StrIndex:
    return StrIndex(spec)

@always_inline
fn _opt_int(tok: String) -> Optional[Int]:
    # empty → None ; valid int → Some(v) ; invalid → None
    if len(tok) == 0:
        return None
    else:
        var (ok, v) = parse_int_safe(tok)
        if ok:
            return Optional[Int](v)
        else:
            return None

@always_inline
fn _opt_step(tok: String) -> Optional[Int]:
    # empty → None ; valid nonzero → Some(v) ; invalid/zero → Some(1) (non-raising clamp)
    if len(tok) == 0:
        return None
    else:
        var (ok, v) = parse_int_safe(tok)
        if ok and v != 0:
            return Optional[Int](v)
        else:
            return None

struct EllipsisSpec:
    fn __init__(out self): pass
# ---------------------------------------------------------------------------
# Slice normalization (Python semantics for 1-D axis)
# ---------------------------------------------------------------------------
@always_inline
fn _normalize_slice_1d(n: Int,
                       start_opt: Optional[Int],
                       stop_opt: Optional[Int],
                       step_opt: Optional[Int]) -> Tuple[Int, Int, Int]:
    # Debug: raw inputs


    var step: Int
    if step_opt is None: step = 1 else: step = step_opt.value()
    if step == 0: step = 1

    var start: Int
    var stop: Int

    if step > 0:
        if start_opt is None: start = 0 else: start = start_opt.value()
        if stop_opt  is None: stop  = n else: stop  = stop_opt.value()
        if start < 0: start += n
        if stop  < 0: stop  += n
        if start < 0: start = 0
        if start > n: start = n
        if stop  < 0: stop  = 0
        if stop  > n: stop  = n

    else:
        # step < 0
        if start_opt is None: start = n - 1 else: start = start_opt.value()
        if stop_opt  is None: stop  = -1      else: stop  = stop_opt.value()

        # translate negatives (keep -1 sentinel)
        if start < 0: start = start + n
        if stop  < 0 and stop != -1: stop = stop + n

        # clamp allowing -1
        if start < -1: start = -1
        if start >  n-1: start = n-1
        if stop  < -1: stop  = -1
        if stop  >  n-1: stop  = n-1



    # Final trip
    return (start, stop, step)



@always_inline
fn make_fancy_sel(js: List[Int]) -> IndexSel:
    return IndexSel.fancy(js)

@always_inline
fn clone_sel(x: IndexSel) -> IndexSel:
    # ctor خودش js.copy() انجام می‌دهد، پس alias نمی‌شود
    return IndexSel(x.tag, x.i, x.start, x.stop, x.step, x.idxs)

# ---------------------------------------------------------------------------
# Sample 1D slice execution (only if your select(...) doesn't handle SLICE)
# ---------------------------------------------------------------------------

@always_inline
fn _apply_slice_axis0_1d[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T], trip: SliceSpec
) -> Tensor[T]:
    # trip = (start, stop, step), half-open for step>0, mirrored rule for step<0
    var start = trip[0]
    var stop  = trip[1]
    var step  = trip[2]

    # Build output list by stepping through the view
    var out = List[T]()
    if step > 0:
        var i = start
        while i < stop:
            out.append(x._data[i])
            i += step
    else:
        var i2 = start
        while i2 > stop:
            out.append(x._data[i2])
            i2 += step  # step is negative
    # Shape/strides for 1-D result
    var out_shape = List[Int]()
    out_shape.append(len(out))
    var out_strides = mk_strides(out_shape)
    return Tensor[T](out, out_shape, out_strides, 0)

fn share[T: ImplicitlyCopyable & Copyable & Movable](
        data: List[T],
        shape: List[Int],
        strides: List[Int],
        offset: Int
    ) -> Tensor[T]:
        var t: Tensor[T]
        # Direct field assignment (no copies!)
        t._data = data                # **share the same storage**
        t._shape = shape.copy()       # metadata can be copied (cheap)
        t._strides = strides.copy()
        t._offset = offset
        return t
# =============================== Tensor ===============================
struct Tensor[T: ImplicitlyCopyable & Copyable & Movable](Copyable, Movable):
    var _data: List[T]
    var _shape: List[Int]
    var _strides: List[Int]
    var _offset: Int


    fn resize_like_with_pad(self, new_tensor: Tensor[T]) -> Tensor[T]:
        return resize_like_with_pad(self, new_shape)
     # (data, shape, strides, offset)
    fn __init__(out self, data: List[T], shape: List[Int], strides: List[Int], off: Int):
        self._data    = data.copy()
        self._shape   = shape.copy()
        self._strides = strides.copy()
        self._offset  = off

    # (data, shape) -> row-major, offset=0
    fn __init__(out self, data: List[T], shape: List[Int]):
        var shp  = copy_ints(shape)
        var strd = mk_strides(shp)
        self._data    = data.copy()
        self._shape   = shp.copy()
        self._strides = strd.copy()
        self._offset  = 0

    # (shape, fill) => dense new buffer, offset=0
    fn __init__(out self, shape: List[Int], fill: T):
        var shp  = copy_ints(shape)
        var strd = mk_strides(shp)
        var n    = numel_shape(shp)
        var buf  = List[T]()
        buf.reserve(n)
        var i = 0
        while i < n:
            buf.append(fill)
            i += 1
        self._data    = buf.copy()
        self._shape   = shp.copy()
        self._strides = strd.copy()
        self._offset  = 0

    # (shape, flat) => copy/extend/truncate, offset=0
    fn __init__(out self, shape: List[Int], flat: List[T]):
        var shp  = copy_ints(shape)
        var strd = mk_strides(shp)
        var n    = numel_shape(shp)
        var buf  = List[T]()
        buf.reserve(n)
        var src_n = len(flat)
        var i = 0
        if src_n != 0:
            var last = flat[src_n - 1]
            while i < n:
                var v = last
                if i < src_n: v = flat[i]
                buf.append(v)
                i += 1
        self._data    = buf.copy()
        self._shape   = shp.copy()
        self._strides = strd.copy()
        self._offset  = 0

    # 1D nested list
    fn __init__(out self, data: List[T]):
        var shp  = infer_shape_1d[T](data)
        var strd = mk_strides(shp)
        var buf  = List[T]()
        var n    = len(data)
        buf.reserve(n)
        append_block_unrolled16(buf, data, 0, n)
        self._data    = buf.copy()
        self._shape   = shp.copy()
        self._strides = strd.copy()
        self._offset  = 0

    # 2D nested list (safe truncate)
    fn __init__(out self, rows: List[List[T]]):
        var (n0, n1) = infer_shape_2d[T](rows)
        var shp = List[Int]()
        shp.append(n0); shp.append(n1)
        var buf  = flatten_2d[T](rows, n0, n1)
        var strd = mk_strides(shp)
        self._data    = buf.copy()
        self._shape   = shp.copy()
        self._strides = strd.copy()
        self._offset  = 0

    # 3D nested list (safe truncate)
    fn __init__(out self, rows: List[List[List[T]]]):
        var (d0, d1, d2) = infer_shape_3d[T](rows)
        var shp = List[Int]()
        shp.append(d0); shp.append(d1); shp.append(d2)
        var buf  = flatten_3d[T](rows, d0, d1, d2)
        var strd = mk_strides(shp)
        self._data    = buf.copy()
        self._shape   = shp.copy()
        self._strides = strd.copy()
        self._offset  = 0

    # 4D nested list (safe truncate)
    fn __init__(out self, rows: List[List[List[List[T]]]]):
        var (a, b, c, d) = infer_shape_4d[T](rows)
        var shp = List[Int]()
        shp.append(a); shp.append(b); shp.append(c); shp.append(d)
        var buf  = flatten_4d[T](rows, a, b, c, d)
        var strd = mk_strides(shp)
        self._data    = buf.copy()
        self._shape   = shp.copy()
        self._strides = strd.copy()
        self._offset  = 0

    # 5D nested list (safe truncate)
    fn __init__(out self, rows: List[List[List[List[List[T]]]]]):
        var (a, b, c, d, e) = infer_shape_5d[T](rows)
        var shp = List[Int]()
        shp.append(a); shp.append(b); shp.append(c); shp.append(d); shp.append(e)
        var buf  = flatten_5d[T](rows, a, b, c, d, e)
        var strd = mk_strides(shp)
        self._data    = buf.copy()
        self._shape   = shp.copy()
        self._strides = strd.copy()
        self._offset  = 0

    # copy-init
    fn __copyinit__(out self, other: Self):
        self._data    = other._data.copy()
        self._shape   = other._shape.copy()
        self._strides = other._strides.copy()
        self._offset  = other._offset




    # ---------------- view maker (zero-copy) ----------------
    @always_inline
    fn _make_view(self, new_shape: List[Int], new_strides: List[Int], new_off: Int) -> Tensor[T]:
        # No data copy; same buffer, adjusted view
        return Tensor[T](self._data, new_shape, new_strides, new_off)

    # =========================== Indexing helpers ===========================

    @always_inline
    fn idx(self, i: Int) -> IndexSel:
        return make_index_sel(i)

    @always_inline
    fn slc(self, a: Int, b: Int, step: Int = 1) -> IndexSel:
        return make_slice_sel((a, b, step))

    @always_inline
    fn all_axis(self) -> IndexSel:
        # Full axis: start=0, stop=dim will be normalized later per-axis
        return make_slice_sel((0, 0, 1))  # stop will be filled from axis dim

    # @always_inline
    # fn take(self, js: List[Int]) -> IndexSel:
    #     return make_fancy_sel(js)

    # @always_inline
    # fn take(self, js: Tensor[Int]) -> IndexSel:
    #     ls=js.to_list()
    #     return make_fancy_sel(ls)

    @always_inline
    fn take(self, js: List[Int], axis: Int = 0) -> Tensor[T]:
        return take(self, js, axis)

    @always_inline
    fn take(self, js: Tensor[Int], axis: Int = 0) -> Tensor[T]:
        return take(self, js, axis)



    # Normalize a slice selector against dim and (possibly) negative step.
    # Returns (start, stop, step) normalized s.t. resulting axis length >= 0
    @always_inline
    fn _normalize_slice(self, dim: Int, st0: Int, sp0: Int, step0: Int) -> (Int, Int, Int):
        var step = step0
        if step == 0: step = 1

        # all_axis() sentinel (0,0,1) → full
        if st0 == 0 and sp0 == 0 and step == 1:
            return (0, dim, 1)

        var start = st0
        var stop  = sp0

        if step > 0:
            # "to the end" if stop unspecified but start set
            if sp0 == 0 and st0 != 0:
                stop = dim
            if start < 0: start = dim + start
            if stop  < 0: stop  = dim + stop
            start = clamp_int(start, 0, dim)
            stop  = clamp_int(stop,  0, dim)
        else:
            # reverse full when both unset/zero-ish
            if st0 == 0 and sp0 == 0:
                start = dim - 1
                stop  = -1
            else:
                if start < 0: start = dim + start
                # IMPORTANT: for negative step keep stop negative if user passed -1;
                # do NOT translate stop via (dim + stop). Just clamp to [-1, dim-1].
                stop = clamp_int(stop, -1, dim - 1)
            start = clamp_int(start, -1, dim - 1)
        return (start, stop, step)

    # =========================== Core: select_view ===========================
    # Build a ZERO-COPY view if possible (only indices and slices). Fancy indices force copy.
    fn select_view(self, sels_in: List[IndexSel]) -> Tensor[T]:
        var sels = self.pad_full_axes(sels_in)
        var ndim = len(self._shape)

        var new_shape   = List[Int]()
        var new_strides = List[Int]()
        var new_off     = self._offset

        var ax = 0
        while ax < ndim:
            var dim = self._shape[ax]
            if is_index(sels[ax]):
                var ii = wrap_axis_index(get_index(sels[ax]), dim)
                new_off = new_off + ii * self._strides[ax]
            elif is_slice(sels[ax]):
                var (st0, sp0, step0) = get_slice(sels[ax])

                var (st, sp, step1) = self._normalize_slice(dim, st0, sp0, step0)
                var out_len = axis_len_from_slice(st, sp, step1)

                new_off = new_off + st * self._strides[ax]
                new_shape.append(out_len)
                new_strides.append(self._strides[ax] * step1)

            else:
                return self.select_copy(sels)   # fancy ⇒ کپی
            ax = ax + 1

        return self._make_view(new_shape, new_strides, new_off)



    # =========================== Core: select_copy ===========================
    fn select_copy(self, sels_in: List[IndexSel]) -> Tensor[T]:
        var sels = self.pad_full_axes(sels_in)
        var ndim = len(self._shape)

        var keep_axes = List[Bool]()
        var maps = List[List[Int]]()
        keep_axes.reserve(ndim)

        var out_shape = List[Int]()

        var ax = 0
        while ax < ndim:
            var dim = self._shape[ax]
            if is_index(sels[ax]):
                _ = wrap_axis_index(get_index(sels[ax]), dim)
                keep_axes.append(False)
                var dummy = List[Int]()
                maps.append(dummy.copy())
            elif is_slice(sels[ax]):
                var (st0, sp0, step0) = get_slice(sels[ax])
                var (st, sp, step1) = self._normalize_slice(dim, st0, sp0, step0)
                var L = axis_len_from_slice(st, sp, step1)
                var idxs = List[Int]()
                idxs.reserve(L)
                var k = 0
                var cur = st
                while k < L:
                    idxs.append(cur)
                    cur = cur + step1
                    k = k + 1
                keep_axes.append(True)
                maps.append(idxs.copy())
                out_shape.append(L)
            else:
                var js_raw = get_fancy_list(sels[ax])
                var js = List[Int]()
                js.reserve(len(js_raw))
                var p = 0
                while p < len(js_raw):
                    js.append(wrap_axis_index(js_raw[p], dim))
                    p = p + 1
                keep_axes.append(True)
                maps.append(js.copy())
                out_shape.append(len(js))
            ax = ax + 1

        if len(out_shape) == 0:
            var lin = self._offset
            var a2 = 0
            while a2 < ndim:
                if is_index(sels[a2]):
                    var ii2 = wrap_axis_index(get_index(sels[a2]), self._shape[a2])
                    lin = lin + ii2 * self._strides[a2]
                a2 = a2 + 1
            var data = List[T]()
            data.append(self._data[lin])
            var shp = List[Int]()
            return Tensor[T](data, shp)

        var out_rm = compute_row_major_strides(out_shape)
        var out_n = numel_shape(out_shape)
        var out_flat = List[T]()
        out_flat.reserve(out_n)

        var q = 0
        while q < out_n:
            var lin_src = self._offset
            var rem = q
            var kept_i = 0
            var a3 = 0
            while a3 < ndim:
                if keep_axes[a3]:
                    var stride_q = out_rm[kept_i]
                    var coord = rem // stride_q
                    var len_ax = out_shape[kept_i]
                    if len_ax > 0:
                        coord = coord % len_ax
                    var src_index = maps[a3][coord]
                    lin_src = lin_src + src_index * self._strides[a3]
                    kept_i = kept_i + 1
                else:
                    var ii3 = wrap_axis_index(get_index(sels[a3]), self._shape[a3])
                    lin_src = lin_src + ii3 * self._strides[a3]
                a3 = a3 + 1
            out_flat.append(self._data[lin_src])
            q = q + 1
        return Tensor[T](out_flat, out_shape)

    # =========================== Public: select (prefers view) ===========================
    fn select(self, sels_in: List[IndexSel]) -> Tensor[T]:
        var sels = self.pad_full_axes(sels_in)
        var viewable = True
        var i = 0
        while i < len(sels):
            if is_fancy(sels[i]):
                viewable = False
                break
            i += 1



        if viewable:
            return self.select_view(sels)
        return self.select_copy(sels)


    # ------------------------------------------
    # String parser across axes ("," separated)
    # Supports ":", "1", "-1", "1:3", "::2", "5:1:-1"
    # ------------------------------------------
    @always_inline
    fn _parse_selector_string(self, spec: String) -> List[IndexSel]:
        var sels = List[IndexSel]()
        var tokens = spec.split(",")
        var tcount = len(tokens)

        var d = len(self._shape)        # rank
        var ax = 0                      # current axis we are filling
        var i = 0

        var explicit = 0
        var k = 0
        while k < tcount:
            var rawk = trim_ascii(to_string_owned(tokens[k]))
            if rawk != "...":
                explicit = explicit + 1
            k = k + 1

        var saw_ellipsis = False

        while i < tcount and ax < d:
            var raw = trim_ascii(to_string_owned(tokens[i]))

            # -------------------------
            # Ellipsis: expand to fill remaining axes
            # -------------------------
            if raw == "...":
                if not saw_ellipsis:
                    var remaining = d - explicit
                    if remaining < 0:
                        remaining = 0
                    var c = 0
                    while c < remaining and ax < d:
                        var n_ax = self._axis_len(ax)
                        sels.append(make_slice_sel((0, n_ax, 1)))
                        ax = ax + 1
                        c = c + 1
                    saw_ellipsis = True
                else:
                    # سیاست non-raising: Ellipsis اضافه را نادیده بگیر
                    pass
                i = i + 1
                continue

            # -------------------------
            # ":" یا خالی → فول‌اسلایس روی محور جاری
            # -------------------------
            if len(raw) == 0 or raw == ":":
                var n_ax2 = self._axis_len(ax)
                sels.append(make_slice_sel((0, n_ax2, 1)))
                ax = ax + 1
                i = i + 1
                continue

            # -------------------------
            # برسی وجود ":" برای حالت اسلایس
            # -------------------------
            var colon_pos = raw.find(":")
            if colon_pos >= 0:
                var parts = raw.split(":")
                var p0 = String("")
                var p1 = String("")
                var p2 = String("")
                if len(parts) >= 1:
                    p0 = trim_ascii(to_string_owned(parts[0]))
                if len(parts) >= 2:
                    p1 = trim_ascii(to_string_owned(parts[1]))
                if len(parts) >= 3:
                    p2 = trim_ascii(to_string_owned(parts[2]))

                # Tokens → Optional[Int]
                var start_opt = _opt_int(p0)
                var stop_opt  = _opt_int(p1)
                var step_opt  = _opt_step(p2)

                var n_ax3 = self._axis_len(ax)
                var trip = _normalize_slice_1d(n_ax3, start_opt, stop_opt, step_opt)

                # دقیقاً ":" (سه None) → فول‌اسلایس صریح
                if start_opt is None and stop_opt is None and step_opt is None:
                    trip = (0, n_ax3, 1)

                sels.append(make_slice_sel(trip))
                ax = ax + 1
                i = i + 1
                continue

            # -------------------------
            # ایندکس عددی تکی
            # -------------------------
            var ii = 0
            var (ok_i, v_i) = parse_int_safe(raw)
            if ok_i:
                ii = v_i
            sels.append(make_index_sel(ii))
            ax = ax + 1
            i = i + 1

        while ax < d:
            var n_ax4 = self._axis_len(ax)
            sels.append(make_slice_sel((0, n_ax4, 1)))
            ax = ax + 1

        return sels.copy()


    fn _view_after_prefix_ints(self: Tensor[T], idxs: List[Int]) -> Tensor[T]:
        var d = len(self._shape)
        var k = len(idxs)

        var off = self._offset
        var ax  = 0
        while ax < d and ax < k:
            var dim = self._shape[ax]
            var ii  = wrap_axis_index(idxs[ax], dim)
            off = off + ii * self._strides[ax]
            ax = ax + 1

        # Remaining axes (if any) stay as view dims
        var new_shape   = List[Int]()
        var new_strides = List[Int]()
        var r = ax
        while r < d:
            new_shape.append(self._shape[r])
            new_strides.append(self._strides[r])
            r = r + 1

        return self._make_view(new_shape, new_strides, off)

    @always_inline
    fn _axis_len(self, axis: Int) -> Int:
        var r = len(self._shape)
        if r == 0: return 1
        return self._shape[axis]

    @always_inline
    fn _sel_from_int(self, axis: Int, i: Int) -> IndexSel:
        return make_index_sel(i)

    @always_inline
    fn _sel_from_empty(self, axis: Int) -> IndexSel:
        var n = self._axis_len(axis)
        return make_slice_sel((0, n, 1))

    @always_inline
    fn _sel_from_slice(self, axis: Int, s: Slice) -> IndexSel:
        var n = self._axis_len(axis)
        var st: Optional[Int] = if s.start is None: None else: Optional[Int](s.start.value())
        var sp: Optional[Int] = if s.end   is None: None else: Optional[Int](s.end.value())
        var stp: Optional[Int]= if s.step  is None: None else: Optional[Int](s.step.value())
        var trip = _normalize_slice_1d(n, st, sp, stp)   # (start, stop, step) — با نگه‌داشتن -1 برای step<0
        return make_slice_sel(trip)

    # ===================== Native slice-based __getitem__ overloads =====================


    # --------------------------------------
    # 2) __getitem__ برای StrIndex
    # --------------------------------------
    @always_inline
    fn __getitem__(self, s: StrIndex) -> Tensor[T]:
        var sels = self._parse_selector_string(s.spec)
        return self.select(sels)

    # --------------------------------------
    #    (توجه: امضای __getitem__(self, spec: String) باید حذف باشد)
    # --------------------------------------
    @always_inline
    fn __getitem__(self, lit: StringLiteral) -> Tensor[T]:
        return self.__getitem__(StrIndex(String(lit)))

    @always_inline
    fn __getitem__(self, i0: Int) -> Tensor[T]:
        var sels = List[IndexSel]()
        sels.append(make_index_sel(i0))
        return self.select(sels)

    @always_inline
    fn __getitem__(self, s: Slice) -> Tensor[T]:
        var n: Int
        var r = len(self._shape)
        if r == 0: n = 1 else: n = self._shape[0]

        var start_opt: Optional[Int]
        if s.start is None: start_opt = None else: start_opt = Optional[Int](s.start.value())
        var stop_opt: Optional[Int]
        if s.end   is None: stop_opt  = None else: stop_opt  = Optional[Int](s.end.value())
        var step_opt: Optional[Int]
        if s.step  is None: step_opt  = None else: step_opt  = Optional[Int](s.step.value())

        var trip = _normalize_slice_1d(n, start_opt, stop_opt, step_opt)

        var sels = List[IndexSel]()
        sels.append(make_slice_sel(trip))
        return self.select(sels)


    @always_inline
    fn _getitem_slice_spec(self, trip: (Int, Int, Int)) -> Tensor[T]:
        var sels = List[IndexSel]()
        sels.append(make_slice_sel(trip))
        return self.select(sels)

    # ------------------------------------------------------------------------------
    # 0) Full 2D "[: , :]" using two empty tuples (native ":")
    # ------------------------------------------------------------------------------
    @always_inline
    fn __getitem__(self, _e0: Tuple[], _e1: Tuple[]) -> Tensor[T]:
        var n0 = self._axis_len(0)
        var n1 = self._axis_len(1)
        var trip0 = (0, n0, 1)
        var trip1 = (0, n1, 1)
        var sels = List[IndexSel]()
        sels.append(make_slice_sel(trip0))
        sels.append(make_slice_sel(trip1))
        print("[EE] TRIP0:", trip0[0], trip0[1], trip0[2])
        print("[EE] TRIP1:", trip1[0], trip1[1], trip1[2])
        return self.select(sels)

    # ------------------------------------------------------------------------------
    # 1) Mixed with empty tuple: arr[ :, i ]  and  arr[ i, : ]
    # ------------------------------------------------------------------------------
    @always_inline
    fn __getitem__(self, _e0: Tuple[], i1: Int) -> Tensor[T]:
        var n0 = self._axis_len(0)
        var trip0 = (0, n0, 1)

        var sels = List[IndexSel]()
        sels.append(make_slice_sel(trip0))
        sels.append(make_index_sel(i1))

        print("TRIP0:", trip0[0], trip0[1], trip0[2])
        return self.select(sels)

    @always_inline
    fn __getitem__(self, i0: Int, _e1: Tuple[]) -> Tensor[T]:
        var n1 = self._axis_len(1)
        var trip1 = (0, n1, 1)

        var sels = List[IndexSel]()
        sels.append(make_index_sel(i0))
        sels.append(make_slice_sel(trip1))

        print("TRIP1:", trip1[0], trip1[1], trip1[2])
        return self.select(sels)



    # ------------------------------------------------------------------------------
    # 3) Mixed index/slice (native Slice on one axis)
    #     arr[i, a:b:c]   and   arr[a:b:c, i]
    # ------------------------------------------------------------------------------
    @always_inline
    fn __getitem__(self, i0: Int, s1: Slice) -> Tensor[T]:
        var n1 = self._axis_len(1)

        var start1: Optional[Int]
        if s1.start is None: start1 = None else: start1 = Optional[Int](s1.start.value())

        var stop1: Optional[Int]
        if s1.end is None:   stop1 = None else: stop1 = Optional[Int](s1.end.value())

        var step1: Optional[Int]
        if s1.step is None:  step1 = None else: step1 = Optional[Int](s1.step.value())

        var trip1 = _normalize_slice_1d(n1, start1, stop1, step1)

        var sels = List[IndexSel]()
        sels.append(make_index_sel(i0))
        sels.append(make_slice_sel(trip1))

        print("TRIP1:", trip1[0], trip1[1], trip1[2])  # e.g., for[::-1] -> (n1-1, -1, -1)
        return self.select(sels)

    @always_inline
    fn __getitem__(self, s0: Slice, i1: Int) -> Tensor[T]:
        var n0 = self._axis_len(0)

        var start0: Optional[Int]
        if s0.start is None: start0 = None else: start0 = Optional[Int](s0.start.value())

        var stop0: Optional[Int]
        if s0.end is None:   stop0 = None else: stop0 = Optional[Int](s0.end.value())

        var step0: Optional[Int]
        if s0.step is None:  step0 = None else: step0 = Optional[Int](s0.step.value())

        var trip0 = _normalize_slice_1d(n0, start0, stop0, step0)

        var sels = List[IndexSel]()
        sels.append(make_slice_sel(trip0))
        sels.append(make_index_sel(i1))

        print("TRIP0:", trip0[0], trip0[1], trip0[2])
        return self.select(sels)

    # ------------------------------------------------------------------------------
    # 4) Pure slices on both axes: arr[a:b:c, d:e:f]
    #     covers "[: , :]", "[::2, 1:]" , "[::-1, ::-1]" , "[0:2, 0:2]" , "[1:, 1:]" ...
    # ------------------------------------------------------------------------------
    @always_inline
    fn __getitem__(self, s0: Slice, s1: Slice) -> Tensor[T]:
        # axis-0
        var n0 = self._axis_len(0)
        var start0: Optional[Int]
        if s0.start is None: start0 = None else: start0 = Optional[Int](s0.start.value())
        var stop0: Optional[Int]
        if s0.end is None:   stop0 = None else: stop0 = Optional[Int](s0.end.value())
        var step0: Optional[Int]
        if s0.step is None:  step0 = None else: step0 = Optional[Int](s0.step.value())
        var trip0 = _normalize_slice_1d(n0, start0, stop0, step0)
        # force full-slice if ":" exactly
        if s0.start is None and s0.end is None and s0.step is None:
            trip0 = (0, n0, 1)

        # axis-1
        var n1 = self._axis_len(1)
        var start1: Optional[Int]
        if s1.start is None: start1 = None else: start1 = Optional[Int](s1.start.value())
        var stop1: Optional[Int]
        if s1.end is None:   stop1 = None else: stop1 = Optional[Int](s1.end.value())
        var step1: Optional[Int]
        if s1.step is None:  step1 = None else: step1 = Optional[Int](s1.step.value())
        var trip1 = _normalize_slice_1d(n1, start1, stop1, step1)
        if s1.start is None and s1.end is None and s1.step is None:
            trip1 = (0, n1, 1)

        var sels = List[IndexSel]()
        sels.append(make_slice_sel(trip0))
        sels.append(make_slice_sel(trip1))

        return self.select(sels)


    # ------------------------------------------------------------------------------
    # 5) Mixed "empty tuple + slice" (for completeness):
    #     arr[:, a:b:c]   and   arr[a:b:c, :]
    # ------------------------------------------------------------------------------
    @always_inline
    fn __getitem__(self, _e0: Tuple[], s1: Slice) -> Tensor[T]:
        var n0 = self._axis_len(0)
        var trip0 = (0, n0, 1)

        var n1 = self._axis_len(1)
        var start1: Optional[Int]
        if s1.start is None: start1 = None else: start1 = Optional[Int](s1.start.value())
        var stop1: Optional[Int]
        if s1.end is None:   stop1 = None else: stop1 = Optional[Int](s1.end.value())
        var step1: Optional[Int]
        if s1.step is None:  step1 = None else: step1 = Optional[Int](s1.step.value())
        var trip1 = _normalize_slice_1d(n1, start1, stop1, step1)

        var sels = List[IndexSel]()
        sels.append(make_slice_sel(trip0))
        sels.append(make_slice_sel(trip1))

        print("TRIP0:", trip0[0], trip0[1], trip0[2])
        print("TRIP1:", trip1[0], trip1[1], trip1[2])
        return self.select(sels)

    @always_inline
    fn __getitem__(self, s0: Slice, _e1: Tuple[]) -> Tensor[T]:
        var n0 = self._axis_len(0)
        var start0: Optional[Int]
        if s0.start is None: start0 = None else: start0 = Optional[Int](s0.start.value())
        var stop0: Optional[Int]
        if s0.end is None:   stop0 = None else: stop0 = Optional[Int](s0.end.value())
        var step0: Optional[Int]
        if s0.step is None:  step0 = None else: step0 = Optional[Int](s0.step.value())
        var trip0 = _normalize_slice_1d(n0, start0, stop0, step0)

        var n1 = self._axis_len(1)
        var trip1 = (0, n1, 1)

        var sels = List[IndexSel]()
        sels.append(make_slice_sel(trip0))
        sels.append(make_slice_sel(trip1))

        print("TRIP0:", trip0[0], trip0[1], trip0[2])
        print("TRIP1:", trip1[0], trip1[1], trip1[2])
        return self.select(sels)

    # --------------------------------------------------------------------------
    # 3D indexing/slicing
    # سازگار با الگوی موجود: Tuple[] برای ":" و Slice برای a:b:c
    # --------------------------------------------------------------------------


    @always_inline
    fn __getitem__(self, _e0: Tuple[], i1: Int, _e2: Tuple[]) -> Tensor[T]:
        var n0 = self._axis_len(0)
        var n2 = self._axis_len(2)
        var sels = List[IndexSel]()
        sels.append(make_slice_sel((0, n0, 1)))
        sels.append(make_index_sel(i1))
        sels.append(make_slice_sel((0, n2, 1)))
        return self.select(sels)
    @always_inline
    fn __getitem__(self, _e0: Tuple[], _e1: Tuple[], i2: Int) -> Tensor[T]:
        var n0 = self._axis_len(0)
        var n1 = self._axis_len(1)
        var sels = List[IndexSel]()
        sels.append(make_slice_sel((0, n0, 1)))
        sels.append(make_slice_sel((0, n1, 1)))
        sels.append(make_index_sel(i2))
        return self.select(sels)
    @always_inline
    fn __getitem__(self, i0: Int, _e1: Tuple[], i2: Int) -> Tensor[T]:
        var n1 = self._axis_len(1)
        var sels = List[IndexSel]()
        sels.append(make_index_sel(i0))
        sels.append(make_slice_sel((0, n1, 1)))
        sels.append(make_index_sel(i2))
        return self.select(sels)
    @always_inline
    fn __getitem__(self, _e0: Tuple[], _e1: Tuple[], s2: Slice) -> Tensor[T]:
        var n0 = self._axis_len(0)
        var n1 = self._axis_len(1)

        # normalize slice of axis-2
        var start2: Optional[Int]
        if s2.start is None: start2 = None else: start2 = Optional[Int](s2.start.value())
        var stop2: Optional[Int]
        if s2.end is None:   stop2 = None else:   stop2 = Optional[Int](s2.end.value())
        var step2: Optional[Int]
        if s2.step is None:  step2 = None else:  step2 = Optional[Int](s2.step.value())
        var trip2 = _normalize_slice_1d(self._axis_len(2), start2, stop2, step2)

        var sels = List[IndexSel]()
        sels.append(make_slice_sel((0, n0, 1)))
        sels.append(make_slice_sel((0, n1, 1)))
        sels.append(make_slice_sel(trip2))
        return self.select(sels)

    @always_inline
    fn __getitem__(self, i0: Int, i1: Int) -> Tensor[T]:
        # index on axes 0 and 1; keep axis 2 as-is (view)
        var sels = List[IndexSel]()
        sels.append(make_index_sel(i0))
        sels.append(make_index_sel(i1))
        return self.select(sels)

    # [..., i]  ≡ [:, :, i]
    @always_inline
    fn __getitem__(self, _e: EllipsisSpec, i2: Int) -> Tensor[T]:
        var n0 = self._axis_len(0)
        var n1 = self._axis_len(1)
        var sels = List[IndexSel]()
        sels.append(make_slice_sel((0, n0, 1)))
        sels.append(make_slice_sel((0, n1, 1)))
        sels.append(make_index_sel(i2))
        return self.select(sels)

    # [i, ...]  ≡ [i, :, :]
    @always_inline
    fn __getitem__(self, i0: Int, _e: EllipsisSpec) -> Tensor[T]:
        var n1 = self._axis_len(1)
        var n2 = self._axis_len(2)
        var sels = List[IndexSel]()
        sels.append(make_index_sel(i0))
        sels.append(make_slice_sel((0, n1, 1)))
        sels.append(make_slice_sel((0, n2, 1)))
        return self.select(sels)



    @always_inline
    fn __getitem__(self, i0: Int, i1: Int, i2: Int, i3: Int) -> Tensor[T]:
        var sels = List[IndexSel]()
        sels.append(make_index_sel(i0))
        sels.append(make_index_sel(i1))
        sels.append(make_index_sel(i2))
        sels.append(make_index_sel(i3))
        return self.select(sels)

    @always_inline
    fn __getitem__(self, i0: Int, _e1: Tuple[], _e2: Tuple[], _e3: Tuple[]) -> Tensor[T]:
        var n1 = self._axis_len(1); var n2 = self._axis_len(2); var n3 = self._axis_len(3)
        var sels = List[IndexSel]()
        sels.append(make_index_sel(i0))
        sels.append(make_slice_sel((0, n1, 1)))
        sels.append(make_slice_sel((0, n2, 1)))
        sels.append(make_slice_sel((0, n3, 1)))
        return self.select(sels)

    @always_inline
    fn __getitem__(self, _e0: Tuple[], i1: Int, _e2: Tuple[], _e3: Tuple[]) -> Tensor[T]:
        var n0 = self._axis_len(0); var n2 = self._axis_len(2); var n3 = self._axis_len(3)
        var sels = List[IndexSel]()
        sels.append(make_slice_sel((0, n0, 1)))
        sels.append(make_index_sel(i1))
        sels.append(make_slice_sel((0, n2, 1)))
        sels.append(make_slice_sel((0, n3, 1)))
        return self.select(sels)

    @always_inline
    fn __getitem__(self, _e0: Tuple[], _e1: Tuple[], i2: Int, _e3: Tuple[]) -> Tensor[T]:
        var n0 = self._axis_len(0); var n1 = self._axis_len(1); var n3 = self._axis_len(3)
        var sels = List[IndexSel]()
        sels.append(make_slice_sel((0, n0, 1)))
        sels.append(make_slice_sel((0, n1, 1)))
        sels.append(make_index_sel(i2))
        sels.append(make_slice_sel((0, n3, 1)))
        return self.select(sels)

    @always_inline
    fn __getitem__(self, _e0: Tuple[], _e1: Tuple[], _e2: Tuple[], i3: Int) -> Tensor[T]:
        var n0 = self._axis_len(0); var n1 = self._axis_len(1); var n2 = self._axis_len(2)
        var sels = List[IndexSel]()
        sels.append(make_slice_sel((0, n0, 1)))
        sels.append(make_slice_sel((0, n1, 1)))
        sels.append(make_slice_sel((0, n2, 1)))
        sels.append(make_index_sel(i3))
        return self.select(sels)

    @always_inline
    fn __getitem__(self, i0: Int, i1: Int, _e2: Tuple[], _e3: Tuple[]) -> Tensor[T]:
        var n2 = self._axis_len(2); var n3 = self._axis_len(3)
        var sels = List[IndexSel]()
        sels.append(make_index_sel(i0))
        sels.append(make_index_sel(i1))
        sels.append(make_slice_sel((0, n2, 1)))
        sels.append(make_slice_sel((0, n3, 1)))
        return self.select(sels)

    @always_inline
    fn __getitem__(self, i0: Int, _e1: Tuple[], i2: Int, _e3: Tuple[]) -> Tensor[T]:
        var n1 = self._axis_len(1); var n3 = self._axis_len(3)
        var sels = List[IndexSel]()
        sels.append(make_index_sel(i0))
        sels.append(make_slice_sel((0, n1, 1)))
        sels.append(make_index_sel(i2))
        sels.append(make_slice_sel((0, n3, 1)))
        return self.select(sels)

    @always_inline
    fn __getitem__(self, i0: Int, _e1: Tuple[], _e2: Tuple[], i3: Int) -> Tensor[T]:
        var n1 = self._axis_len(1); var n2 = self._axis_len(2)
        var sels = List[IndexSel]()
        sels.append(make_index_sel(i0))
        sels.append(make_slice_sel((0, n1, 1)))
        sels.append(make_slice_sel((0, n2, 1)))
        sels.append(make_index_sel(i3))
        return self.select(sels)

    @always_inline
    fn __getitem__(self, _e0: Tuple[], i1: Int, i2: Int, _e3: Tuple[]) -> Tensor[T]:
        var n0 = self._axis_len(0); var n3 = self._axis_len(3)
        var sels = List[IndexSel]()
        sels.append(make_slice_sel((0, n0, 1)))
        sels.append(make_index_sel(i1))
        sels.append(make_index_sel(i2))
        sels.append(make_slice_sel((0, n3, 1)))
        return self.select(sels)

    @always_inline
    fn __getitem__(self, _e0: Tuple[], i1: Int, _e2: Tuple[], i3: Int) -> Tensor[T]:
        var n0 = self._axis_len(0); var n2 = self._axis_len(2)
        var sels = List[IndexSel]()
        sels.append(make_slice_sel((0, n0, 1)))
        sels.append(make_index_sel(i1))
        sels.append(make_slice_sel((0, n2, 1)))
        sels.append(make_index_sel(i3))
        return self.select(sels)

    @always_inline
    fn __getitem__(self, _e0: Tuple[], _e1: Tuple[], i2: Int, i3: Int) -> Tensor[T]:
        var n0 = self._axis_len(0); var n1 = self._axis_len(1)
        var sels = List[IndexSel]()
        sels.append(make_slice_sel((0, n0, 1)))
        sels.append(make_slice_sel((0, n1, 1)))
        sels.append(make_index_sel(i2))
        sels.append(make_index_sel(i3))
        return self.select(sels)



    # [:, :, :, a:b:c]
    @always_inline
    fn __getitem__(self, _e0: Tuple[], _e1: Tuple[], _e2: Tuple[], s3: Slice) -> Tensor[T]:
        var n0 = self._axis_len(0); var n1 = self._axis_len(1); var n2 = self._axis_len(2)
        var trip3 = self._trip_from_slice(3, s3)
        var sels = List[IndexSel]()
        sels.append(make_slice_sel((0, n0, 1)))
        sels.append(make_slice_sel((0, n1, 1)))
        sels.append(make_slice_sel((0, n2, 1)))
        sels.append(make_slice_sel(trip3))
        return self.select(sels)

    # [:, :, a:b:c, :]
    @always_inline
    fn __getitem__(self, _e0: Tuple[], _e1: Tuple[], s2: Slice, _e3: Tuple[]) -> Tensor[T]:
        var n0 = self._axis_len(0); var n1 = self._axis_len(1); var n3 = self._axis_len(3)
        var trip2 = self._trip_from_slice(2, s2)
        var sels = List[IndexSel]()
        sels.append(make_slice_sel((0, n0, 1)))
        sels.append(make_slice_sel((0, n1, 1)))
        sels.append(make_slice_sel(trip2))
        sels.append(make_slice_sel((0, n3, 1)))
        return self.select(sels)

    # [:, 1:, 1:, :2]  → [E, S, S, S]
    @always_inline
    fn __getitem__(self, _e0: Tuple[], s1: Slice, s2: Slice, s3: Slice) -> Tensor[T]:
        var n0 = self._axis_len(0)
        var trip1 = self._trip_from_slice(1, s1)
        var trip2 = self._trip_from_slice(2, s2)
        var trip3 = self._trip_from_slice(3, s3)
        var sels = List[IndexSel]()
        sels.append(make_slice_sel((0, n0, 1)))
        sels.append(make_slice_sel(trip1))
        sels.append(make_slice_sel(trip2))
        sels.append(make_slice_sel(trip3))
        return self.select(sels)

    @always_inline
    fn __getitem__(self, i0: Int, i1: Int, i2: Int) -> Tensor[T]:
        # Index axes 0,1,2; keep axis-3 as is (view of length shape[3])
        var sels = List[IndexSel]()
        sels.append(make_index_sel(i0))
        sels.append(make_index_sel(i1))
        sels.append(make_index_sel(i2))
        return self.select(sels)


    # -----------------------------
    # Extend ":" (single Tuple[]) up to rank 5
    # -----------------------------
    @always_inline
    fn __getitem__(self, _e0: Tuple[]) -> Tensor[T]:
        var r = len(self._shape)
        var sels = List[IndexSel]()

        var n0 = self._axis_len(0)
        sels.append(make_slice_sel((0, n0, 1)))

        if r > 1:
            var n1 = self._axis_len(1)
            sels.append(make_slice_sel((0, n1, 1)))
        if r > 2:
            var n2 = self._axis_len(2)
            sels.append(make_slice_sel((0, n2, 1)))
        if r > 3:
            var n3 = self._axis_len(3)
            sels.append(make_slice_sel((0, n3, 1)))
        if r > 4:
            var n4 = self._axis_len(4)
            sels.append(make_slice_sel((0, n4, 1)))

        return self.select(sels)

    # --------------------------------------
    # Helper
    # --------------------------------------
    @always_inline
    fn self._trip_from_slice( axis: Int, s: Slice) -> (Int, Int, Int):
        var n = self._axis_len(axis)
        var start_opt: Optional[Int]
        if s.start is None: start_opt = None else: start_opt = Optional[Int](s.start.value())
        var stop_opt: Optional[Int]
        if s.end   is None: stop_opt  = None else: stop_opt  = Optional[Int](s.end.value())
        var step_opt: Optional[Int]
        if s.step  is None: step_opt  = None else: step_opt  = Optional[Int](s.step.value())
        var trip = _normalize_slice_1d(n, start_opt, stop_opt, step_opt)
        if s.start is None and s.end is None and s.step is None:
            trip = (0, n, 1)
        return trip

    # --------------------------------------
    # 5D-specific overloads needed by your demo
    # --------------------------------------

    # (A) Five indices: arr5[i0,i1,i2,i3,i4]  → scalar
    @always_inline
    fn __getitem__(self, i0: Int, i1: Int, i2: Int, i3: Int, i4: Int) -> Tensor[T]:
        var sels = List[IndexSel]()
        sels.append(make_index_sel(i0))
        sels.append(make_index_sel(i1))
        sels.append(make_index_sel(i2))
        sels.append(make_index_sel(i3))
        sels.append(make_index_sel(i4))
        return self.select(sels)

    # (B) [:, :, 1, :, :]  → [E, E, I, E, E]
    @always_inline
    fn __getitem__(self, _e0: Tuple[], _e1: Tuple[], i2: Int, _e3: Tuple[], _e4: Tuple[]) -> Tensor[T]:
        var n0 = self._axis_len(0)
        var n1 = self._axis_len(1)
        var n3 = self._axis_len(3)
        var n4 = self._axis_len(4)
        var sels = List[IndexSel]()
        sels.append(make_slice_sel((0, n0, 1)))
        sels.append(make_slice_sel((0, n1, 1)))
        sels.append(make_index_sel(i2))
        sels.append(make_slice_sel((0, n3, 1)))
        sels.append(make_slice_sel((0, n4, 1)))
        return self.select(sels)

    # (C) [::2, :, :, :, 0]  → [S, E, E, E, I]
    @always_inline
    fn __getitem__(self, s0: Slice, _e1: Tuple[], _e2: Tuple[], _e3: Tuple[], i4: Int) -> Tensor[T]:
        var trip0 = self._trip_from_slice(0, s0)
        var n1 = self._axis_len(1)
        var n2 = self._axis_len(2)
        var n3 = self._axis_len(3)
        var sels = List[IndexSel]()
        sels.append(make_slice_sel(trip0))
        sels.append(make_slice_sel((0, n1, 1)))
        sels.append(make_slice_sel((0, n2, 1)))
        sels.append(make_slice_sel((0, n3, 1)))
        sels.append(make_index_sel(i4))
        return self.select(sels)

    # (D) [:, 1, :, :, 1:3]  → [E, I, E, E, S]
    @always_inline
    fn __getitem__(self, _e0: Tuple[], i1: Int, _e2: Tuple[], _e3: Tuple[], s4: Slice) -> Tensor[T]:
        var n0 = self._axis_len(0)
        var n2 = self._axis_len(2)
        var n3 = self._axis_len(3)
        var trip4 = self._trip_from_slice(4, s4)
        var sels = List[IndexSel]()
        sels.append(make_slice_sel((0, n0, 1)))
        sels.append(make_index_sel(i1))
        sels.append(make_slice_sel((0, n2, 1)))
        sels.append(make_slice_sel((0, n3, 1)))
        sels.append(make_slice_sel(trip4))
        return self.select(sels)

    # (E) [:, :, :, ::-1, :]  → [E, E, E, S, E]
    @always_inline
    fn __getitem__(self, _e0: Tuple[], _e1: Tuple[], _e2: Tuple[], s3: Slice, _e4: Tuple[]) -> Tensor[T]:
        var n0 = self._axis_len(0)
        var n1 = self._axis_len(1)
        var n2 = self._axis_len(2)
        var n4 = self._axis_len(4)
        var trip3 = self._trip_from_slice(3, s3)   # e.g., ::-1 -> (n3-1, -1, -1)
        var sels = List[IndexSel]()
        sels.append(make_slice_sel((0, n0, 1)))
        sels.append(make_slice_sel((0, n1, 1)))
        sels.append(make_slice_sel((0, n2, 1)))
        sels.append(make_slice_sel(trip3))
        sels.append(make_slice_sel((0, n4, 1)))
        return self.select(sels)

    # =========================== __setitem__ (scalar) ===========================

    fn select_set_scalar(mut self, sels_in: List[IndexSel], value: T) -> None:
        var sels = self.pad_full_axes(sels_in)
        var d = len(self._shape)
        var all_index = True
        var i = 0
        while i < d:
            if not is_index(sels[i]):
                all_index = False
            i = i + 1
        if all_index:
            var lin = self._offset
            var ax = 0
            while ax < d:
                var ii = wrap_axis_index(get_index(sels[ax]), self._shape[ax])
                lin = lin + ii * self._strides[ax]
                ax = ax + 1
            if lin >= 0 and lin < len(self._data):
                self._data[lin] = value
            return
        var view_only = True
        var j = 0
        while j < d:
            if is_fancy(sels[j]):
                view_only = False
                break
            j = j + 1
        if view_only:
            var new_shape = List[Int]()
            var new_strides = List[Int]()
            var base = self._offset
            var ax2 = 0
            while ax2 < d:
                var dim = self._shape[ax2]
                if is_index(sels[ax2]):
                    var ii = wrap_axis_index(get_index(sels[ax2]), dim)
                    base = base + ii * self._strides[ax2]
                else:
                    var (st0, sp0, step0) = get_slice(sels[ax2])
                    var (ns, np, nst) = self._normalize_slice(dim, st0, sp0, step0)
                    var L = axis_len_from_slice(ns, np, nst)
                    new_shape.append(L)
                    new_strides.append(self._strides[ax2] * nst)
                    base = base + ns * self._strides[ax2]
                ax2 = ax2 + 1
            if len(new_shape) == 0:
                if base >= 0 and base < len(self._data):
                    self._data[base] = value
                return
            var out_rm = compute_row_major_strides(new_shape)
            var out_n = numel_shape(new_shape)
            var q = 0
            while q < out_n:
                var lin = base
                var rem = q
                var k = 0
                while k < len(new_shape):
                    var stride_q = out_rm[k]
                    var coord = rem // stride_q
                    if new_shape[k] > 0:
                        coord = coord % new_shape[k]
                    lin = lin + coord * new_strides[k]
                    k = k + 1
                if lin >= 0 and lin < len(self._data):
                    self._data[lin] = value
                q = q + 1
            return
        var keep_axes = List[Bool]()
        var maps = List[List[Int]]()
        var out_shape = List[Int]()
        var ax3 = 0
        while ax3 < d:
            var dim = self._shape[ax3]
            if is_index(sels[ax3]):
                keep_axes.append(False)
                var dummy = List[Int]()
                maps.append(dummy.copy())
            elif is_slice(sels[ax3]):
                var (st0, sp0, step0) = get_slice(sels[ax3])
                var (st, sp, step1) = self._normalize_slice(dim, st0, sp0, step0)
                var L = axis_len_from_slice(st, sp, step1)
                var idxs = List[Int]()
                idxs.reserve(L)
                var k = 0
                var cur = st
                while k < L:
                    idxs.append(cur)
                    cur = cur + step1
                    k = k + 1
                keep_axes.append(True)
                maps.append(idxs.copy())
                out_shape.append(L)
            else:
                var js_raw = get_fancy_list(sels[ax3])
                var js = List[Int]()
                js.reserve(len(js_raw))
                var p = 0
                while p < len(js_raw):
                    js.append(wrap_axis_index(js_raw[p], dim))
                    p = p + 1
                keep_axes.append(True)
                maps.append(js.copy())
                out_shape.append(len(js))
            ax3 = ax3 + 1
        var out_rm2 = compute_row_major_strides(out_shape)
        var out_n2 = numel_shape(out_shape)
        var q2 = 0
        while q2 < out_n2:
            var lin_dst = self._offset
            var rem2 = q2
            var kept_i = 0
            var a3 = 0
            while a3 < d:
                if keep_axes[a3]:
                    var stride_q = out_rm2[kept_i]
                    var coord = rem2 // stride_q
                    var len_ax = out_shape[kept_i]
                    if len_ax > 0:
                        coord = coord % len_ax
                    var src_index = maps[a3][coord]
                    lin_dst = lin_dst + src_index * self._strides[a3]
                    kept_i = kept_i + 1
                else:
                    var ii3 = wrap_axis_index(get_index(sels[a3]), self._shape[a3])
                    lin_dst = lin_dst + ii3 * self._strides[a3]
                a3 = a3 + 1
            if lin_dst >= 0 and lin_dst < len(self._data):
                self._data[lin_dst] = value
            q2 = q2 + 1
    # Tensor assign: src broadcast must match out shape; here we copy in order.
    fn select_set_tensor(mut self, sels_in: List[IndexSel], src: Tensor[T]) -> None:
        var dst = self.select(sels_in)
        var out_shape = dst.shape()
        var out_rm = compute_row_major_strides(out_shape)
        var out_n = numel_shape(out_shape)
        var n_copy = out_n if out_n <= len(src._data) else len(src._data)
        var sels = self.pad_full_axes(sels_in)
        var d = len(self._shape)
        var keep_axes = List[Bool]()
        var maps = List[List[Int]]()
        var ax = 0
        while ax < d:
            var dim = self._shape[ax]
            if is_index(sels[ax]):
                keep_axes.append(False)
                var dummy = List[Int]()
                maps.append(dummy.copy())
            elif is_slice(sels[ax]):
                var (st0, sp0, step0) = get_slice(sels[ax])
                var (st, sp, step1) = self._normalize_slice(dim, st0, sp0, step0)
                var L = axis_len_from_slice(st, sp, step1)
                var idxs = List[Int]()
                idxs.reserve(L)
                var k = 0
                var cur = st
                while k < L:
                    idxs.append(cur)
                    cur = cur + step1
                    k = k + 1
                keep_axes.append(True)
                maps.append(idxs.copy())
            else:
                var js_raw = get_fancy_list(sels[ax])
                var js = List[Int]()
                js.reserve(len(js_raw))
                var p = 0
                while p < len(js_raw):
                    js.append(wrap_axis_index(js_raw[p], dim))
                    p = p + 1
                keep_axes.append(True)
                maps.append(js.copy())
            ax = ax + 1
        var q = 0
        while q < n_copy:
            var lin_dst = self._offset
            var rem = q
            var kept_i = 0
            var a3 = 0
            while a3 < d:
                if keep_axes[a3]:
                    var stride_q = out_rm[kept_i]
                    var coord = rem // stride_q
                    var len_ax = out_shape[kept_i]
                    if len_ax > 0:
                        coord = coord % len_ax
                    var src_index = maps[a3][coord]
                    lin_dst = lin_dst + src_index * self._strides[a3]
                    kept_i = kept_i + 1
                else:
                    var ii3 = wrap_axis_index(get_index(sels[a3]), self._shape[a3])
                    lin_dst = lin_dst + ii3 * self._strides[a3]
                a3 = a3 + 1
            self._data[lin_dst] = src._data[q]
            q = q + 1

    # --- 1D index: scalar + tensor ---
    # -----------------------------------------------------------------------------
    # Scalar casting helpers: pick a Float64->T converter for current T
    # -----------------------------------------------------------------------------
    @always_inline
    fn _from_f64_for[T: ImplicitlyCopyable & Copyable & Movable]() -> fn (Float64) -> T:
        if T is Int:
            return to_int_from_f64
        elif T is Float32:
            return to_f32_from_f64
        else:
            return to_f64_from_f64      # default: Float64

    # Write a single scalar (provided as f64) into index i0
    @always_inline
    fn _setitem_scalar_f64(
        mut self, i0: Int, s64: Float64
    ) -> None:
        var sels = List[IndexSel]()
        sels.append(make_index_sel(i0))
        var v = self.select(sels)                 # VIEW
        var from_f64 = _from_f64_for[T]()
        _fill_view_scalar_T(v, from_f64(s64))     # write as T

    # -------------------------
    # -------------------------
    # 1) ساخت متادیتا از sels
    # -------------------------
    @always_inline
    fn _build_sel_meta(self, sels: List[IndexSel]) -> _SelViewMeta:

        var base = self._offset
        var out_shape = List[Int]()
        var out_coefs = List[Int]()

        var r = len(self._shape)
        var ax = 0
        while ax < r:
            var stride_ax = self._strides[ax]
            var sel = sels[ax]          # IndexSel: ImplicitlyCopyable → OK

            if sel.tag == 0:
                # index
                var i = sel.i
                if i < 0:
                    i += self._axis_len(ax)
                base += i * stride_ax

            else:
                # slice
                var s  = sel.start
                var e  = sel.stop
                var st = sel.step
                var ln = _trip_len(s, e, st)

                base += s * stride_ax
                out_shape.append(ln)
                out_coefs.append(st * stride_ax)

            ax += 1
        var meta = _SelViewMeta(base, out_shape, out_coefs)
        return meta.copy()


    @always_inline
    fn _assign_into_self_from_meta(
        mut self, meta: _SelViewMeta, rhs: Tensor[T]
    ) -> None:


        var rhs_total = _prod_shape(rhs._shape)

        if len(meta.out_shape) == 0:
            if rhs_total != 1:
                print("[ASG ERROR] scalar-dst but rhs_total=" + String(rhs_total))
                print("  dst.shape=[] rhs.shape=" + _list_str(rhs._shape))
                return
            var zeros = _zero_indices(len(rhs._shape))
            var j_rhs = _flat_index(rhs._offset, rhs._strides, zeros)
            self._data[meta.base_offset] = rhs._data[j_rhs]
            return

        if rhs_total == 1:
            var zeros2 = _zero_indices(len(rhs._shape))
            var j_rhs2 = _flat_index(rhs._offset, rhs._strides, zeros2)
            var v = rhs._data[j_rhs2]
            var idx = _zero_indices(len(meta.out_shape))
            var wrote = 0
            while True:
                var j_dst = _flat_index(meta.base_offset, meta.out_coefs, idx)
                self._data[j_dst] = v
                wrote += 1
                #if wrote <= 6:
                #    print("[ASG] fill j_dst=" + String(j_dst))
                if _bump_indices(idx, meta.out_shape) == False:
                    break
            return
        if _shapes_equal(meta.out_shape, rhs._shape) == False:
            print("[ASG ERROR] shape mismatch")
            print("  dst.shape=" + _list_str(meta.out_shape) + " rhs.shape=" + _list_str(rhs._shape))
            return

        var idx2 = _zero_indices(len(meta.out_shape))
        var copied = 0
        while True:
            var j_dst2 = _flat_index(meta.base_offset, meta.out_coefs, idx2)
            var j_rhs3 = _flat_index(rhs._offset, rhs._strides, idx2)
            self._data[j_dst2] = rhs._data[j_rhs3]
            copied += 1
            #if copied <= 6:
            #    print("[ASG] copy j_dst=" + String(j_dst2) + " <= j_rhs=" + String(j_rhs3))
            if _bump_indices(idx2, meta.out_shape) == False:
                break


    @always_inline
    fn _assign_with_sels_tensor(mut self, sels: List[IndexSel], rhs: Tensor[T]) -> None:
        var meta = self._build_sel_meta(sels)
        self._assign_into_self_from_meta(meta, rhs)


    @always_inline
    fn _trip_from_slice(self, axis: Int, s: Slice) -> (Int, Int, Int):
        var n = self._axis_len(axis)
        var start_opt: Optional[Int]
        if s.start is None: start_opt = None else: start_opt = Optional[Int](s.start.value())
        var stop_opt: Optional[Int]
        if s.end   is None: stop_opt  = None else: stop_opt  = Optional[Int](s.end.value())
        var step_opt: Optional[Int]
        if s.step  is None: step_opt  = None else: step_opt  = Optional[Int](s.step.value())
        var trip = _normalize_slice_1d(n, start_opt, stop_opt, step_opt)
        if s.start is None and s.end is None and s.step is None:
            trip = (0, n, 1)
        return trip




    @always_inline
    fn _parse_selector_string_lite(self, spec: String) -> List[IndexSel]:
        var r = len(self._shape)
        var sels = List[IndexSel]()

        var raw = String(spec.strip())
        if len(raw) == 0:
            var ax = 0
            while ax < r:
                sels.append(IndexSel.slice(0, self._axis_len(ax), 1))
                ax += 1
            return sels.copy()

        # split به String
        var parts_slice = raw.split(",")
        var parts = List[String]()
        var pi = 0
        while pi < len(parts_slice):
            parts.append(String(parts_slice[pi]))
            pi += 1

        # ellipsis
        var has_ellipsis = False
        var ellipsis_pos = -1
        var i = 0
        while i < len(parts):
            var ptrim = String(parts[i].strip())
            if ptrim == "...":
                has_ellipsis = True
                ellipsis_pos = i
                break
            i += 1

        var tokens = List[String]()
        if has_ellipsis:
            var left_count  = ellipsis_pos
            var right_count = len(parts) - ellipsis_pos - 1
            var missing = r - (left_count + right_count)
            if missing < 0:
                print("[PARSE ERROR] too many indices for rank=" + String(r))
                missing = 0

            var j = 0
            while j < left_count:
                tokens.append(String(parts[j].strip()))
                j += 1
            var k = 0
            while k < missing:
                tokens.append(String(":"))
                k += 1
            var m = ellipsis_pos + 1
            while m < len(parts):
                tokens.append(String(parts[m].strip()))
                m += 1
        else:
            var t = 0
            while t < len(parts):
                tokens.append(String(parts[t].strip()))
                t += 1

        # تراز با rank
        if len(tokens) < r:
            var pad = r - len(tokens)
            var p = 0
            while p < pad:
                tokens.append(String(":"))
                p += 1
        elif len(tokens) > r:
            var clipped = List[String]()
            var q = 0
            while q < r:
                clipped.append(tokens[q])
                q += 1
            tokens = clipped.copy()


        var ax2 = 0
        while ax2 < r:
            var tok = tokens[ax2]
            var n_ax = self._axis_len(ax2)

            # شمارش ':' با split
            var trio_slice = tok.split(":")
            var colon_cnt = len(trio_slice) - 1

            if colon_cnt > 0:
                # slice
                var trio = List[String]()
                var ti = 0
                while ti < len(trio_slice):
                    trio.append(String(trio_slice[ti]))
                    ti += 1

                var a_opt = Optional[Int]()
                var b_opt = Optional[Int]()
                var c_opt = Optional[Int]()
                if len(trio) >= 1: a_opt = _parse_opt_int(trio[0])
                if len(trio) >= 2: b_opt = _parse_opt_int(trio[1])
                if len(trio) >= 3: c_opt = _parse_opt_int(trio[2])

                var trip = _normalize_trip(n_ax, a_opt, b_opt, c_opt)
                var s = IndexSel.slice(trip[0], trip[1], trip[2])
                sels.append(s)
            else:
                var i_opt = _parse_opt_int(tok)   # "" -> None، عدد معتبر -> Some
                var i_val: Int
                if i_opt is None:
                    print("[PARSE ERROR] invalid integer token \"" + tok + "\"; using 0")
                    i_val = 0
                else:
                    i_val = i_opt.value()
                var s2 = IndexSel.index(i_val)
                sels.append(s2)

            ax2 += 1

        return sels.copy()


    # -----------------------------------------------------------------------------
        # __setitem__ overloads (single index, scalar RHS)
    # -----------------------------------------------------------------------------
    @always_inline
    fn _assign_after_dump(mut self, label: String, sels: List[IndexSel], rhs: Tensor[T]) -> None:
        var dbg = self.select(sels)
        # _dump_view("[VIEW dst]", dbg)
        # _dump_view("[VIEW rhs]", rhs)
        self._assign_with_sels_tensor(sels, rhs)


    @always_inline
    fn __setitem__(mut self, s: StrIndex, rhs: Tensor[T]) -> None:
        var sels = self._parse_selector_string_lite(s.spec)  # خروجی: List[IndexSel]
        self._assign_with_sels_tensor(sels, rhs)

    @always_inline
    fn __setitem__(mut self, lit: StringLiteral, rhs: Tensor[T]) -> None:
        self.__setitem__(StrIndex(String(lit)), rhs)

    @always_inline
    fn __setitem__(mut self, i0: Int, rhs: Tensor[T]) -> None:
        var sels = List[IndexSel]()
        sels.append(_ls_index(i0))
        self._assign_with_sels_tensor(sels, rhs)

    @always_inline
    fn __setitem__(mut self, s: Slice, rhs: Tensor[T]) -> None:
        var trip = self._trip_from_slice(0, s)
        var sels = List[IndexSel]()
        sels.append(_ls_slice(trip[0], trip[1], trip[2]))
        self._assign_with_sels_tensor(sels, rhs)

    @always_inline
    fn __setitem__(mut self, _e0: Tuple[], rhs: Tensor[T]) -> None:
        var r = len(self._shape)
        var sels = List[IndexSel]()
        if r >= 1: sels.append(_ls_slice(0, self._axis_len(0), 1))
        if r >= 2: sels.append(_ls_slice(0, self._axis_len(1), 1))
        if r >= 3: sels.append(_ls_slice(0, self._axis_len(2), 1))
        if r >= 4: sels.append(_ls_slice(0, self._axis_len(3), 1))
        if r >= 5: sels.append(_ls_slice(0, self._axis_len(4), 1))
        self._assign_with_sels_tensor(sels, rhs)


    # [:, :]
    @always_inline
    fn __setitem__(mut self, _e0: Tuple[], _e1: Tuple[], rhs: Tensor[T]) -> None:
        var n0 = self._axis_len(0); var n1 = self._axis_len(1)
        var sels = List[IndexSel]()
        sels.append(make_slice_sel((0, n0, 1)))
        sels.append(make_slice_sel((0, n1, 1)))
        self._assign_with_sels_tensor(sels, rhs)

    # [i, :]
    @always_inline
    fn __setitem__(mut self, i0: Int, _e1: Tuple[], rhs: Tensor[T]) -> None:
        var n1 = self._axis_len(1)
        var sels = List[IndexSel]()
        sels.append(make_index_sel(i0))
        sels.append(make_slice_sel((0, n1, 1)))
        self._assign_with_sels_tensor(sels, rhs)

    @always_inline
    fn __setitem__(mut self, _e0: Tuple[], i1: Int, rhs: Tensor[T]) -> None:
        var n0 = self._axis_len(0)
        var sels = List[IndexSel]()
        sels.append(make_slice_sel((0, n0, 1)))
        sels.append(make_index_sel(i1))
        self._assign_after_dump("2D [:,i]", sels, rhs)

    # [i, a:b:c]
    @always_inline
    fn __setitem__(mut self, i0: Int, s1: Slice, rhs: Tensor[T]) -> None:
        var trip1 = self._trip_from_slice( 1, s1)
        var sels = List[IndexSel]()
        sels.append(make_index_sel(i0))
        sels.append(make_slice_sel(trip1))
        self._assign_with_sels_tensor(sels, rhs)

    # [a:b:c, j]
    @always_inline
    fn __setitem__(mut self, s0: Slice, i1: Int, rhs: Tensor[T]) -> None:
        var trip0 = self._trip_from_slice( 0, s0)
        var sels = List[IndexSel]()
        sels.append(make_slice_sel(trip0))
        sels.append(make_index_sel(i1))
        self._assign_with_sels_tensor(sels, rhs)

    # [a:b:c, d:e:f]
    @always_inline
    fn __setitem__(mut self, s0: Slice, s1: Slice, rhs: Tensor[T]) -> None:
        var t0 = self._trip_from_slice(0, s0)
        var t1 = self._trip_from_slice(1, s1)
        var sels = List[IndexSel]()
        sels.append(_ls_slice(t0[0], t0[1], t0[2]))
        sels.append(_ls_slice(t1[0], t1[1], t1[2]))
        self._assign_with_sels_tensor(sels, rhs)

    # [i, j, :]
    @always_inline
    fn __setitem__(mut self, i0: Int, i1: Int, rhs: Tensor[T]) -> None:
        var sels = List[IndexSel]()
        sels.append(make_index_sel(i0))
        sels.append(make_index_sel(i1))
        self._assign_with_sels_tensor(sels, rhs)

    # [:, j, :]
    @always_inline
    fn __setitem__(mut self, _e0: Tuple[], i1: Int, _e2: Tuple[], rhs: Tensor[T]) -> None:
        var n0 = self._axis_len(0); var n2 = self._axis_len(2)
        var sels = List[IndexSel]()
        sels.append(make_slice_sel((0, n0, 1)))
        sels.append(make_index_sel(i1))
        sels.append(make_slice_sel((0, n2, 1)))
        self._assign_with_sels_tensor(sels, rhs)

    # [:, :, k]
    @always_inline
    fn __setitem__(mut self, _e0: Tuple[], _e1: Tuple[], i2: Int, rhs: Tensor[T]) -> None:
        var n0 = self._axis_len(0); var n1 = self._axis_len(1)
        var sels = List[IndexSel]()
        sels.append(make_slice_sel((0, n0, 1)))
        sels.append(make_slice_sel((0, n1, 1)))
        sels.append(make_index_sel(i2))
        self._assign_after_dump("3D [:,:,i]", sels, rhs)

    # [i, :, k]
    @always_inline
    fn __setitem__(mut self, i0: Int, _e1: Tuple[], i2: Int, rhs: Tensor[T]) -> None:
        var n1 = self._axis_len(1)
        var sels = List[IndexSel]()
        sels.append(make_index_sel(i0))
        sels.append(make_slice_sel((0, n1, 1)))
        sels.append(make_index_sel(i2))
        self._assign_with_sels_tensor(sels, rhs)

    # [:, :, a:b:c]
    @always_inline
    fn __setitem__(mut self, _e0: Tuple[], _e1: Tuple[], s2: Slice, rhs: Tensor[T]) -> None:
        var n0 = self._axis_len(0); var n1 = self._axis_len(1)
        var trip2 = self._trip_from_slice( 2, s2)
        var sels = List[IndexSel]()
        sels.append(make_slice_sel((0, n0, 1)))
        sels.append(make_slice_sel((0, n1, 1)))
        sels.append(make_slice_sel(trip2))
        self._assign_with_sels_tensor(sels, rhs)


    # [i, j, k, l]
    @always_inline
    fn __setitem__(mut self, i0: Int, i1: Int, i2: Int, i3: Int, rhs: Tensor[T]) -> None:
        var sels = List[IndexSel]()
        sels.append(make_index_sel(i0))
        sels.append(make_index_sel(i1))
        sels.append(make_index_sel(i2))
        sels.append(make_index_sel(i3))
        self._assign_with_sels_tensor(sels, rhs)

    # [:, :, :, k]
    @always_inline
    fn __setitem__(mut self, _e0: Tuple[], _e1: Tuple[], _e2: Tuple[], i3: Int, rhs: Tensor[T]) -> None:
        var n0 = self._axis_len(0); var n1 = self._axis_len(1); var n2 = self._axis_len(2)
        var sels = List[IndexSel]()
        sels.append(make_slice_sel((0, n0, 1)))
        sels.append(make_slice_sel((0, n1, 1)))
        sels.append(make_slice_sel((0, n2, 1)))
        sels.append(make_index_sel(i3))
        self._assign_after_dump("4D [:,:,:,i]", sels, rhs)

    # [:, :, i, :]
    @always_inline
    fn __setitem__(mut self, _e0: Tuple[], _e1: Tuple[], i2: Int, _e3: Tuple[], rhs: Tensor[T]) -> None:
        var n0 = self._axis_len(0); var n1 = self._axis_len(1); var n3 = self._axis_len(3)
        var sels = List[IndexSel]()
        sels.append(make_slice_sel((0, n0, 1)))
        sels.append(make_slice_sel((0, n1, 1)))
        sels.append(make_index_sel(i2))
        sels.append(make_slice_sel((0, n3, 1)))
        self._assign_with_sels_tensor(sels, rhs)

    # [:, 1:, 1:, :2]  → [E, S, S, S]
    @always_inline
    fn __setitem__(mut self, _e0: Tuple[], s1: Slice, s2: Slice, s3: Slice, rhs: Tensor[T]) -> None:
        var n0 = self._axis_len(0)
        var trip1 = self._trip_from_slice( 1, s1)
        var trip2 = self._trip_from_slice( 2, s2)
        var trip3 = self._trip_from_slice( 3, s3)
        var sels = List[IndexSel]()
        sels.append(make_slice_sel((0, n0, 1)))
        sels.append(make_slice_sel(trip1))
        sels.append(make_slice_sel(trip2))
        sels.append(make_slice_sel(trip3))
        self._assign_with_sels_tensor(sels, rhs)


    # (A) [i0,i1,i2,i3,i4]
    @always_inline
    fn __setitem__(mut self, i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, rhs: Tensor[T]) -> None:
        var sels = List[IndexSel]()
        sels.append(make_index_sel(i0)); sels.append(make_index_sel(i1))
        sels.append(make_index_sel(i2)); sels.append(make_index_sel(i3))
        sels.append(make_index_sel(i4))
        self._assign_with_sels_tensor(sels, rhs)

    # (B) [:, :, 1, :, :]
    @always_inline
    fn __setitem__(mut self, _e0: Tuple[], _e1: Tuple[], i2: Int, _e3: Tuple[], _e4: Tuple[], rhs: Tensor[T]) -> None:
        var n0 = self._axis_len(0); var n1 = self._axis_len(1); var n3 = self._axis_len(3); var n4 = self._axis_len(4)
        var sels = List[IndexSel]()
        sels.append(make_slice_sel((0, n0, 1)))
        sels.append(make_slice_sel((0, n1, 1)))
        sels.append(make_index_sel(i2))
        sels.append(make_slice_sel((0, n3, 1)))
        sels.append(make_slice_sel((0, n4, 1)))
        self._assign_with_sels_tensor(sels, rhs)

    # (C) [::2, :, :, :, 0]
    @always_inline
    fn __setitem__(mut self, s0: Slice, _e1: Tuple[], _e2: Tuple[], _e3: Tuple[], i4: Int, rhs: Tensor[T]) -> None:
        var trip0 = self._trip_from_slice( 0, s0)
        var n1 = self._axis_len(1); var n2 = self._axis_len(2); var n3 = self._axis_len(3)
        var sels = List[IndexSel]()
        sels.append(make_slice_sel(trip0))
        sels.append(make_slice_sel((0, n1, 1)))
        sels.append(make_slice_sel((0, n2, 1)))
        sels.append(make_slice_sel((0, n3, 1)))
        sels.append(make_index_sel(i4))
        self._assign_with_sels_tensor(sels, rhs)

    # (D) [:, 1, :, :, 1:3]
    @always_inline
    fn __setitem__(mut self, _e0: Tuple[], i1: Int, _e2: Tuple[], _e3: Tuple[], s4: Slice, rhs: Tensor[T]) -> None:
        var n0 = self._axis_len(0); var n2 = self._axis_len(2); var n3 = self._axis_len(3)
        var trip4 = self._trip_from_slice( 4, s4)
        var sels = List[IndexSel]()
        sels.append(make_slice_sel((0, n0, 1)))
        sels.append(make_index_sel(i1))
        sels.append(make_slice_sel((0, n2, 1)))
        sels.append(make_slice_sel((0, n3, 1)))
        sels.append(make_slice_sel(trip4))
        self._assign_with_sels_tensor(sels, rhs)

    # (E) [:, :, :, ::-1, :]
    @always_inline
    fn __setitem__(mut self, _e0: Tuple[], _e1: Tuple[], _e2: Tuple[], s3: Slice, _e4: Tuple[], rhs: Tensor[T]) -> None:
        var n0 = self._axis_len(0); var n1 = self._axis_len(1); var n2 = self._axis_len(2); var n4 = self._axis_len(4)
        var trip3 = self._trip_from_slice( 3, s3)
        var sels = List[IndexSel]()
        sels.append(make_slice_sel((0, n0, 1)))
        sels.append(make_slice_sel((0, n1, 1)))
        sels.append(make_slice_sel((0, n2, 1)))
        sels.append(make_slice_sel(trip3))
        sels.append(make_slice_sel((0, n4, 1)))
        self._assign_with_sels_tensor(sels, rhs)



    # Central dispatcher
    @always_inline
    fn _assign_sels_scalar(mut self: Tensor[T], sels: List[IndexSel], s: T) -> None:
        var view = self.select(sels)   # view over self
        _fill_view(view, s)

    @always_inline
    fn _assign_sels_tensor(mut self: Tensor[T], sels: List[IndexSel], rhs: Tensor[T]) -> None:
        var view = self.select(sels)   # view over self
        _copy_view_broadcast(view, rhs)

    # Convenience: normalized 1D slice for a given axis
    @always_inline
    fn _trip_from_slice_axis(self: Tensor[T], axis: Int, s: Slice) -> (Int, Int, Int):
        var n = self._axis_len(axis)
        var start_opt: Optional[Int]
        if s.start is None: start_opt = None else: start_opt = Optional[Int](s.start.value())
        var stop_opt: Optional[Int]
        if s.end   is None: stop_opt  = None else: stop_opt  = Optional[Int](s.end.value())
        var step_opt: Optional[Int]
        if s.step  is None: step_opt  = None else: step_opt  = Optional[Int](s.step.value())
        var trip = _normalize_slice_1d(n, start_opt, stop_opt, step_opt)
        if s.start is None and s.end is None and s.step is None:
            trip = (0, n, 1)
        return trip
     # ---------------- introspection ----------------
    fn shape(self) -> List[Int]:
        return self._shape.copy()

    fn strides(self) -> List[Int]:
        return self._strides.copy()

    fn offset(self) -> Int:
        return self._offset

    fn rank(self) -> Int:
        return len(self._shape)

    fn size(self) -> Int:
        return numel_shape(self._shape)


    fn len(self) -> Int:
        return numel(self._shape)

    # rank (number of dimensions)
    fn ndim(self) -> Int:
        return len(self._shape)


    fn rank(self) -> Int:
        return len(self._shape)

    fn size(self) -> Int:
        return numel(self._shape)

    @always_inline
    fn idx(self, i: Int) -> IndexSel:
        return make_index_sel(i)

    @always_inline
    fn rng(self, a: Int, b: Int, step: Int = 1) -> IndexSel:
        return make_slice_sel((a, b, step))

    @always_inline
    fn reange(self, a: Int, b: Int) -> IndexSel:
        return make_slice_sel((a, b, 1))

    @always_inline
    fn clone(self) -> Tensor[T]:
        return self.copy()

    # --------------------------- slice builders ------------------------
    fn full(self, axis: Int) -> SliceSpec:
        var dim = self._shape[normalize_axis(axis, len(self._shape))]
        return SliceSpec(start=0, stop=dim, step=1)

    fn unsqueeze(self,axis_in: Int) -> Tensor[T]:
        return unsqueeze(self, axis_in)

    fn squeeze_all(self) -> Tensor[T]:
        return squeeze_all(self)


    fn squeeze(self,axis_in: Int) -> Tensor[T]:
        return squeeze_axis(self, axis_in)

    fn permute(self, axes: List[Int]) -> Tensor[T]:
        return permute(self, axes)

    fn repeat(self, reps: List[Int]) -> Tensor[T]:
        return repeat(self, reps)

    fn transpose(self, i: Int, j: Int) -> Tensor[T]: return transpose(self, i, j)
    fn transpose(self, perm: List[Int]) -> Tensor[T]:  return transpose(self, perm)
    fn transpose(self) -> Tensor[T]:  return transpose(self)

    fn flatten(self, start_dim: Int = 0,    end_dim_opt: Optional[Int] = None) -> Tensor[T]:
        return flatten(self,start_dim,    end_dim_opt)

    fn narrow(self, axis: Int, start: Int, length: Int) -> SliceSpec:
        var d = len(self._shape)
        var ax = normalize_axis(axis, d)
        var dim = self._shape[ax]
        var s = wrap_index(start, dim)
        if s < 0:
            s = 0
        var e = s + length
        if e > dim:
            e = dim
        return SliceSpec(start=s, stop=e, step=1)

    # ------------------------- multi-axis slice ------------------------


    fn slice(self, specs: List[SliceSpec]) -> Tensor[T]:
        var d = len(self._shape)

        var out_shape = List[Int]()
        out_shape.reserve(d)
        var sizes = List[Int]()
        sizes.reserve(d)

        var ax = 0
        while ax < d:
            var (s0, s1, st) = specs[ax]
            var n_ax = range_len(s0, s1, st)
            sizes.append(n_ax)
            out_shape.append(n_ax)
            ax = ax + 1

        var out_n = 1
        var i = 0
        while i < d:
            out_n = out_n * sizes[i]
            i = i + 1

        var out_flat = List[T]()
        out_flat.reserve(out_n)
        if out_n == 0:
            return Tensor[T](out_shape, out_flat)

        var counter = List[Int]()
        counter.reserve(d)
        i = 0
        while i < d:
            counter.append(0)
            i = i + 1

        var done = False
        while not done:
            var lin = 0
            ax = 0
            while ax < d:
                var (s0, _s1, st) = specs[ax]
                var j = s0 + counter[ax] * st
                lin = lin + j * self._strides[ax]
                ax = ax + 1
            out_flat.append(self._data[lin])

            ax = d - 1
            while ax >= 0:
                counter[ax] = counter[ax] + 1
                if counter[ax] < sizes[ax]:
                    break
                counter[ax] = 0
                ax = ax - 1
            if ax < 0:
                done = True

        return Tensor[T](out_shape, out_flat)

    @always_inline
    fn slice(
        self,
        ax: Int,
        start: Int,
        stop: Int,
        step: Int = 1,
        debug: Bool = True
    ) -> Tensor[T]:


        var r = len(self._shape)
        if r == 0:
            print("[slice] rank-0 tensor → return self")
            return self.copy()
        if ax < 0 or ax >= r:
            print("[slice] axis out of range → return self")
            return self.copy()
        if step == 0:
            print("[slice] step == 0 → return self")
            return self.copy()

        var n_ax = self._shape[ax]

        # Normalize like Python
        var s = start
        if start < 0: s =start + n_ax
        var e =stop
        if stop  < 0: e =stop  + n_ax

        # Clamp
        if s < 0: s = 0
        if e < 0: e = 0
        if s > n_ax: s = n_ax
        if e > n_ax: e = n_ax

        # Length on this axis
        var len_ax = range_len(s, e, step)

        var out_shape = self._shape.copy()
        var out_strides = self._strides.copy()
        out_shape[ax]   = len_ax
        out_strides[ax] = self._strides[ax] * step
        var out_offset  = self._offset + s * self._strides[ax]



        # Constructor order MUST be (data, shape, strides, offset)

        return Tensor[T](self._data, out_shape, out_strides, out_offset)

    fn slice_inplace(
        mut self,
        ax: Int,
        start: Int,
        stop: Int,
        step: Int = 1,
    ) -> Tensor[T]:

        # Basic validation
        var r = len(self._shape)
        if r == 0:
            print("[slice_inplace] rank-0 tensor → no-op")
            return self.copy()
        if ax < 0 or ax >= r:
            print("[slice_inplace] ERROR: axis out of range → no-op")
            return self.copy()
        if step == 0:
            print("[slice_inplace] ERROR: step == 0 → no-op")
            return self.copy()

        var n_ax = self._shape[ax]

        # Normalize indices like Python (relative to axis length)
        var s = start
        if s < 0: s = s + n_ax
        var e = stop
        if e < 0: e = e + n_ax

        # Clamp depending on step sign (Python-ish)
        if step > 0:
            if s < 0: s = 0
            if s > n_ax: s = n_ax
            if e < 0: e = 0
            if e > n_ax: e = n_ax
        else:
            # step < 0
            # For negative step, valid index range is [-1 .. n_ax-1] conceptually,
            # but we'll clamp to actual storage-safe values.
            if s < -1: s = -1
            if s >= n_ax: s = n_ax - 1
            if e < -1: e = -1
            if e >= n_ax: e = n_ax - 1

        # Compute logical length along this axis
        var len_ax = range_len(s, e, step)

        # Compute new stride/offset for this axis
        var new_stride_ax = self._strides[ax] * step

        # Base index for offset shift:
        # When length == 0, we don't want an out-of-bounds base; pick 0 safely.
        var base_idx = s
        if len_ax == 0:
            base_idx = 0
        else:
            # if base index is within [0 .. n_ax-1] to keep offset sane
            if base_idx < 0: base_idx = 0
            if base_idx >= n_ax and n_ax > 0: base_idx = n_ax - 1

        var new_offset = self._offset + base_idx * self._strides[ax]


        # Apply in-place: update only the selected axis
        self._shape[ax]   = len_ax
        self._strides[ax] = new_stride_ax
        self._offset      = new_offset


        return self.copy()

    fn index_axis(self, axis: Int, index: Int) -> Tensor[T]:
        var d = len(self._shape)
        var ax = normalize_axis(axis, d)
        var dim = self._shape[ax]
        var idx0 = wrap_index(index, dim)

        var specs = List[SliceSpec]()
        specs.reserve(d)
        var i = 0
        while i < d:
            if i == ax:
                specs.append(make_slice(idx0, idx0 + 1, 1))
            else:
                specs.append(make_slice(0, self._shape[i], 1))
            i = i + 1

        var tmp = self.slice(specs)
        return tmp.squeeze_axis(ax)

    fn squeeze_axis(self, axis: Int) -> Tensor[T]:
        var d = len(self._shape)
        var ax = normalize_axis(axis, d)
        var out_shape = List[Int]()
        out_shape.reserve(d - 1)
        var i = 0
        while i < d:
            if i != ax:
                out_shape.append(self._shape[i])
            i = i + 1
        return Tensor[T](out_shape, self._data)

    @always_inline
    fn as_strided(
        self,
        new_shape: List[Int],
        new_strides: List[Int],
        offset: Int = 0
    ) -> Tensor[T]:
        # Validate ranks
        var ok = True
        if len(new_shape) != len(new_strides):
            print("as_strided_safe: shape/strides rank mismatch")
            ok = False

        # Validate offset
        if offset < 0:
            print("as_strided_safe: negative offset")
            ok = False

        # Validate dims and compute out_n
        var out_n = 1
        var i = 0
        while i < len(new_shape):
            var d = new_shape[i]
            if d < 0:
                print("as_strided_safe: negative dimension at axis " + i.__str__())
                ok = False
            out_n = out_n * d
            i += 1

        # Early fallback if anything is wrong so far
        if not ok:
            return self.copy()

        # Zero-size fast path (no bounds walk needed)
        if out_n == 0:
            return Tensor[T](self._data, new_shape, new_strides, offset)

        # Bounds check for reachable indices
        var min_idx = offset
        var max_idx = offset
        i = 0
        while i < len(new_shape):
            var d = new_shape[i]
            var st = new_strides[i]
            if d > 1:
                var last = (d - 1) * st
                if last < 0:
                    min_idx = min_idx + last
                else:
                    max_idx = max_idx + last
            i += 1

        if min_idx < 0 or max_idx >= len(self._data):
            print("as_strided_safe: out-of-bounds view (min=" + min_idx.__str__() +
                ", max=" + max_idx.__str__() + ", data_len=" + len(self._data).__str__() + ")")
            return self.copy()

        # Pure view (shared storage)
        return Tensor[T](self._data, new_shape, new_strides, offset)


    fn get2(self, i: Int, j: Int ) -> T:
        var r = len(self._shape)
        if r != 2:

        var n0 = self._shape[0]; var n1 = self._shape[1]
        var ii =i
        if i < 0: ii =i + n0
        var jj =j
        if j < 0: jj =j + n1
        if ii < 0 or ii >= n0 or jj < 0 or jj >= n1:

            return self._data[self._offset]   # fallback

        var lin = self._offset + ii * self._strides[0] + jj * self._strides[1]

        return self._data[lin]

    fn set2(
        mut self,
        i: Int,
        j: Int,
        value: T,
    ) -> None:
        var r = len(self._shape)

        if r != 2:
            return

        var n0 = self._shape[0]
        var n1 = self._shape[1]

        # Normalize negative indices
        var ii =i
        if i < 0: ii =i + n0
        var jj =j
        if j < 0: jj =j + n1


        # Bounds check
        if ii < 0 or ii >= n0 or jj < 0 or jj >= n1:
            return

        # Linear position using strides + offset (supports views)
        var lin = self._offset + ii * self._strides[0] + jj * self._strides[1]

        # Guard read (lin checked by OOB above for logical indices; still sanity-print)
        if lin < 0 or lin >= len(self._data):
            print("[set2] ERROR: storage index OOB: lin=" + lin.__str__() +
                    " data_len=" + len(self._data).__str__())
            return

        self._data[lin] = value

    fn put(self,index: Tensor[Int], src: Tensor[T]) -> Tensor[T]:
        return put(self,index, src)
    # ---------------- string-based multi-axis selector ----------------
    @always_inline
    fn _trim(self, s: String) -> String:
        var n = len(s)
        var i = 0
        var j = n - 1

        # skip leading spaces/tabs
        while i < n and (s[i] == ' ' or s[i] == '\t'):
            i = i + 1
        # skip trailing spaces/tabs
        while j >= i and (s[j] == ' ' or s[j] == '\t'):
            j = j - 1

        if j < i:
            return String("")

        # build substring manually (String.slice(...) not available)
        var out = String("")
        var p = i
        while p <= j:
            out = out + String(s[p])
            p = p + 1
        return out.copy()

    @always_inline
    fn _to_int_safe(self, s: String, defv: Int) -> Int:
        var n = len(s)
        if n == 0:
            return defv

        var sign = 1
        var k = 0
        if s[0] == '-':
            sign = -1
            k = 1

        var v = 0
        while k < n:
            var c = s[k]
            var d = -1
            if c == '0':
                d = 0
            elif c == '1':
                d = 1
            elif c == '2':
                d = 2
            elif c == '3':
                d = 3
            elif c == '4':
                d = 4
            elif c == '5':
                d = 5
            elif c == '6':
                d = 6
            elif c == '7':
                d = 7
            elif c == '8':
                d = 8
            elif c == '9':
                d = 9
            else:
                return defv

            v = v * 10 + d
            k = k + 1

        return sign * v


    fn _parse_axis_token(self, tok: String, axis: Int) -> IndexSel:
        var t = self._trim(tok)
        var n = len(t)
        var has_colon = False
        var i = 0
        while i < n:
            if t[i] == ':':
                has_colon = True
                break
            i = i + 1

        if not has_colon:
            var k = self._to_int_safe(t, 0)
            return make_index_sel(k)  # <-- was IndexSel.of_index(k)

        var a = String("")
        var b = String("")
        var c = String("")
        var part = 0
        var cur = String("")
        i = 0
        while i < n:
            var ch = t[i]
            if ch == ':':
                if part == 0:
                    a = cur
                elif part == 1:
                    b = cur
                cur = String("")
                part = part + 1
            else:
                cur = cur + String(ch)
            i = i + 1
        if part == 0:
            a = cur
        elif part == 1:
            b = cur
        else:
            c = cur

        var dim = self._shape[axis]
        var start = 0
        var stop = dim
        var step = 1

        if len(a) != 0:
            start = self._to_int_safe(a, 0)
        if len(b) != 0:
            stop = self._to_int_safe(b, dim)
        if len(c) != 0:
            step = self._to_int_safe(c, 1)

        return make_slice_sel((start, stop, step))  # unchanged (tuple-based SliceSpec)






    # ------------------------------- Introspection -----------------------------






    fn matmul(self: Tensor[Float64], x: Tensor[Float64]) -> Tensor[Float64]:
        return _matmul_free(self, x)

    # Tensor[Int] @ Tensor[Int] -> Tensor[Float64]  (upcast then call Float64 impl)
    fn matmul(self: Tensor[Int], x: Tensor[Int]) -> Tensor[Float64]:
        var Af = astype_f64_from_int(self)
        var xf = astype_f64_from_int(x)
        return _matmul_free(Af, xf)
    fn matmul(self: Tensor[Float32], x: Tensor[Float32]) -> Tensor[Float64]:
        var Af = astype_f64_from_f32(self)
        var xf = astype_f64_from_f32(x)
        return _matmul_free(Af, xf)


    fn matmul_vec(self: Tensor[Float64], x: Tensor[Float64]) -> Tensor[Float64]:
        return matmul_core_vec(self, x)

    # Tensor[Int] @ Tensor[Int] -> Tensor[Float64]  (upcast then call Float64 impl)
    fn matmul_vec(self: Tensor[Int], x: Tensor[Int]) -> Tensor[Float64]:
        var Af = astype_f64_from_int(self)
        var xf = astype_f64_from_int(x)
        return matmul_core_vec(Af, xf)
    fn matmul_vec(self: Tensor[Float32], x: Tensor[Float32]) -> Tensor[Float64]:
        var Af = astype_f64_from_f32(self)
        var xf = astype_f64_from_f32(x)
        return matmul_core_vec(Af, xf)

    # Float64 x Float64
    fn tensordot(self: Tensor[Float64], B: Tensor[Float64], axis: Int = 1) -> Tensor[Float64]:
        return _tensordot_free(self, B, axis)


    fn tensordot(self: Tensor[Int], B: Tensor[Int], axis: Int = 1) -> Tensor[Float64]:
        var Af = astype_f64_from_int(self)
        var Bf = astype_f64_from_int(B)
        return _tensordot_free(Af, Bf, axis)


    fn sum(self: Tensor[Float64], axis: Optional[Int] = None, keepdims: Bool = False) -> Tensor[Float64]:
        return sum(self, axis, keepdims)
    fn sum(self: Tensor[Float32], axis: Optional[Int] = None, keepdims: Bool = False) -> Tensor[Float32]:
        return sum(self, axis, keepdims)
    fn sum(self: Tensor[Int], axis: Optional[Int], keepdims: Bool = False) -> Tensor[Int]:
        var f = astype_f64_from_int(self)
        return sum(self, axis, keepdims)
    @always_inline
    fn sum_all(self: Tensor[Int]) -> Int:
        return sum1d_unrolled(self)

    @always_inline
    fn sum_all(self: Tensor[Float64]) -> Float64:
        return sum1d_unrolled(self)

    @always_inline
    fn sum_all(self: Tensor[Float32]) -> Float32:
        return sum1d_unrolled(self)


    fn std(self: Tensor[Float64], axis: Optional[Int] = None, keepdims: Bool = False) -> Tensor[Float64]:
        return std(self, axis, keepdims)
    fn std(self: Tensor[Float32], axis: Optional[Int] = None, keepdims: Bool = False) -> Tensor[Float64]:
        return std(self, axis, keepdims)
    fn std(self: Tensor[Int], axis: Optional[Int], keepdims: Bool = False) -> Tensor[Float64]:
        var f = astype_f64_from_int(self)
        return std(f, axis, keepdims)
    @always_inline
    fn std_all(self: Tensor[Int]) -> Int:
        return std1d_unrolled(self)

    @always_inline
    fn std_all(self: Tensor[Float64]) -> Float64:
        return std1d_unrolled(self)

    @always_inline
    fn std_all(self: Tensor[Float32]) -> Float32:
        return std1d_unrolled(self)

    fn nanmean(self:Tensor[Float64], axis: Optional[Int] = None, keepdims: Bool = False) -> Tensor[Float64]:
        return nanmean(self, axis, keepdims)

    fn nansum(self:Tensor[Float64], axis: Optional[Int] = None, keepdims: Bool = False) -> Tensor[Float64]:
        return nansum(self, axis, keepdims)

    fn nanmin(self:Tensor[Float64], axis: Optional[Int] = None, keepdims: Bool = False) -> Tensor[Float64]:
        return nansum(self, axis, keepdims)

    fn nanmean(self:Tensor[Float32], axis: Optional[Int] = None, keepdims: Bool = False) -> Tensor[Float64]:
        return nanmean(self, axis, keepdims)

    fn nansum(self:Tensor[Float32], axis: Optional[Int] = None, keepdims: Bool = False) -> Tensor[Float64]:
        return nansum(self, axis, keepdims)

    fn nanmin(self:Tensor[Float32], axis: Optional[Int] = None, keepdims: Bool = False) -> Tensor[Float64]:
        return nansum(self, axis, keepdims)


    fn nanmean(self:Tensor[Int], axis: Optional[Int] = None, keepdims: Bool = False) -> Tensor[Float64]:
        return nanmean(self, axis, keepdims)

    fn nansum(self:Tensor[Int], axis: Optional[Int] = None, keepdims: Bool = False) -> Tensor[Float64]:
        return nansum(self, axis, keepdims)

    fn nanmin(self:Tensor[Int], axis: Optional[Int] = None, keepdims: Bool = False) -> Tensor[Float64]:
        return nansum(self, axis, keepdims)



    # ---------- Float64 overloads ----------

    fn mean(self: Tensor[Float64]) -> Tensor[Float64]:
        var none = Optional[Int]()
        return _mean_free(self, none, False)

    fn mean(self: Tensor[Float64], axis: Int) -> Tensor[Float64]:
        var ax = Optional[Int](axis)
        return _mean_free(self, ax, False)

    # allows x.mean(axis=[...]) i.e., keyword 'axis' is a List[Int]
    fn mean(self: Tensor[Float64], axis: List[Int]) -> Tensor[Float64]:
        return mean_axes_f64(self, axis, False)


    # ---------- Float32 overloads (upcast to f64) ----------

    fn mean(self: Tensor[Float32]) -> Tensor[Float64]:
        var A = astype_f64_from_f32(self)
        var none = Optional[Int]()
        return _mean_free(A, none, False)

    fn mean(self: Tensor[Float32], axis: Int) -> Tensor[Float64]:
        var A = astype_f64_from_f32(self)
        var ax = Optional[Int](axis)
        return _mean_free(A, ax, False)

    fn mean(self: Tensor[Float32], axis: List[Int]) -> Tensor[Float64]:
        var A = astype_f64_from_f32(self)
        return mean_axes_f64(A, axis, False)


    # ---------- Int overloads (upcast to f64) ----------

    fn mean(self: Tensor[Int]) -> Tensor[Float64]:
        var A = astype_f64_from_int(self)
        var none = Optional[Int]()
        return _mean_free(A, none, False)

    fn mean(self: Tensor[Int], axis: Int) -> Tensor[Float64]:
        var A = astype_f64_from_int(self)
        var ax = Optional[Int](axis)
        return _mean_free(A, ax, False)

    # allows diff.mean(axis=[0,2,3])
    fn mean(self: Tensor[Int], axis: List[Int]) -> Tensor[Float64]:
        var A = astype_f64_from_int(self)
        return mean_axes_f64(A, axis, False)





    @always_inline
    fn astype[U: ImplicitlyCopyable & Copyable & Movable](
        self,
        f: fn (T) -> U
    ) -> Tensor[U]:
        return astype_with[T, U](self, f)


    # ------------------------------ __str__ -----------------------------------
    # Public: pretty string with full data

    fn to_string(self: Tensor[Int]) -> String:
        return tensor_core_print[Int](self, _fmt_int)

    fn to_string(self: Tensor[Int16]) -> String:
        return tensor_core_print[Int16](self, _fmt_i16)

    fn to_string(self: Tensor[Int32]) -> String:
        return tensor_core_print[Int32](self, _fmt_i32)

    fn to_string(self: Tensor[Int64]) -> String:
        return tensor_core_print[Int64](self, _fmt_i64)

    fn to_string(self: Tensor[Float32]) -> String:
        return tensor_core_print[Float32](self, _fmt_f32)

    fn to_string(self: Tensor[Float64]) -> String:
        return tensor_core_print[Float64](self, _fmt_f64)

    fn to_string(self: Tensor[Bool]) -> String:
        return tensor_core_print[Bool](self, _fmt_bool)

    fn to_string(self: Tensor[String]) -> String:
        return tensor_core_print[String](self, _fmt_str)


    fn dump(self: Tensor[Int]) -> None:
        print(tensor_core_print[Int](self, _fmt_int))

    fn dump(self: Tensor[Int16]) -> None:
        print(tensor_core_print[Int16](self, _fmt_i16))

    fn dump(self: Tensor[Int32]) -> None:
        print(tensor_core_print[Int32](self, _fmt_i32))

    fn dump(self: Tensor[Int64]) -> None:
        print(tensor_core_print[Int64](self, _fmt_i64))

    fn dump(self: Tensor[Float32]) -> None:
        print(tensor_core_print[Float32](self, _fmt_f32))

    fn dump(self: Tensor[Float64]) -> None:
        print(tensor_core_print[Float64](self, _fmt_f64))

    fn dump(self: Tensor[Bool]) -> None:
        print(tensor_core_print[Bool](self, _fmt_bool))

    fn dump(self: Tensor[String]) -> None:
        print(tensor_core_print[String](self, _fmt_str))


    fn __str__(self: Tensor[Int]) -> String:
        return tensor_core_print[Int](self, _fmt_int)

    fn __str__(self: Tensor[Int16]) -> String:
        return tensor_core_print[Int16](self, _fmt_i16)

    fn __str__(self: Tensor[Int32]) -> String:
        return tensor_core_print[Int32](self, _fmt_i32)

    fn __str__(self: Tensor[Int64]) -> String:
        return tensor_core_print[Int64](self, _fmt_i64)

    fn __str__(self: Tensor[Float32]) -> String:
        return tensor_core_print[Float32](self, _fmt_f32)

    fn __str__(self: Tensor[Float64]) -> String:
        return tensor_core_print[Float64](self, _fmt_f64)

    fn __str__(self: Tensor[Bool]) -> String:
        return tensor_core_print[Bool](self, _fmt_bool)

    fn __str__(self: Tensor[String]) -> String:
        return tensor_core_print[String](self, _fmt_str)

    # cast helper
    fn astype_float64(self: Tensor[Int]) -> Tensor[Float64]:
        var shp = self._shape.copy()
        var flat = List[Float64]()
        flat.reserve(len(self._data))
        var i = 0
        while i < len(self._data):
            flat.append(Float64(self._data[i]))
            i = i + 1
        return Tensor[Float64](shp, flat)

    # existing same-type overloads must already exist:
    # fn set_union(self: Tensor[Int], other: Tensor[Int]) -> Tensor[Int]: ...
    # fn set_union(self: Tensor[Float64], other: Tensor[Float64]) -> Tensor[Float64]: ...

    @always_inline
    fn set_union(self: Tensor[Int], other: Tensor[Int]) -> Tensor[Int]:
        var u1 = setops_sorted_unique_int(self)   # -> List[Int] sorted-unique
        var u2 = setops_sorted_unique_int(other)

        var i = 0
        var j = 0
        var out: List[Int] = List[Int]()
        out.reserve(len(u1) + len(u2))

        while i < len(u1) and j < len(u2):
            var a = u1[i]
            var b = u2[j]
            if a == b:
                out.append(a); i += 1; j += 1
            elif a < b:
                out.append(a); i += 1
            else:
                out.append(b); j += 1

        while i < len(u1): out.append(u1[i]); i += 1
        while j < len(u2): out.append(u2[j]); j += 1

        var shp: List[Int] = List[Int]()
        shp.append(len(out))
        var strides = compute_row_major_strides(shp)
        return Tensor[Int](out, shp, strides, 0)


    @always_inline
    fn set_intersection(self: Tensor[Int], other: Tensor[Int]) -> Tensor[Int]:
        var u1 = setops_sorted_unique_int(self)
        var u2 = setops_sorted_unique_int(other)

        var i = 0
        var j = 0
        var out: List[Int] = List[Int]()
        if len(u1) < len(u2): out.reserve(len(u1)) else: out.reserve(len(u2))

        while i < len(u1) and j < len(u2):
            var a = u1[i]
            var b = u2[j]
            if a == b:
                out.append(a); i += 1; j += 1
            elif a < b:
                i += 1
            else:
                j += 1

        var shp: List[Int] = List[Int]()
        shp.append(len(out))
        var strides = compute_row_major_strides(shp)
        return Tensor[Int](out, shp, strides, 0)


    @always_inline
    fn set_difference(self: Tensor[Int], other: Tensor[Int]) -> Tensor[Int]:
        # u1 - u2, inputs already unique+sorted
        var u1 = setops_sorted_unique_int(self)   # expected: List[Int]
        var u2 = setops_sorted_unique_int(other)  # expected: List[Int]

        var i = 0
        var j = 0
        var out = List[Int]()
        out.reserve(len(u1))

        while i < len(u1):
            var a = u1[i]
            # Advance j while u2[j] < a
            while j < len(u2) and u2[j] < a:
                j = j + 1
            # If not found equal, keep a
            if j >= len(u2) or u2[j] != a:
                out.append(a)
            i = i + 1

        return tensor1d_from_list[Int](out)

    @always_inline
    fn set_xor(self: Tensor[Int], other: Tensor[Int]) -> Tensor[Int]:
        # Symmetric difference of two sorted-unique lists
        var u1 = setops_sorted_unique_int(self)
        var u2 = setops_sorted_unique_int(other)

        var i = 0
        var j = 0
        var out = List[Int]()
        out.reserve(len(u1) + len(u2))

        while i < len(u1) or j < len(u2):
            if i < len(u1) and (j >= len(u2) or u1[i] < u2[j]):
                out.append(u1[i]); i = i + 1
            elif j < len(u2) and (i >= len(u1) or u2[j] < u1[i]):
                out.append(u2[j]); j = j + 1
            else:
                # equal: skip both to keep symmetric difference
                i = i + 1
                j = j + 1

        return tensor1d_from_list[Int](out)

    # ========================= Float64 ===========================

    @always_inline
    fn set_union(self: Tensor[Float64], other: Tensor[Float64]) -> Tensor[Float64]:
        var u1 = setops_sorted_unique_f64(self)
        var u2 = setops_sorted_unique_f64(other)

        var i = 0
        var j = 0
        var out = List[Float64]()
        out.reserve(len(u1) + len(u2))

        while i < len(u1) and j < len(u2):
            var a = u1[i]
            var b = u2[j]
            if a == b:
                out.append(a); i = i + 1; j = j + 1
            elif a < b:
                out.append(a); i = i + 1
            else:
                out.append(b); j = j + 1

        while i < len(u1): out.append(u1[i]); i = i + 1
        while j < len(u2): out.append(u2[j]); j = j + 1

        return tensor1d_from_list[Float64](out)

    @always_inline
    fn set_intersection(self: Tensor[Float64], other: Tensor[Float64]) -> Tensor[Float64]:
        var u1 = setops_sorted_unique_f64(self)
        var u2 = setops_sorted_unique_f64(other)

        var i = 0
        var j = 0
        var out = List[Float64]()
        if len(u1) < len(u2): out.reserve(len(u1)) else: out.reserve(len(u2))

        while i < len(u1) and j < len(u2):
            var a = u1[i]
            var b = u2[j]
            if a == b:
                out.append(a); i = i + 1; j = j + 1
            elif a < b:
                i = i + 1
            else:
                j = j + 1

        return tensor1d_from_list[Float64](out)

    @always_inline
    fn set_difference(self: Tensor[Float64], other: Tensor[Float64]) -> Tensor[Float64]:
        var u1 = setops_sorted_unique_f64(self)
        var u2 = setops_sorted_unique_f64(other)

        var i = 0
        var j = 0
        var out = List[Float64]()
        out.reserve(len(u1))

        while i < len(u1):
            var a = u1[i]
            while j < len(u2) and u2[j] < a:
                j = j + 1
            if j >= len(u2) or u2[j] != a:
                out.append(a)
            i = i + 1

        return tensor1d_from_list[Float64](out)

    @always_inline
    fn set_xor(self: Tensor[Float64], other: Tensor[Float64]) -> Tensor[Float64]:
        var u1 = setops_sorted_unique_f64(self)
        var u2 = setops_sorted_unique_f64(other)

        var i = 0
        var j = 0
        var out = List[Float64]()
        out.reserve(len(u1) + len(u2))

        while i < len(u1) or j < len(u2):
            if i < len(u1) and (j >= len(u2) or u1[i] < u2[j]):
                out.append(u1[i]); i = i + 1
            elif j < len(u2) and (i >= len(u1) or u2[j] < u1[i]):
                out.append(u2[j]); j = j + 1
            else:
                i = i + 1
                j = j + 1

        return tensor1d_from_list[Float64](out)

    # ---- mixed overloads (result is Float64) ----
    fn set_union(self: Tensor[Int], other: Tensor[Float64]) -> Tensor[Float64]:
        return astype_float64(self).set_union(other)

    fn set_union(self: Tensor[Float64], other: Tensor[Int]) -> Tensor[Float64]:
        return self.set_union(astype_float64(other))

    fn set_intersection(self: Tensor[Int], other: Tensor[Float64]) -> Tensor[Float64]:
        return astype_float64(self).set_intersection(other)

    fn set_intersection(self: Tensor[Float64], other: Tensor[Int]) -> Tensor[Float64]:
        return self.set_intersection(astype_float64(other))

    fn set_difference(self: Tensor[Int], other: Tensor[Float64]) -> Tensor[Float64]:
        return astype_float64(self).set_difference(other)

    fn set_difference(self: Tensor[Float64], other: Tensor[Int]) -> Tensor[Float64]:
        return self.set_difference(astype_float64(other))

    fn set_xor(self: Tensor[Int], other: Tensor[Float64]) -> Tensor[Float64]:
        return astype_float64(self).set_xor(other)

    fn set_xor(self: Tensor[Float64], other: Tensor[Int]) -> Tensor[Float64]:
        return self.set_xor(astype_float64(other))

    fn __len__(self) -> Int:
        var n = 1
        var i = 0
        var d = len(self._shape)          # uses List's len
        while i < d:
            n = n * self._shape[i]
            i = i + 1
        return n

    fn unique(self: Tensor[Int]) -> UniqueResult[Int]:
        return tensor_unique_int(self)

    fn unique(self: Tensor[Float64]) -> UniqueResult[Float64]:
        return tensor_unique_f64(self)

    fn unique(self: Tensor[Float32]) -> UniqueResult[Float32]:
        return tensor_unique_f32(self)

    fn topk(self:Tensor[Float64], k: Int, largest: Bool = True, axis: Optional[Int] = None) -> (Tensor[Float64], Tensor[Int32]):
        return topk(self, k, largest , axis)

    fn topk(self:Tensor[Float32], k: Int, largest: Bool = True, axis: Optional[Int] = None) -> (Tensor[Float32], Tensor[Int32]):
        return topk(self, k, largest , axis)

    fn topk(self:Tensor[Int], k: Int, largest: Bool = True, axis: Optional[Int] = None) -> (Tensor[Int], Tensor[Int32]):
        return topk(self, k, largest , axis)


    @always_inline
    fn sort(self: Tensor[Float64]) -> Tensor[Float64]:
        return tensor_sort_f64(self)

    @always_inline
    fn sort(self: Tensor[Float32]) -> Tensor[Float32]:
        return tensor_sort_f32(self)

    @always_inline
    fn sort(self: Tensor[Int]) -> Tensor[Int]:
        return tensor_sort_int(self)



    @always_inline
    fn argsort(self: Tensor[Float64]) -> Tensor[Int]:
        return argsort_f64(self)

    @always_inline
    fn argsort(self: Tensor[Float32]) -> Tensor[Int]:
        return argsort_f32(self)

    @always_inline
    fn argsort(self: Tensor[Int]) -> Tensor[Int]:
        return argsort_int(self)


    @always_inline
    fn bincount(self: Tensor[Int]) -> Tensor[Int]:
        return tensor_bincount_int(self)

    @always_inline
    fn histogram(self: Tensor[Int], bins: List[Int]) -> UniqueResult[Int]:
        return tensor_histogram_int(self, bins)

    @always_inline
    fn digitize(self: Tensor[Int], edges: List[Int]) -> Tensor[Int]:
        return tensor_digitize_int(self, edges)





    # =========================
    # Elementwise EXP overloads
    # =========================

    @always_inline
    fn exp(self: Tensor[Int]) -> Tensor[Float64]:
        return exp_t(self)

    @always_inline
    fn exp(self: Tensor[Float64]) -> Tensor[Float64]:
        return exp_t(self)

    @always_inline
    fn exp(self: Tensor[Float32]) -> Tensor[Float64]:
        return exp_t(self)


    # =========================
    # Elementwise LOG overloads
    # =========================

    @always_inline
    fn log(self: Tensor[Int]) -> Tensor[Float64]:
        return log_t(self)

    @always_inline
    fn log(self: Tensor[Float64]) -> Tensor[Float64]:
        return log_t(self)

    @always_inline
    fn log(self: Tensor[Float32]) -> Tensor[Float64]:
        return log_t(self)


    # ==========================
    # Elementwise SQRT overloads
    # ==========================

    @always_inline
    fn sqrt(self: Tensor[Int]) -> Tensor[Float64]:
        return sqrt_t(self)

    @always_inline
    fn sqrt(self: Tensor[Float64]) -> Tensor[Float64]:
        return sqrt_t(self)

    @always_inline
    fn sqrt(self: Tensor[Float32]) -> Tensor[Float64]:
        return sqrt_t(self)


    # =========================
    # Elementwise ABS overloads
    # =========================

    @always_inline
    fn abs(self: Tensor[Int]) -> Tensor[Float64]:
        return abs_t(self)

    @always_inline
    fn abs(self: Tensor[Float64]) -> Tensor[Float64]:
        return abs_t(self)

    @always_inline
    fn abs(self: Tensor[Float32]) -> Tensor[Float64]:
        return abs_t(self)




    # =========================
    # Elementwise sin overloads
    # =========================

    @always_inline
    fn sin(self: Tensor[Int]) -> Tensor[Float64]:
        return sin_t(self)

    @always_inline
    fn sin(self: Tensor[Float64]) -> Tensor[Float64]:
        return sin_t(self)

    @always_inline
    fn sin(self: Tensor[Float32]) -> Tensor[Float64]:
        return sin_t(self)



    # =========================
    # Elementwise cos overloads
    # =========================

    @always_inline
    fn cos(self: Tensor[Int]) -> Tensor[Float64]:
        return cos_t(self)

    @always_inline
    fn cos(self: Tensor[Float64]) -> Tensor[Float64]:
        return cos_t(self)

    @always_inline
    fn cos(self: Tensor[Float32]) -> Tensor[Float64]:
        return cos_t(self)



    # =========================
    # Elementwise tan overloads
    # =========================

    @always_inline
    fn tan(self: Tensor[Int]) -> Tensor[Float64]:
        return tan_t(self)

    @always_inline
    fn tan(self: Tensor[Float64]) -> Tensor[Float64]:
        return tan_t(self)

    @always_inline
    fn tan(self: Tensor[Float32]) -> Tensor[Float64]:
        return tan_t(self)



    # =========================
    # Elementwise relu overloads
    # =========================

    @always_inline
    fn relu(self: Tensor[Int]) -> Tensor[Float64]:
        return relu_t(self)

    @always_inline
    fn relu(self: Tensor[Float64]) -> Tensor[Float64]:
        return relu_t(self)

    @always_inline
    fn relu(self: Tensor[Float32]) -> Tensor[Float64]:
        return relu_t(self)



    # =========================
    # Elementwise expm1 overloads
    # =========================

    @always_inline
    fn expm1(self: Tensor[Int]) -> Tensor[Float64]:
        return expm1_t(self)

    @always_inline
    fn expm1(self: Tensor[Float64]) -> Tensor[Float64]:
        return expm1_t(self)

    @always_inline
    fn expm1(self: Tensor[Float32]) -> Tensor[Float64]:
        return expm1_t(self)



    # =========================
    # Elementwise log1p overloads
    # =========================

    @always_inline
    fn log1p(self: Tensor[Int]) -> Tensor[Float64]:
        return log1p_t(self)

    @always_inline
    fn log1p(self: Tensor[Float64]) -> Tensor[Float64]:
        return log1p_t(self)

    @always_inline
    fn log1p(self: Tensor[Float32]) -> Tensor[Float64]:
        return log1p_t(self)


    # =========================
    # Elementwise floor overloads
    # =========================

    @always_inline
    fn floor(self: Tensor[Int]) -> Tensor[Float64]:
        return floor_t(self)

    @always_inline
    fn floor(self: Tensor[Float64]) -> Tensor[Float64]:
        return floor_t(self)

    @always_inline
    fn floor(self: Tensor[Float32]) -> Tensor[Float64]:
        return floor_t(self)


    # =========================
    # Elementwise ceil overloads
    # =========================

    @always_inline
    fn ceil(self: Tensor[Int]) -> Tensor[Float64]:
        return ceil_t(self)

    @always_inline
    fn ceil(self: Tensor[Float64]) -> Tensor[Float64]:
        return ceil_t(self)

    @always_inline
    fn ceil(self: Tensor[Float32]) -> Tensor[Float64]:
        return ceil_t(self)


    # =========================
    # Elementwise round overloads
    # =========================

    @always_inline
    fn round(self: Tensor[Int]) -> Tensor[Float64]:
        return round_t(self)

    @always_inline
    fn round(self: Tensor[Float64]) -> Tensor[Float64]:
        return round_t(self)

    @always_inline
    fn round(self: Tensor[Float32]) -> Tensor[Float64]:
        return round_t(self)


    # =========================
    # Elementwise sign overloads
    # =========================

    @always_inline
    fn sign(self: Tensor[Int]) -> Tensor[Float64]:
        return sign_t(self)

    @always_inline
    fn sign(self: Tensor[Float64]) -> Tensor[Float64]:
        return sign_t(self)

    @always_inline
    fn sign(self: Tensor[Float32]) -> Tensor[Float64]:
        return sign_t(self)


    # =========================
    # Elementwise sigmoid overloads
    # =========================

    @always_inline
    fn sigmoid(self: Tensor[Int]) -> Tensor[Float64]:
        return sigmoid_t(self)

    @always_inline
    fn sigmoid(self: Tensor[Float64]) -> Tensor[Float64]:
        return sigmoid_t(self)

    @always_inline
    fn sigmoid(self: Tensor[Float32]) -> Tensor[Float64]:
        return sigmoid_t(self)


    # =========================
    # Elementwise tanh overloads
    # =========================

    @always_inline
    fn tanh(self: Tensor[Int]) -> Tensor[Float64]:
        return tanh_t(self)

    @always_inline
    fn tanh(self: Tensor[Float64]) -> Tensor[Float64]:
        return tanh_t(self)

    @always_inline
    fn tanh(self: Tensor[Float32]) -> Tensor[Float64]:
        return tanh_t(self)


    # =========================
    # Elementwise silu overloads
    # =========================

    @always_inline
    fn silu(self: Tensor[Int]) -> Tensor[Float64]:
        return silu_t(self)

    @always_inline
    fn silu(self: Tensor[Float64]) -> Tensor[Float64]:
        return silu_t(self)

    @always_inline
    fn silu(self: Tensor[Float32]) -> Tensor[Float64]:
        return silu_t(self)


    # =========================
    # Elementwise gelu overloads
    # =========================

    @always_inline
    fn gelu(self: Tensor[Int]) -> Tensor[Float64]:
        return gelu_t(self)

    @always_inline
    fn gelu(self: Tensor[Float64]) -> Tensor[Float64]:
        return gelu_t(self)

    @always_inline
    fn gelu(self: Tensor[Float32]) -> Tensor[Float64]:
        return gelu_t(self)


    # =========================
    # Elementwise elu overloads
    # =========================

    @always_inline
    fn elu(self: Tensor[Int]) -> Tensor[Float64]:
        return elu_t(self)

    @always_inline
    fn elu(self: Tensor[Float64]) -> Tensor[Float64]:
        return elu_t(self)

    @always_inline
    fn elu(self: Tensor[Float32]) -> Tensor[Float64]:
        return elu_t(self)

    # =========================
    # Elementwise selu overloads
    # =========================

    @always_inline
    fn selu(self: Tensor[Int]) -> Tensor[Float64]:
        return selu_t(self)

    @always_inline
    fn selu(self: Tensor[Float64]) -> Tensor[Float64]:
        return selu_t(self)

    @always_inline
    fn selu(self: Tensor[Float32]) -> Tensor[Float64]:
        return selu_t(self)


   # Any / All / Count for Float64
    fn any(self: Tensor[Float64]) -> Bool:
        var n = len(self._data)
        var i = 0
        while i < n:
            if self._data[i] != 0.0: return True
            i += 1
        return False

    fn all(self: Tensor[Float64]) -> Bool:
        var n = len(self._data)
        var i = 0
        while i < n:
            if self._data[i] == 0.0: return False
            i += 1
        return True

    fn count_nonzero(self: Tensor[Float64]) -> Int:
        var n = len(self._data)
        var c = 0
        var i = 0
        while i < n:
            if self._data[i] != 0.0: c += 1
            i += 1
        return c

    # Any / All / Count for Float32
    fn any(self: Tensor[Float32]) -> Bool:
        var n = len(self._data)
        var i = 0
        while i < n:
            if self._data[i] != 0.0: return True
            i += 1
        return False

    fn all(self: Tensor[Float32]) -> Bool:
        var n = len(self._data)
        var i = 0
        while i < n:
            if self._data[i] == 0.0: return False
            i += 1
        return True

    fn count_nonzero(self: Tensor[Float32]) -> Int:
        var n = len(self._data)
        var c = 0
        var i = 0
        while i < n:
            if self._data[i] != 0.0: c += 1
            i += 1
        return c

    # Any / All / Count for Int (assumes 0==False, nonzero==True)
    fn any(self: Tensor[Int]) -> Bool:
        var n = len(self._data)
        var i = 0
        while i < n:
            if self._data[i] != 0: return True
            i += 1
        return False

    fn all(self: Tensor[Int]) -> Bool:
        var n = len(self._data)
        var i = 0
        while i < n:
            if self._data[i] == 0: return False
            i += 1
        return True

    fn count_nonzero(self: Tensor[Int]) -> Int:
        var n = len(self._data)
        var c = 0
        var i = 0
        while i < n:
            if self._data[i] != 0: c += 1
            i += 1
        return c
    # Optional: Bool tensor
    fn any(self: Tensor[Bool]) -> Bool:
        var n = len(self._data)
        var i = 0
        while i < n:
            if self._data[i]: return True
            i += 1
        return False

    fn all(self: Tensor[Bool]) -> Bool:
        var n = len(self._data)
        var i = 0
        while i < n:
            if not self._data[i]: return False
            i += 1
        return True

    fn count_nonzero(self: Tensor[Bool]) -> Int:
        var n = len(self._data)
        var c = 0
        var i = 0
        while i < n:
            if self._data[i]: c += 1
            i += 1
        return c

    # ----------------------
    # self: Tensor[Float64]
    # ----------------------
    fn clip(self: Tensor[Float64], lo: Float64, hi: Float64) -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(self, scalar_f64(hi)), self, scalar_f64(hi)), scalar_f64(lo)), where_f64(le_t(self, scalar_f64(hi)), self, scalar_f64(hi)), scalar_f64(lo))     # clip f64,f64
    fn clip(self: Tensor[Float64], lo: Float64, hi: Float32) -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(self, to_float64(scalar_f32(hi))), self, to_float64(scalar_f32(hi))), scalar_f64(lo)), where_f64(le_t(self, to_float64(scalar_f32(hi))), self, to_float64(scalar_f32(hi))), scalar_f64(lo)) # clip f64,f32→f64
    fn clip(self: Tensor[Float64], lo: Float64, hi: Int)     -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(self, to_float64(scalar_int(hi))), self, to_float64(scalar_int(hi))), scalar_f64(lo)), where_f64(le_t(self, to_float64(scalar_int(hi))), self, to_float64(scalar_int(hi))), scalar_f64(lo)) # clip f64,int→f64
    fn clip(self: Tensor[Float64], lo: Float32, hi: Float64) -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(self, scalar_f64(hi)), self, scalar_f64(hi)), to_float64(scalar_f32(lo))), where_f64(le_t(self, scalar_f64(hi)), self, scalar_f64(hi)), to_float64(scalar_f32(lo)))                                 # clip f32→f64,f64
    fn clip(self: Tensor[Float64], lo: Float32, hi: Float32) -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(self, to_float64(scalar_f32(hi))), self, to_float64(scalar_f32(hi))), to_float64(scalar_f32(lo))), where_f64(le_t(self, to_float64(scalar_f32(hi))), self, to_float64(scalar_f32(hi))), to_float64(scalar_f32(lo))) # clip f32→f64,f32→f64
    fn clip(self: Tensor[Float64], lo: Float32, hi: Int)     -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(self, to_float64(scalar_int(hi))), self, to_float64(scalar_int(hi))), to_float64(scalar_f32(lo))), where_f64(le_t(self, to_float64(scalar_int(hi))), self, to_float64(scalar_int(hi))), to_float64(scalar_f32(lo)))     # clip f32→f64,int→f64
    fn clip(self: Tensor[Float64], lo: Int,     hi: Float64) -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(self, scalar_f64(hi)), self, scalar_f64(hi)), to_float64(scalar_int(lo))), where_f64(le_t(self, scalar_f64(hi)), self, scalar_f64(hi)), to_float64(scalar_int(lo)))                                 # clip int→f64,f64
    fn clip(self: Tensor[Float64], lo: Int,     hi: Float32) -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(self, to_float64(scalar_f32(hi))), self, to_float64(scalar_f32(hi))), to_float64(scalar_int(lo))), where_f64(le_t(self, to_float64(scalar_f32(hi))), self, to_float64(scalar_f32(hi))), to_float64(scalar_int(lo)))         # clip int→f64,f32→f64
    fn clip(self: Tensor[Float64], lo: Int,     hi: Int)     -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(self, to_float64(scalar_int(hi))), self, to_float64(scalar_int(hi))), to_float64(scalar_int(lo))), where_f64(le_t(self, to_float64(scalar_int(hi))), self, to_float64(scalar_int(hi))), to_float64(scalar_int(lo)))         # clip int→f64,int→f64

    fn minimum_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Float64]: return where_f64(le_t(self, scalar_f64(s)), self, scalar_f64(s))                                   # min f64
    fn minimum_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Float64]: return where_f64(le_t(self, to_float64(scalar_f32(s))), self, to_float64(scalar_f32(s)))           # min f32→f64
    fn minimum_scalar(self: Tensor[Float64], s: Int)     -> Tensor[Float64]: return where_f64(le_t(self, to_float64(scalar_int(s))), self, to_float64(scalar_int(s)))       # min int→f64

    fn maximum_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Float64]: return where_f64(ge_t(self, scalar_f64(s)), self, scalar_f64(s))                                   # max f64
    fn maximum_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Float64]: return where_f64(ge_t(self, to_float64(scalar_f32(s))), self, to_float64(scalar_f32(s)))           # max f32→f64
    fn maximum_scalar(self: Tensor[Float64], s: Int)     -> Tensor[Float64]: return where_f64(ge_t(self, to_float64(scalar_int(s))), self, to_float64(scalar_int(s)))       # max int→f64

    fn min(self: Tensor[Float64]) -> Float64: return min_t(self)                                   # min f64
    fn max(self: Tensor[Float64]) -> Float64: return max_t(self)                                   # max f64



    fn can_broadcast_with(self: Tensor[Float64], other: Tensor[Float64],strict: Bool = False) -> Bool:
        return can_broadcast_shapes(self._shape, other._shape,strict)

    fn can_broadcast_with(self: Tensor[Float64], other: Tensor[Float32],strict: Bool = False) -> Bool:
        return can_broadcast_shapes(self._shape, other._shape,strict)

    fn can_broadcast_with(self: Tensor[Float64], other: Tensor[Int],strict: Bool = False) -> Bool:
        return can_broadcast_shapes(self._shape, other._shape,strict)

    fn can_broadcast_with(self: Tensor[Float32], other: Tensor[Float64],strict: Bool = False) -> Bool:
        return can_broadcast_shapes(self._shape, other._shape,strict)

    fn can_broadcast_with(self: Tensor[Float32], other: Tensor[Float32],strict: Bool = False) -> Bool:
        return can_broadcast_shapes(self._shape, other._shape,strict)

    fn can_broadcast_with(self: Tensor[Float32], other: Tensor[Int],strict: Bool = False) -> Bool:
        return can_broadcast_shapes(self._shape, other._shape,strict)

    fn can_broadcast_with(self: Tensor[Int], other: Tensor[Float64],strict: Bool = False) -> Bool:
        return can_broadcast_shapes(self._shape, other._shape,strict)

    fn can_broadcast_with(self: Tensor[Int], other: Tensor[Float32],strict: Bool = False) -> Bool:
        return can_broadcast_shapes(self._shape, other._shape,strict)

    fn can_broadcast_with(self: Tensor[Int], other: Tensor[Int],strict: Bool = False) -> Bool:
        return can_broadcast_shapes(self._shape, other._shape,strict)

    # ----------------------
    # self: Tensor[Float32]
    # ----------------------
    # clip → if any Float64 is present, output Float64; else output Float32
    fn clip(self: Tensor[Float32], lo: Float64, hi: Float64) -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(to_float64(self), scalar_f64(hi)), to_float64(self), scalar_f64(hi)), scalar_f64(lo)), where_f64(le_t(to_float64(self), scalar_f64(hi)), to_float64(self), scalar_f64(hi)), scalar_f64(lo))                         # f64,f64 ⇒ f64
    fn clip(self: Tensor[Float32], lo: Float64, hi: Float32) -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(to_float64(self), to_float64(scalar_f32(hi))), to_float64(self), to_float64(scalar_f32(hi))), scalar_f64(lo)), where_f64(le_t(to_float64(self), to_float64(scalar_f32(hi))), to_float64(self), to_float64(scalar_f32(hi))), scalar_f64(lo)) # f64,f32 ⇒ f64
    fn clip(self: Tensor[Float32], lo: Float64, hi: Int)     -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(to_float64(self), to_float64(scalar_int(hi))), to_float64(self), to_float64(scalar_int(hi))), scalar_f64(lo)), where_f64(le_t(to_float64(self), to_float64(scalar_int(hi))), to_float64(self), to_float64(scalar_int(hi))), scalar_f64(lo))     # f64,int ⇒ f64
    fn clip(self: Tensor[Float32], lo: Float32, hi: Float64) -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(to_float64(self), scalar_f64(hi)), to_float64(self), scalar_f64(hi)), to_float64(scalar_f32(lo))), where_f64(le_t(to_float64(self), scalar_f64(hi)), to_float64(self), scalar_f64(hi)), to_float64(scalar_f32(lo)))                 # f32,f64 ⇒ f64
    fn clip(self: Tensor[Float32], lo: Int,     hi: Float64) -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(to_float64(self), scalar_f64(hi)), to_float64(self), scalar_f64(hi)), to_float64(scalar_int(lo))), where_f64(le_t(to_float64(self), scalar_f64(hi)), to_float64(self), scalar_f64(hi)), to_float64(scalar_int(lo)))                 # int,f64 ⇒ f64

    fn clip(self: Tensor[Float32], lo: Float32, hi: Float32) -> Tensor[Float32]: return where_f32(ge_t(where_f32(le_t(self, scalar_f32(hi)), self, scalar_f32(hi)), scalar_f32(lo)), where_f32(le_t(self, scalar_f32(hi)), self, scalar_f32(hi)), scalar_f32(lo))     # f32,f32 ⇒ f32
    fn clip(self: Tensor[Float32], lo: Float32, hi: Int)     -> Tensor[Float32]: return where_f32(ge_t(where_f32(le_t(self, to_float32(scalar_int(hi))), self, to_float32(scalar_int(hi))), scalar_f32(lo)), where_f32(le_t(self, to_float32(scalar_int(hi))), self, to_float32(scalar_int(hi))), scalar_f32(lo))                         # f32,int ⇒ f32
    fn clip(self: Tensor[Float32], lo: Int,     hi: Float32) -> Tensor[Float32]: return where_f32(ge_t(where_f32(le_t(self, scalar_f32(hi)), self, scalar_f32(hi)), to_float32(scalar_int(lo))), where_f32(le_t(self, scalar_f32(hi)), self, scalar_f32(hi)), to_float32(scalar_int(lo)))                                             # int,f32 ⇒ f32
    fn clip(self: Tensor[Float32], lo: Int,     hi: Int)     -> Tensor[Float32]: return where_f32(ge_t(where_f32(le_t(self, to_float32(scalar_int(hi))), self, to_float32(scalar_int(hi))), to_float32(scalar_int(lo))), where_f32(le_t(self, to_float32(scalar_int(hi))), self, to_float32(scalar_int(hi))), to_float32(scalar_int(lo)))         # int,int ⇒ f32? (promotion says Float32 since self is Float32)

    # min/max → if s is Float64 ⇒ Float64; else ⇒ Float32
    fn minimum_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Float64]: return where_f64(le_t(to_float64(self), scalar_f64(s)), to_float64(self), scalar_f64(s))                           # min s=f64 ⇒ f64
    fn maximum_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Float64]: return where_f64(ge_t(to_float64(self), scalar_f64(s)), to_float64(self), scalar_f64(s))                           # max s=f64 ⇒ f64
    fn minimum_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Float32]: return where_f32(le_t(self, scalar_f32(s)), self, scalar_f32(s))                                                  # min s=f32 ⇒ f32
    fn maximum_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Float32]: return where_f32(ge_t(self, scalar_f32(s)), self, scalar_f32(s))                                                  # max s=f32 ⇒ f32
    fn minimum_scalar(self: Tensor[Float32], s: Int)     -> Tensor[Float32]: return where_f32(le_t(self, to_float32(scalar_int(s))), self, to_float32(scalar_int(s)))                      # min s=int ⇒ f32
    fn maximum_scalar(self: Tensor[Float32], s: Int)     -> Tensor[Float32]: return where_f32(ge_t(self, to_float32(scalar_int(s))), self, to_float32(scalar_int(s)))                      # max s=int ⇒ f32


    fn min(self: Tensor[Float32]) -> Float32: return min_t(self)
    fn max(self: Tensor[Float32]) -> Float32: return max_t(self)
    # ----------------------
    # self: Tensor[Int]
    # ----------------------
    # clip → if any Float64 ⇒ Float64; else if any Float32 ⇒ Float32; else ⇒ Int
    # ⇒ 9 overloads grouped by target

    # target Float64 (any Float64 present)
    fn clip(self: Tensor[Int], lo: Float64, hi: Float64) -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(to_float64(self), scalar_f64(hi)), to_float64(self), scalar_f64(hi)), scalar_f64(lo)), where_f64(le_t(to_float64(self), scalar_f64(hi)), to_float64(self), scalar_f64(hi)), scalar_f64(lo))                                 # f64,f64 ⇒ f64
    fn clip(self: Tensor[Int], lo: Float64, hi: Float32) -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(to_float64(self), to_float64(scalar_f32(hi))), to_float64(self), to_float64(scalar_f32(hi))), scalar_f64(lo)), where_f64(le_t(to_float64(self), to_float64(scalar_f32(hi))), to_float64(self), to_float64(scalar_f32(hi))), scalar_f64(lo)) # f64,f32 ⇒ f64
    fn clip(self: Tensor[Int], lo: Float64, hi: Int)     -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(to_float64(self), to_float64(scalar_int(hi))), to_float64(self), to_float64(scalar_int(hi))), scalar_f64(lo)), where_f64(le_t(to_float64(self), to_float64(scalar_int(hi))), to_float64(self), to_float64(scalar_int(hi))), scalar_f64(lo))     # f64,int ⇒ f64
    fn clip(self: Tensor[Int], lo: Float32, hi: Float64) -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(to_float64(self), scalar_f64(hi)), to_float64(self), scalar_f64(hi)), to_float64(scalar_f32(lo))), where_f64(le_t(to_float64(self), scalar_f64(hi)), to_float64(self), scalar_f64(hi)), to_float64(scalar_f32(lo)))                 # f32,f64 ⇒ f64
    fn clip(self: Tensor[Int], lo: Int,     hi: Float64) -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(to_float64(self), scalar_f64(hi)), to_float64(self), scalar_f64(hi)), to_float64(scalar_int(lo))), where_f64(le_t(to_float64(self), scalar_f64(hi)), to_float64(self), scalar_f64(hi)), to_float64(scalar_int(lo)))                 # int,f64 ⇒ f64

    # target Float32 (no Float64, but Float32 present)
    fn clip(self: Tensor[Int], lo: Float32, hi: Float32) -> Tensor[Float32]: return where_f32(ge_t(where_f32(le_t(to_float32(self), scalar_f32(hi)), to_float32(self), scalar_f32(hi)), scalar_f32(lo)), where_f32(le_t(to_float32(self), scalar_f32(hi)), to_float32(self), scalar_f32(hi)), scalar_f32(lo))                                 # f32,f32 ⇒ f32
    fn clip(self: Tensor[Int], lo: Float32, hi: Int)     -> Tensor[Float32]: return where_f32(ge_t(where_f32(le_t(to_float32(self), to_float32(scalar_int(hi))), to_float32(self), to_float32(scalar_int(hi))), scalar_f32(lo)), where_f32(le_t(to_float32(self), to_float32(scalar_int(hi))), to_float32(self), to_float32(scalar_int(hi))), scalar_f32(lo)) # f32,int ⇒ f32
    fn clip(self: Tensor[Int], lo: Int,     hi: Float32) -> Tensor[Float32]: return where_f32(ge_t(where_f32(le_t(to_float32(self), scalar_f32(hi)), to_float32(self), scalar_f32(hi)), to_float32(scalar_int(lo))), where_f32(le_t(to_float32(self), scalar_f32(hi)), to_float32(self), scalar_f32(hi)), to_float32(scalar_int(lo)))                 # int,f32 ⇒ f32

    # target Int (all Int)
    fn clip(self: Tensor[Int], lo: Int,     hi: Int)     -> Tensor[Int]: return where_int(ge_t(where_int(le_t(self, scalar_int(hi)), self, scalar_int(hi)), scalar_int(lo)), where_int(le_t(self, scalar_int(hi)), self, scalar_int(hi)), scalar_int(lo)) # int,int ⇒ int

    # min/max → result dtype is max(self, s)
    fn minimum_scalar(self: Tensor[Int], s: Float64) -> Tensor[Float64]: return where_f64(le_t(to_float64(self), scalar_f64(s)), to_float64(self), scalar_f64(s))             # min s=f64 ⇒ f64
    fn maximum_scalar(self: Tensor[Int], s: Float64) -> Tensor[Float64]: return where_f64(ge_t(to_float64(self), scalar_f64(s)), to_float64(self), scalar_f64(s))             # max s=f64 ⇒ f64
    fn minimum_scalar(self: Tensor[Int], s: Float32) -> Tensor[Float32]: return where_f32(le_t(to_float32(self), scalar_f32(s)), to_float32(self), scalar_f32(s))             # min s=f32 ⇒ f32
    fn maximum_scalar(self: Tensor[Int], s: Float32) -> Tensor[Float32]: return where_f32(ge_t(to_float32(self), scalar_f32(s)), to_float32(self), scalar_f32(s))             # max s=f32 ⇒ f32
    fn minimum_scalar(self: Tensor[Int], s: Int)     -> Tensor[Int]:     return where_int(le_t(self, scalar_int(s)), self, scalar_int(s))                               # min s=int ⇒ int
    fn maximum_scalar(self: Tensor[Int], s: Int)     -> Tensor[Int]:     return where_int(ge_t(self, scalar_int(s)), self, scalar_int(s))                               # max s=int ⇒ int

    fn min(self: Tensor[Int]) -> Int: return min_t(self)
    fn max(self: Tensor[Int]) -> Int: return max_t(self)


    fn __bool__(self) -> Bool:
        var n = len(self._data)
        if n == 0: return False
        var i = 0
        while i < n:
            if self._data[i] != 0.0: return True
            i = i + 1
        return False




    # =========================
    # Float64 overloads — all rhs combos (Tensor/Scalar: Int, Float32, Float64)
    # =========================

    # + (result: Float64)
    fn __add__(self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Float64]: return add_t(self, rhs)                                # a(f64) + b(f64)
    fn __add__(self: Tensor[Float64], rhs: Tensor[Float32]) -> Tensor[Float64]: return add_t(self, to_float64(rhs))                   # a(f64) + b(f32→f64)
    fn __add__(self: Tensor[Float64], rhs: Tensor[Int])     -> Tensor[Float64]: return add_t(self, to_float64(rhs))                   # a(f64) + b(int→f64)
    fn __add__(self: Tensor[Float64], rhs: Float64)         -> Tensor[Float64]: return add_t(self, scalar_f64(rhs))                    # a + s(f64)
    fn __add__(self: Tensor[Float64], rhs: Float32)         -> Tensor[Float64]: return add_t(self, to_float64(scalar_f32(rhs)))        # a + s(f32→f64)
    fn __add__(self: Tensor[Float64], rhs: Int)             -> Tensor[Float64]: return add_t(self, to_float64(scalar_int(rhs)))      # a + s(int→f64)

    # - (result: Float64)
    fn __sub__(self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Float64]: return sub_t(self, rhs)                                # a - b(f64)
    fn __sub__(self: Tensor[Float64], rhs: Tensor[Float32]) -> Tensor[Float64]: return sub_t(self, to_float64(rhs))                   # a - b(f32→f64)
    fn __sub__(self: Tensor[Float64], rhs: Tensor[Int])     -> Tensor[Float64]: return sub_t(self, to_float64(rhs))                   # a - b(int→f64)
    fn __sub__(self: Tensor[Float64], rhs: Float64)         -> Tensor[Float64]: return sub_t(self, scalar_f64(rhs))                    # a - s(f64)
    fn __sub__(self: Tensor[Float64], rhs: Float32)         -> Tensor[Float64]: return sub_t(self, to_float64(scalar_f32(rhs)))        # a - s(f32→f64)
    fn __sub__(self: Tensor[Float64], rhs: Int)             -> Tensor[Float64]: return sub_t(self, to_float64(scalar_int(rhs)))      # a - s(int→f64)

    # * (result: Float64)
    fn __mul__(self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Float64]: return mul_t(self, rhs)                                # a * b(f64)
    fn __mul__(self: Tensor[Float64], rhs: Tensor[Float32]) -> Tensor[Float64]: return mul_t(self, to_float64(rhs))                   # a * b(f32→f64)
    fn __mul__(self: Tensor[Float64], rhs: Tensor[Int])     -> Tensor[Float64]: return mul_t(self, to_float64(rhs))                   # a * b(int→f64)
    fn __mul__(self: Tensor[Float64], rhs: Float64)         -> Tensor[Float64]: return mul_t(self, scalar_f64(rhs))                    # a * s(f64)
    fn __mul__(self: Tensor[Float64], rhs: Float32)         -> Tensor[Float64]: return mul_t(self, to_float64(scalar_f32(rhs)))        # a * s(f32→f64)
    fn __mul__(self: Tensor[Float64], rhs: Int)             -> Tensor[Float64]: return mul_t(self, to_float64(scalar_int(rhs)))      # a * s(int→f64)

    # / (result: Float64)
    fn __truediv__(self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Float64]: return div_t(self, rhs)                           # a / b(f64)
    fn __truediv__(self: Tensor[Float64], rhs: Tensor[Float32]) -> Tensor[Float64]: return div_t(self, to_float64(rhs))              # a / b(f32→f64)
    fn __truediv__(self: Tensor[Float64], rhs: Tensor[Int])     -> Tensor[Float64]: return div_t(self, to_float64(rhs))              # a / b(int→f64)
    fn __truediv__(self: Tensor[Float64], rhs: Float64)         -> Tensor[Float64]: return div_t(self, scalar_f64(rhs))               # a / s(f64)
    fn __truediv__(self: Tensor[Float64], rhs: Float32)         -> Tensor[Float64]: return div_t(self, to_float64(scalar_f32(rhs)))   # a / s(f32→f64)
    fn __truediv__(self: Tensor[Float64], rhs: Int)             -> Tensor[Float64]: return div_t(self, to_float64(scalar_int(rhs))) # a / s(int→f64)

    # % (result: Float64)
    fn __mod__(self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Float64]: return mod_t(self, rhs)                               # a % b(f64)
    fn __mod__(self: Tensor[Float64], rhs: Tensor[Float32]) -> Tensor[Float64]: return mod_t(self, to_float64(rhs))                  # a % b(f32→f64)
    fn __mod__(self: Tensor[Float64], rhs: Tensor[Int])     -> Tensor[Float64]: return mod_t(self, to_float64(rhs))                  # a % b(int→f64)
    fn __mod__(self: Tensor[Float64], rhs: Float64)         -> Tensor[Float64]: return mod_t(self, scalar_f64(rhs))                   # a % s(f64)
    fn __mod__(self: Tensor[Float64], rhs: Float32)         -> Tensor[Float64]: return mod_t(self, to_float64(scalar_f32(rhs)))       # a % s(f32→f64)
    fn __mod__(self: Tensor[Float64], rhs: Int)             -> Tensor[Float64]: return mod_t(self, to_float64(scalar_int(rhs)))     # a % s(int→f64)
    # =========================
    # Float32 overloads — all rhs combos (Tensor/Scalar: Int, Float32, Float64)
    # +,-,*: promote to Float64 only if Float64 is involved; otherwise stay Float32
    # /,%: result is Float64 in all cases (per your earlier API)
    # =========================

    # + (result: Float64 if rhs has Float64, else Float32)
    fn __add__(self: Tensor[Float32], rhs: Tensor[Float64]) -> Tensor[Float64]: return add_t(to_float64(self), rhs)                  # a(f32→f64) + b(f64)
    fn __add__(self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Float32]: return add_t(self, rhs)                              # a + b(f32)
    fn __add__(self: Tensor[Float32], rhs: Tensor[Int])     -> Tensor[Float32]: return add_t(self, to_float32(rhs))                  # a + b(int→f32)
    fn __add__(self: Tensor[Float32], rhs: Float64)         -> Tensor[Float64]: return add_t(to_float64(self), scalar_f64(rhs))       # a(f32→f64) + s(f64)
    fn __add__(self: Tensor[Float32], rhs: Float32)         -> Tensor[Float32]: return add_t(self, scalar_f32(rhs))                   # a + s(f32)
    fn __add__(self: Tensor[Float32], rhs: Int)             -> Tensor[Float32]: return add_t(self, to_float32(scalar_int(rhs)))     # a + s(int→f32)

    # - (result: Float64 if rhs has Float64, else Float32)
    fn __sub__(self: Tensor[Float32], rhs: Tensor[Float64]) -> Tensor[Float64]: return sub_t(to_float64(self), rhs)                  # a(f32→f64) - b(f64)
    fn __sub__(self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Float32]: return sub_t(self, rhs)                              # a - b(f32)
    fn __sub__(self: Tensor[Float32], rhs: Tensor[Int])     -> Tensor[Float32]: return sub_t(self, to_float32(rhs))                  # a - b(int→f32)
    fn __sub__(self: Tensor[Float32], rhs: Float64)         -> Tensor[Float64]: return sub_t(to_float64(self), scalar_f64(rhs))       # a(f32→f64) - s(f64)
    fn __sub__(self: Tensor[Float32], rhs: Float32)         -> Tensor[Float32]: return sub_t(self, scalar_f32(rhs))                   # a - s(f32)
    fn __sub__(self: Tensor[Float32], rhs: Int)             -> Tensor[Float32]: return sub_t(self, to_float32(scalar_int(rhs)))     # a - s(int→f32)

    # * (result: Float64 if rhs has Float64, else Float32)
    fn __mul__(self: Tensor[Float32], rhs: Tensor[Float64]) -> Tensor[Float64]: return mul_t(to_float64(self), rhs)                  # a(f32→f64) * b(f64)
    fn __mul__(self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Float32]: return mul_t(self, rhs)                              # a * b(f32)
    fn __mul__(self: Tensor[Float32], rhs: Tensor[Int])     -> Tensor[Float32]: return mul_t(self, to_float32(rhs))                  # a * b(int→f32)
    fn __mul__(self: Tensor[Float32], rhs: Float64)         -> Tensor[Float64]: return mul_t(to_float64(self), scalar_f64(rhs))       # a(f32→f64) * s(f64)
    fn __mul__(self: Tensor[Float32], rhs: Float32)         -> Tensor[Float32]: return mul_t(self, scalar_f32(rhs))                   # a * s(f32)
    fn __mul__(self: Tensor[Float32], rhs: Int)             -> Tensor[Float32]: return mul_t(self, to_float32(scalar_int(rhs)))     # a * s(int→f32)

    # / (result: Float64 always)
    fn __truediv__(self: Tensor[Float32], rhs: Tensor[Float64]) -> Tensor[Float64]: return div_t(to_float64(self), rhs)              # a(f32→f64) / b(f64)
    fn __truediv__(self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Float64]: return div_t(to_float64(self), to_float64(rhs))  # a(f32→f64) / b(f32→f64)
    fn __truediv__(self: Tensor[Float32], rhs: Tensor[Int])     -> Tensor[Float64]: return div_t(to_float64(self), to_float64(rhs))  # a(f32→f64) / b(int→f64)
    fn __truediv__(self: Tensor[Float32], rhs: Float64)         -> Tensor[Float64]: return div_t(to_float64(self), scalar_f64(rhs))   # a(f32→f64) / s(f64)
    fn __truediv__(self: Tensor[Float32], rhs: Float32)         -> Tensor[Float64]: return div_t(to_float64(self), to_float64(scalar_f32(rhs))) # a(f32→f64) / s(f32→f64)
    fn __truediv__(self: Tensor[Float32], rhs: Int)             -> Tensor[Float64]: return div_t(to_float64(self), to_float64(scalar_int(rhs))) # a(f32→f64) / s(int→f64)

    # % (result: Float64 always)
    fn __mod__(self: Tensor[Float32], rhs: Tensor[Float64]) -> Tensor[Float64]: return mod_t(to_float64(self), rhs)                  # a(f32→f64) % b(f64)
    fn __mod__(self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Float64]: return mod_t(to_float64(self), to_float64(rhs))      # a(f32→f64) % b(f32→f64)
    fn __mod__(self: Tensor[Float32], rhs: Tensor[Int])     -> Tensor[Float64]: return mod_t(to_float64(self), to_float64(rhs))      # a(f32→f64) % b(int→f64)
    fn __mod__(self: Tensor[Float32], rhs: Float64)         -> Tensor[Float64]: return mod_t(to_float64(self), scalar_f64(rhs))       # a(f32→f64) % s(f64)
    fn __mod__(self: Tensor[Float32], rhs: Float32)         -> Tensor[Float64]: return mod_t(to_float64(self), to_float64(scalar_f32(rhs))) # a(f32→f64) % s(f32→f64)
    fn __mod__(self: Tensor[Float32], rhs: Int)             -> Tensor[Float64]: return mod_t(to_float64(self), to_float64(scalar_int(rhs))) # a(f32→f64) % s(int→f64)
    # =========================
    # Int overloads — all rhs combos (Tensor/Scalar: Int, Float32, Float64)
    # +,-,*: promote to widest (Float64 if present, else Float32 if present, else Int)
    # /,%: result is Float64 always
    # =========================

    # + (promotion)
    fn __add__(self: Tensor[Int], rhs: Tensor[Float64]) -> Tensor[Float64]: return add_t(to_float64(self), rhs)                     # a(int→f64) + b(f64)
    fn __add__(self: Tensor[Int], rhs: Tensor[Float32]) -> Tensor[Float32]: return add_t(to_float32(self), rhs)                    # a(int→f32) + b(f32)
    fn __add__(self: Tensor[Int], rhs: Tensor[Int])     -> Tensor[Int]:     return add_t(self, rhs)                                # a + b(int)
    fn __add__(self: Tensor[Int], rhs: Float64)         -> Tensor[Float64]: return add_t(to_float64(self), scalar_f64(rhs))         # a(int→f64) + s(f64)
    fn __add__(self: Tensor[Int], rhs: Float32)         -> Tensor[Float32]: return add_t(to_float32(self), scalar_f32(rhs))         # a(int→f32) + s(f32)
    fn __add__(self: Tensor[Int], rhs: Int)             -> Tensor[Int]:     return add_t(self, scalar_int(rhs))                   # a + s(int)

    # - (promotion)
    fn __sub__(self: Tensor[Int], rhs: Tensor[Float64]) -> Tensor[Float64]: return sub_t(to_float64(self), rhs)                     # a(int→f64) - b(f64)
    fn __sub__(self: Tensor[Int], rhs: Tensor[Float32]) -> Tensor[Float32]: return sub_t(to_float32(self), rhs)                    # a(int→f32) - b(f32)
    fn __sub__(self: Tensor[Int], rhs: Tensor[Int])     -> Tensor[Int]:     return sub_t(self, rhs)                                # a - b(int)
    fn __sub__(self: Tensor[Int], rhs: Float64)         -> Tensor[Float64]: return sub_t(to_float64(self), scalar_f64(rhs))         # a(int→f64) - s(f64)
    fn __sub__(self: Tensor[Int], rhs: Float32)         -> Tensor[Float32]: return sub_t(to_float32(self), scalar_f32(rhs))         # a(int→f32) - s(f32)
    fn __sub__(self: Tensor[Int], rhs: Int)             -> Tensor[Int]:     return sub_t(self, scalar_int(rhs))                   # a - s(int)

    # * (promotion)
    fn __mul__(self: Tensor[Int], rhs: Tensor[Float64]) -> Tensor[Float64]: return mul_t(to_float64(self), rhs)                     # a(int→f64) * b(f64)
    fn __mul__(self: Tensor[Int], rhs: Tensor[Float32]) -> Tensor[Float32]: return mul_t(to_float32(self), rhs)                    # a(int→f32) * b(f32)
    fn __mul__(self: Tensor[Int], rhs: Tensor[Int])     -> Tensor[Int]:     return mul_t(self, rhs)                                # a * b(int)
    fn __mul__(self: Tensor[Int], rhs: Float64)         -> Tensor[Float64]: return mul_t(to_float64(self), scalar_f64(rhs))         # a(int→f64) * s(f64)
    fn __mul__(self: Tensor[Int], rhs: Float32)         -> Tensor[Float32]: return mul_t(to_float32(self), scalar_f32(rhs))         # a(int→f32) * s(f32)
    fn __mul__(self: Tensor[Int], rhs: Int)             -> Tensor[Int]:     return mul_t(self, scalar_int(rhs))                   # a * s(int)

    # / (Float64 result)
    fn __truediv__(self: Tensor[Int], rhs: Tensor[Float64]) -> Tensor[Float64]: return div_t(to_float64(self), rhs)                # a(int→f64) / b(f64)
    fn __truediv__(self: Tensor[Int], rhs: Tensor[Float32]) -> Tensor[Float64]: return div_t(to_float64(self), to_float64(rhs))    # a(int→f64) / b(f32→f64)
    fn __truediv__(self: Tensor[Int], rhs: Tensor[Int])     -> Tensor[Float64]: return div_t(to_float64(self), to_float64(rhs))    # a(int→f64) / b(int→f64)
    fn __truediv__(self: Tensor[Int], rhs: Float64)         -> Tensor[Float64]: return div_t(to_float64(self), scalar_f64(rhs))     # a(int→f64) / s(f64)
    fn __truediv__(self: Tensor[Int], rhs: Float32)         -> Tensor[Float64]: return div_t(to_float64(self), to_float64(scalar_f32(rhs))) # a(int→f64) / s(f32→f64)
    fn __truediv__(self: Tensor[Int], rhs: Int)             -> Tensor[Float64]: return div_t(to_float64(self), to_float64(scalar_int(rhs))) # a(int→f64) / s(int→f64)

    # % (Float64 result)
    fn __mod__(self: Tensor[Int], rhs: Tensor[Float64]) -> Tensor[Float64]: return mod_t(to_float64(self), rhs)                    # a(int→f64) % b(f64)
    fn __mod__(self: Tensor[Int], rhs: Tensor[Float32]) -> Tensor[Float64]: return mod_t(to_float64(self), to_float64(rhs))        # a(int→f64) % b(f32→f64)
    fn __mod__(self: Tensor[Int], rhs: Tensor[Int])     -> Tensor[Float64]: return mod_t(to_float64(self), to_float64(rhs))        # a(int→f64) % b(int→f64)
    fn __mod__(self: Tensor[Int], rhs: Float64)         -> Tensor[Float64]: return mod_t(to_float64(self), scalar_f64(rhs))         # a(int→f64) % s(f64)
    fn __mod__(self: Tensor[Int], rhs: Float32)         -> Tensor[Float64]: return mod_t(to_float64(self), to_float64(scalar_f32(rhs))) # a(int→f64) % s(f32→f64)
    fn __mod__(self: Tensor[Int], rhs: Int)             -> Tensor[Float64]: return mod_t(to_float64(self), to_float64(scalar_int(rhs))) # a(int→f64) % s(int→f64)
    # =========================
    # Float64 — Reflected arithmetic (lhs ⊕ self)
    # =========================
    fn __radd__(self: Tensor[Float64], lhs: Float64) -> Tensor[Float64]: return add_t(scalar_f64(lhs), self)                         # s(f64) + a
    fn __radd__(self: Tensor[Float64], lhs: Float32) -> Tensor[Float64]: return add_t(to_float64(scalar_f32(lhs)), self)            # s(f32→f64) + a
    fn __radd__(self: Tensor[Float64], lhs: Int)     -> Tensor[Float64]: return add_t(to_float64(scalar_int(lhs)), self)          # s(int→f64) + a

    fn __rsub__(self: Tensor[Float64], lhs: Float64) -> Tensor[Float64]: return sub_t(scalar_f64(lhs), self)                         # s(f64) - a
    fn __rsub__(self: Tensor[Float64], lhs: Float32) -> Tensor[Float64]: return sub_t(to_float64(scalar_f32(lhs)), self)            # s(f32→f64) - a
    fn __rsub__(self: Tensor[Float64], lhs: Int)     -> Tensor[Float64]: return sub_t(to_float64(scalar_int(lhs)), self)          # s(int→f64) - a

    fn __rmul__(self: Tensor[Float64], lhs: Float64) -> Tensor[Float64]: return mul_t(scalar_f64(lhs), self)                         # s(f64) * a
    fn __rmul__(self: Tensor[Float64], lhs: Float32) -> Tensor[Float64]: return mul_t(to_float64(scalar_f32(lhs)), self)            # s(f32→f64) * a
    fn __rmul__(self: Tensor[Float64], lhs: Int)     -> Tensor[Float64]: return mul_t(to_float64(scalar_int(lhs)), self)          # s(int→f64) * a

    fn __rtruediv__(self: Tensor[Float64], lhs: Float64) -> Tensor[Float64]: return div_t(scalar_f64(lhs), self)                    # s(f64) / a
    fn __rtruediv__(self: Tensor[Float64], lhs: Float32) -> Tensor[Float64]: return div_t(to_float64(scalar_f32(lhs)), self)       # s(f32→f64) / a
    fn __rtruediv__(self: Tensor[Float64], lhs: Int)     -> Tensor[Float64]: return div_t(to_float64(scalar_int(lhs)), self)     # s(int→f64) / a

    fn __rmod__(self: Tensor[Float64], lhs: Float64) -> Tensor[Float64]: return mod_t(scalar_f64(lhs), self)                        # s(f64) % a
    fn __rmod__(self: Tensor[Float64], lhs: Float32) -> Tensor[Float64]: return mod_t(to_float64(scalar_f32(lhs)), self)           # s(f32→f64) % a
    fn __rmod__(self: Tensor[Float64], lhs: Int)     -> Tensor[Float64]: return mod_t(to_float64(scalar_int(lhs)), self)         # s(int→f64) % a


    # =========================
    # Float32 — Reflected arithmetic (lhs ⊕ self)
    # Note: for / and % we return Float64 (project convention).
    # =========================
    fn __radd__(self: Tensor[Float32], lhs: Float64) -> Tensor[Float64]: return add_t(scalar_f64(lhs), to_float64(self))            # s(f64) + a(f32→f64)
    fn __radd__(self: Tensor[Float32], lhs: Float32) -> Tensor[Float32]: return add_t(scalar_f32(lhs), self)                        # s(f32) + a
    fn __radd__(self: Tensor[Float32], lhs: Int)     -> Tensor[Float32]: return add_t(to_float32(scalar_int(lhs)), self)          # s(int→f32) + a

    fn __rsub__(self: Tensor[Float32], lhs: Float64) -> Tensor[Float64]: return sub_t(scalar_f64(lhs), to_float64(self))            # s(f64) - a(f32→f64)
    fn __rsub__(self: Tensor[Float32], lhs: Float32) -> Tensor[Float32]: return sub_t(scalar_f32(lhs), self)                        # s(f32) - a
    fn __rsub__(self: Tensor[Float32], lhs: Int)     -> Tensor[Float32]: return sub_t(to_float32(scalar_int(lhs)), self)          # s(int→f32) - a

    fn __rmul__(self: Tensor[Float32], lhs: Float64) -> Tensor[Float64]: return mul_t(scalar_f64(lhs), to_float64(self))            # s(f64) * a(f32→f64)
    fn __rmul__(self: Tensor[Float32], lhs: Float32) -> Tensor[Float32]: return mul_t(scalar_f32(lhs), self)                        # s(f32) * a
    fn __rmul__(self: Tensor[Float32], lhs: Int)     -> Tensor[Float32]: return mul_t(to_float32(scalar_int(lhs)), self)          # s(int→f32) * a

    fn __rtruediv__(self: Tensor[Float32], lhs: Float64) -> Tensor[Float64]: return div_t(scalar_f64(lhs), to_float64(self))        # s(f64) / a(f32→f64)
    fn __rtruediv__(self: Tensor[Float32], lhs: Float32) -> Tensor[Float64]: return div_t(to_float64(scalar_f32(lhs)), to_float64(self)) # s(f32→f64) / a(f32→f64)
    fn __rtruediv__(self: Tensor[Float32], lhs: Int)     -> Tensor[Float64]: return div_t(to_float64(scalar_int(lhs)), to_float64(self)) # s(int→f64) / a(f32→f64)

    fn __rmod__(self: Tensor[Float32], lhs: Float64) -> Tensor[Float64]: return mod_t(scalar_f64(lhs), to_float64(self))            # s(f64) % a(f32→f64)
    fn __rmod__(self: Tensor[Float32], lhs: Float32) -> Tensor[Float64]: return mod_t(to_float64(scalar_f32(lhs)), to_float64(self))# s(f32→f64) % a(f32→f64)
    fn __rmod__(self: Tensor[Float32], lhs: Int)     -> Tensor[Float64]: return mod_t(to_float64(scalar_int(lhs)), to_float64(self)) # s(int→f64) % a(f32→f64)


    # =========================
    # Int — Reflected arithmetic (lhs ⊕ self)
    # Note: for / and % we return Float64 (project convention).
    # =========================
    fn __radd__(self: Tensor[Int], lhs: Float64) -> Tensor[Float64]: return add_t(scalar_f64(lhs), to_float64(self))                # s(f64) + a(int→f64)
    fn __radd__(self: Tensor[Int], lhs: Float32) -> Tensor[Float32]: return add_t(scalar_f32(lhs), to_float32(self))                # s(f32) + a(int→f32)
    fn __radd__(self: Tensor[Int], lhs: Int)     -> Tensor[Int]:     return add_t(scalar_int(lhs), self)                          # s(int) + a

    fn __rsub__(self: Tensor[Int], lhs: Float64) -> Tensor[Float64]: return sub_t(scalar_f64(lhs), to_float64(self))                # s(f64) - a(int→f64)
    fn __rsub__(self: Tensor[Int], lhs: Float32) -> Tensor[Float32]: return sub_t(scalar_f32(lhs), to_float32(self))                # s(f32) - a(int→f32)
    fn __rsub__(self: Tensor[Int], lhs: Int)     -> Tensor[Int]:     return sub_t(scalar_int(lhs), self)                          # s(int) - a

    fn __rmul__(self: Tensor[Int], lhs: Float64) -> Tensor[Float64]: return mul_t(scalar_f64(lhs), to_float64(self))                # s(f64) * a(int→f64)
    fn __rmul__(self: Tensor[Int], lhs: Float32) -> Tensor[Float32]: return mul_t(scalar_f32(lhs), to_float32(self))                # s(f32) * a(int→f32)
    fn __rmul__(self: Tensor[Int], lhs: Int)     -> Tensor[Int]:     return mul_t(scalar_int(lhs), self)                          # s(int) * a

    fn __rtruediv__(self: Tensor[Int], lhs: Float64) -> Tensor[Float64]: return div_t(scalar_f64(lhs), to_float64(self))            # s(f64) / a(int→f64)
    fn __rtruediv__(self: Tensor[Int], lhs: Float32) -> Tensor[Float64]: return div_t(to_float64(scalar_f32(lhs)), to_float64(self))# s(f32→f64) / a(int→f64)
    fn __rtruediv__(self: Tensor[Int], lhs: Int)     -> Tensor[Float64]: return div_t(to_float64(scalar_int(lhs)), to_float64(self)) # s(int→f64) / a(int→f64)

    fn __rmod__(self: Tensor[Int], lhs: Float64) -> Tensor[Float64]: return mod_t(scalar_f64(lhs), to_float64(self))                # s(f64) % a(int→f64)
    fn __rmod__(self: Tensor[Int], lhs: Float32) -> Tensor[Float64]: return mod_t(to_float64(scalar_f32(lhs)), to_float64(self))    # s(f32→f64) % a(int→f64)
    fn __rmod__(self: Tensor[Int], lhs: Int)     -> Tensor[Float64]: return mod_t(to_float64(scalar_int(lhs)), to_float64(self))  # s(int→f64) % a(int→f64)
    # =========================
    # In-place arithmetic — Float64 (full scalar combos)
    # Promotion policy for in-place: keep self dtype (convert rhs up to Float64)
    # =========================
    fn __iadd__(mut self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Float64]: self = add_t(self, rhs); return self                           # a += b
    fn __iadd__(mut self: Tensor[Float64], rhs: Float64)        -> Tensor[Float64]: self = add_t(self, scalar_f64(rhs)); return self                  # a += s(f64)
    fn __iadd__(mut self: Tensor[Float64], rhs: Float32)        -> Tensor[Float64]: self = add_t(self, to_float64(scalar_f32(rhs))); return self      # a += s(f32→f64)
    fn __iadd__(mut self: Tensor[Float64], rhs: Int)            -> Tensor[Float64]: self = add_t(self, to_float64(scalar_int(rhs))); return self    # a += s(int→f64)

    fn __isub__(mut self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Float64]: self = sub_t(self, rhs); return self                           # a -= b
    fn __isub__(mut self: Tensor[Float64], rhs: Float64)        -> Tensor[Float64]: self = sub_t(self, scalar_f64(rhs)); return self                  # a -= s(f64)
    fn __isub__(mut self: Tensor[Float64], rhs: Float32)        -> Tensor[Float64]: self = sub_t(self, to_float64(scalar_f32(rhs))); return self      # a -= s(f32→f64)
    fn __isub__(mut self: Tensor[Float64], rhs: Int)            -> Tensor[Float64]: self = sub_t(self, to_float64(scalar_int(rhs))); return self    # a -= s(int→f64)

    fn __imul__(mut self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Float64]: self = mul_t(self, rhs); return self                           # a *= b
    fn __imul__(mut self: Tensor[Float64], rhs: Float64)        -> Tensor[Float64]: self = mul_t(self, scalar_f64(rhs)); return self                  # a *= s(f64)
    fn __imul__(mut self: Tensor[Float64], rhs: Float32)        -> Tensor[Float64]: self = mul_t(self, to_float64(scalar_f32(rhs))); return self      # a *= s(f32→f64)
    fn __imul__(mut self: Tensor[Float64], rhs: Int)            -> Tensor[Float64]: self = mul_t(self, to_float64(scalar_int(rhs))); return self    # a *= s(int→f64)

    fn __itruediv__(mut self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Float64]: self = div_t(self, rhs); return self                       # a /= b
    fn __itruediv__(mut self: Tensor[Float64], rhs: Float64)        -> Tensor[Float64]: self = div_t(self, scalar_f64(rhs)); return self              # a /= s(f64)
    fn __itruediv__(mut self: Tensor[Float64], rhs: Float32)        -> Tensor[Float64]: self = div_t(self, to_float64(scalar_f32(rhs))); return self  # a /= s(f32→f64)
    fn __itruediv__(mut self: Tensor[Float64], rhs: Int)            -> Tensor[Float64]: self = div_t(self, to_float64(scalar_int(rhs))); return self# a /= s(int→f64)

    fn __imod__(mut self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Float64]: self = mod_t(self, rhs); return self                           # a %= b
    fn __imod__(mut self: Tensor[Float64], rhs: Float64)        -> Tensor[Float64]: self = mod_t(self, scalar_f64(rhs)); return self                  # a %= s(f64)
    fn __imod__(mut self: Tensor[Float64], rhs: Float32)        -> Tensor[Float64]: self = mod_t(self, to_float64(scalar_f32(rhs))); return self      # a %= s(f32→f64)
    fn __imod__(mut self: Tensor[Float64], rhs: Int)            -> Tensor[Float64]: self = mod_t(self, to_float64(scalar_int(rhs))); return self    # a %= s(int→f64)


    # =========================
    # In-place arithmetic — Float32 (full scalar combos)
    # Promotion policy for in-place: keep self dtype (convert rhs up to Float32)
    # =========================
    fn __iadd__(mut self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Float32]: self = add_t(self, rhs); return self                           # a += b
    fn __iadd__(mut self: Tensor[Float32], rhs: Float32)        -> Tensor[Float32]: self = add_t(self, scalar_f32(rhs)); return self                  # a += s(f32)
    fn __iadd__(mut self: Tensor[Float32], rhs: Float64)        -> Tensor[Float32]: self = add_t(self, to_float32(scalar_f64(rhs))); return self      # a += s(f64→f32)
    fn __iadd__(mut self: Tensor[Float32], rhs: Int)            -> Tensor[Float32]: self = add_t(self, to_float32(scalar_int(rhs))); return self    # a += s(int→f32)

    fn __isub__(mut self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Float32]: self = sub_t(self, rhs); return self                           # a -= b
    fn __isub__(mut self: Tensor[Float32], rhs: Float32)        -> Tensor[Float32]: self = sub_t(self, scalar_f32(rhs)); return self                  # a -= s(f32)
    fn __isub__(mut self: Tensor[Float32], rhs: Float64)        -> Tensor[Float32]: self = sub_t(self, to_float32(scalar_f64(rhs))); return self      # a -= s(f64→f32)
    fn __isub__(mut self: Tensor[Float32], rhs: Int)            -> Tensor[Float32]: self = sub_t(self, to_float32(scalar_int(rhs))); return self    # a -= s(int→f32)

    fn __imul__(mut self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Float32]: self = mul_t(self, rhs); return self                           # a *= b
    fn __imul__(mut self: Tensor[Float32], rhs: Float32)        -> Tensor[Float32]: self = mul_t(self, scalar_f32(rhs)); return self                  # a *= s(f32)
    fn __imul__(mut self: Tensor[Float32], rhs: Float64)        -> Tensor[Float32]: self = mul_t(self, to_float32(scalar_f64(rhs))); return self      # a *= s(f64→f32)
    fn __imul__(mut self: Tensor[Float32], rhs: Int)            -> Tensor[Float32]: self = mul_t(self, to_float32(scalar_int(rhs))); return self    # a *= s(int→f32)

    fn __itruediv__(mut self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Float32]: self = div_t(self, rhs); return self                       # a /= b
    fn __itruediv__(mut self: Tensor[Float32], rhs: Float32)        -> Tensor[Float32]: self = div_t(self, scalar_f32(rhs)); return self              # a /= s(f32)
    fn __itruediv__(mut self: Tensor[Float32], rhs: Float64)        -> Tensor[Float32]: self = div_t(self, to_float32(scalar_f64(rhs))); return self  # a /= s(f64→f32)
    fn __itruediv__(mut self: Tensor[Float32], rhs: Int)            -> Tensor[Float32]: self = div_t(self, to_float32(scalar_int(rhs))); return self# a /= s(int→f32)

    fn __imod__(mut self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Float32]: self = mod_t(self, rhs); return self                           # a %= b
    fn __imod__(mut self: Tensor[Float32], rhs: Float32)        -> Tensor[Float32]: self = mod_t(self, scalar_f32(rhs)); return self                  # a %= s(f32)
    fn __imod__(mut self: Tensor[Float32], rhs: Float64)        -> Tensor[Float32]: self = mod_t(self, to_float32(scalar_f64(rhs))); return self      # a %= s(f64→f32)
    fn __imod__(mut self: Tensor[Float32], rhs: Int)            -> Tensor[Float32]: self = mod_t(self, to_float32(scalar_int(rhs))); return self    # a %= s(int→f32)


    # =========================
    # In-place arithmetic — Int (full scalar combos)
    # Promotion policy for in-place: keep self dtype (convert rhs up to Int)
    # NOTE: true-division/mod on Int keeps Int here (uses div_t/mod_t for Int domain)
    # =========================
    fn __iadd__(mut self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]: self = add_t(self, rhs); return self                                 # a += b
    fn __iadd__(mut self: Tensor[Int], rhs: Int)        -> Tensor[Int]: self = add_t(self, scalar_int(rhs)); return self                     # a += s(int)
    fn __iadd__(mut self: Tensor[Int], rhs: Float32)    -> Tensor[Int]: self = add_t(self, to_int(scalar_f32(rhs))); return self               # a += s(f32→int)
    fn __iadd__(mut self: Tensor[Int], rhs: Float64)    -> Tensor[Int]: self = add_t(self, to_int(scalar_f64(rhs))); return self               # a += s(f64→int)

    fn __isub__(mut self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]: self = sub_t(self, rhs); return self                                 # a -= b
    fn __isub__(mut self: Tensor[Int], rhs: Int)        -> Tensor[Int]: self = sub_t(self, scalar_int(rhs)); return self                     # a -= s(int)
    fn __isub__(mut self: Tensor[Int], rhs: Float32)    -> Tensor[Int]: self = sub_t(self, to_int(scalar_f32(rhs))); return self               # a -= s(f32→int)
    fn __isub__(mut self: Tensor[Int], rhs: Float64)    -> Tensor[Int]: self = sub_t(self, to_int(scalar_f64(rhs))); return self               # a -= s(f64→int)

    fn __imul__(mut self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]: self = mul_t(self, rhs); return self                                 # a *= b
    fn __imul__(mut self: Tensor[Int], rhs: Int)        -> Tensor[Int]: self = mul_t(self, scalar_int(rhs)); return self                     # a *= s(int)
    fn __imul__(mut self: Tensor[Int], rhs: Float32)    -> Tensor[Int]: self = mul_t(self, to_int(scalar_f32(rhs))); return self               # a *= s(f32→int)
    fn __imul__(mut self: Tensor[Int], rhs: Float64)    -> Tensor[Int]: self = mul_t(self, to_int(scalar_f64(rhs))); return self               # a *= s(f64→int)

    fn __itruediv__(mut self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]: self = div_t(self, rhs); return self                             # a /= b (int domain)
    fn __itruediv__(mut self: Tensor[Int], rhs: Int)        -> Tensor[Int]: self = div_t(self, scalar_int(rhs)); return self                 # a /= s(int)
    fn __itruediv__(mut self: Tensor[Int], rhs: Float32)    -> Tensor[Int]: self = div_t(self, to_int(scalar_f32(rhs))); return self           # a /= s(f32→int)
    fn __itruediv__(mut self: Tensor[Int], rhs: Float64)    -> Tensor[Int]: self = div_t(self, to_int(scalar_f64(rhs))); return self           # a /= s(f64→int)

    fn __imod__(mut self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]: self = mod_t(self, rhs); return self                                 # a %= b (int domain)
    fn __imod__(mut self: Tensor[Int], rhs: Int)        -> Tensor[Int]: self = mod_t(self, scalar_int(rhs)); return self                     # a %= s(int)
    fn __imod__(mut self: Tensor[Int], rhs: Float32)    -> Tensor[Int]: self = mod_t(self, to_int(scalar_f32(rhs))); return self               # a %= s(f32→int)
    fn __imod__(mut self: Tensor[Int], rhs: Float64)    -> Tensor[Int]: self = mod_t(self, to_int(scalar_f64(rhs))); return self               # a %= s(f64→int)
    # =========================
    # Power overloads — full combos, promotion Int < Float32 < Float64
    # Uses only: pow_t, to_int, to_float32, to_float64, scalar_int/scalar_f32/scalar_f64
    # =========================

    # -------------------------
    # Tensor[Float64]
    # -------------------------
    fn __pow__(self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Float64]: return pow_t(self, rhs)                                         # a(f64) ** b(f64)
    fn __pow__(self: Tensor[Float64], rhs: Float64)        -> Tensor[Float64]: return pow_t(self, scalar_f64(rhs))                              # a(f64) ** s(f64)
    fn __pow__(self: Tensor[Float64], rhs: Float32)        -> Tensor[Float64]: return pow_t(self, to_float64(scalar_f32(rhs)))                  # a(f64) ** s(f32→f64)
    fn __pow__(self: Tensor[Float64], rhs: Int)            -> Tensor[Float64]: return pow_t(self, to_float64(scalar_int(rhs)))                # a(f64) ** s(int→f64)

    fn __rpow__(self: Tensor[Float64], lhs: Float64)       -> Tensor[Float64]: return pow_t(scalar_f64(lhs), self)                              # s(f64) ** a(f64)
    fn __rpow__(self: Tensor[Float64], lhs: Float32)       -> Tensor[Float64]: return pow_t(to_float64(scalar_f32(lhs)), self)                  # s(f32→f64) ** a(f64)
    fn __rpow__(self: Tensor[Float64], lhs: Int)           -> Tensor[Float64]: return pow_t(to_float64(scalar_int(lhs)), self)                # s(int→f64) ** a(f64)

    fn __ipow__(mut self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Float64]: self = pow_t(self, rhs); return self                      # a **= b(f64)
    fn __ipow__(mut self: Tensor[Float64], rhs: Float64)        -> Tensor[Float64]: self = pow_t(self, scalar_f64(rhs)); return self            # a **= s(f64)
    fn __ipow__(mut self: Tensor[Float64], rhs: Float32)        -> Tensor[Float64]: self = pow_t(self, to_float64(scalar_f32(rhs))); return self# a **= s(f32→f64)
    fn __ipow__(mut self: Tensor[Float64], rhs: Int)            -> Tensor[Float64]: self = pow_t(self, to_float64(scalar_int(rhs))); return self# a **= s(int→f64)

    # -------------------------
    # Tensor[Float32]
    # -------------------------
    # result Float64 if rhs/lhs is Float64 or Tensor[Float64]
    fn __pow__(self: Tensor[Float32], rhs: Tensor[Float64]) -> Tensor[Float64]: return pow_t(to_float64(self), rhs)                            # a(f32→f64) ** b(f64)
    fn __pow__(self: Tensor[Float32], rhs: Float64)        -> Tensor[Float64]: return pow_t(to_float64(self), scalar_f64(rhs))                 # a(f32→f64) ** s(f64)
    fn __rpow__(self: Tensor[Float32], lhs: Float64)       -> Tensor[Float64]: return pow_t(scalar_f64(lhs), to_float64(self))                 # s(f64) ** a(f32→f64)

    # result Float32 otherwise
    fn __pow__(self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Float32]: return pow_t(self, rhs)                                       # a(f32) ** b(f32)
    fn __pow__(self: Tensor[Float32], rhs: Float32)        -> Tensor[Float32]: return pow_t(self, scalar_f32(rhs))                             # a(f32) ** s(f32)
    fn __pow__(self: Tensor[Float32], rhs: Int)            -> Tensor[Float32]: return pow_t(self, to_float32(scalar_int(rhs)))               # a(f32) ** s(int→f32)

    fn __rpow__(self: Tensor[Float32], lhs: Float32)       -> Tensor[Float32]: return pow_t(scalar_f32(lhs), self)                             # s(f32) ** a(f32)
    fn __rpow__(self: Tensor[Float32], lhs: Int)           -> Tensor[Float32]: return pow_t(to_float32(scalar_int(lhs)), self)               # s(int→f32) ** a(f32)

    # in-place only when result dtype remains Float32
    fn __ipow__(mut self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Float32]: self = pow_t(self, rhs); return self                     # a **= b(f32)
    fn __ipow__(mut self: Tensor[Float32], rhs: Float32)        -> Tensor[Float32]: self = pow_t(self, scalar_f32(rhs)); return self           # a **= s(f32)
    fn __ipow__(mut self: Tensor[Float32], rhs: Int)            -> Tensor[Float32]: self = pow_t(self, to_float32(scalar_int(rhs))); return self # a **= s(int→f32)

    # -------------------------
    # Tensor[Int]
    # -------------------------
    # result Float64 if rhs/lhs is Float64 or Tensor[Float64]
    fn __pow__(self: Tensor[Int], rhs: Tensor[Float64]) -> Tensor[Float64]: return pow_t(to_float64(self), rhs)                                # a(int→f64) ** b(f64)
    fn __pow__(self: Tensor[Int], rhs: Float64)        -> Tensor[Float64]: return pow_t(to_float64(self), scalar_f64(rhs))                     # a(int→f64) ** s(f64)
    fn __rpow__(self: Tensor[Int], lhs: Float64)       -> Tensor[Float64]: return pow_t(scalar_f64(lhs), to_float64(self))                     # s(f64) ** a(int→f64)

    # result Float32 if no Float64 present but Float32/ Tensor[Float32] present
    fn __pow__(self: Tensor[Int], rhs: Tensor[Float32]) -> Tensor[Float32]: return pow_t(to_float32(self), rhs)                               # a(int→f32) ** b(f32)
    fn __pow__(self: Tensor[Int], rhs: Float32)        -> Tensor[Float32]: return pow_t(to_float32(self), scalar_f32(rhs))                    # a(int→f32) ** s(f32)
    fn __rpow__(self: Tensor[Int], lhs: Float32)       -> Tensor[Float32]: return pow_t(scalar_f32(lhs), to_float32(self))                    # s(f32) ** a(int→f32)

    # result Int when both are Int
    fn __pow__(self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]: return pow_t(self, rhs)                                                   # a(int) ** b(int)
    fn __pow__(self: Tensor[Int], rhs: Int)        -> Tensor[Int]: return pow_t(self, scalar_int(rhs))                                       # a(int) ** s(int)
    fn __rpow__(self: Tensor[Int], lhs: Int)       -> Tensor[Int]: return pow_t(scalar_int(lhs), self)                                       # s(int) ** a(int)

    # in-place only when result dtype remains Int
    fn __ipow__(mut self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]: self = pow_t(self, rhs); return self                                 # a **= b(int)
    fn __ipow__(mut self: Tensor[Int], rhs: Int)        -> Tensor[Int]: self = pow_t(self, scalar_int(rhs)); return self                     # a **= s(int)
    # =========================
    # Float64 unary overloads (one-liners)
    # =========================
    fn __neg__(self: Tensor[Float64]) -> Tensor[Float64]: return neg_t(self)     # -a
    fn __pos__(self: Tensor[Float64]) -> Tensor[Float64]: return pos_t(self)     # +a
    fn __abs__(self: Tensor[Float64]) -> Tensor[Float64]: return abs_t(self)     # abs(a)
    fn __invert__(self: Tensor[Float64]) -> Tensor[Int]:   return not_t(self)    # ~a (logical NOT → Int mask)

    # =========================
    # Float32 unary overloads (one-liners)
    # =========================
    fn __neg__(self: Tensor[Float32]) -> Tensor[Float32]: return neg_t(self)     # -a
    fn __pos__(self: Tensor[Float32]) -> Tensor[Float32]: return pos_t(self)     # +a
    fn __abs__(self: Tensor[Float32]) -> Tensor[Float32]: return abs_t(self)     # abs(a)
    fn __invert__(self: Tensor[Float32]) -> Tensor[Int]:  return not_t(self)     # ~a (logical NOT → Int mask)

    # =========================
    # Int unary overloads (one-liners)
    # =========================
    fn __neg__(self: Tensor[Int]) -> Tensor[Int]: return neg_t(self)             # -a
    fn __pos__(self: Tensor[Int]) -> Tensor[Int]: return pos_t(self)             # +a
    fn __abs__(self: Tensor[Int]) -> Tensor[Int]: return abs_t(self)             # abs(a)
    fn __invert__(self: Tensor[Int]) -> Tensor[Int]: return not_t(self)          # ~a (logical NOT → Int mask)

    # =========================
    # Float64 logical overloads (elementwise → Tensor[Int] mask)
    # =========================
    fn __and__(self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Int]: return and_t(self, rhs)                                 # a & b (f64,f64)
    fn __and__(self: Tensor[Float64], rhs: Tensor[Float32]) -> Tensor[Int]: return and_t(self, to_float64(rhs))                    # a & b (f64,f32→f64)
    fn __and__(self: Tensor[Float64], rhs: Tensor[Int])     -> Tensor[Int]: return and_t(self, to_float64(rhs))                    # a & b (f64,int→f64)
    fn __and__(self: Tensor[Float64], rhs: Float64)         -> Tensor[Int]: return and_t(self, scalar_f64(rhs))                     # a & s (f64)
    fn __and__(self: Tensor[Float64], rhs: Float32)         -> Tensor[Int]: return and_t(self, to_float64(scalar_f32(rhs)))         # a & s (f32→f64)
    fn __and__(self: Tensor[Float64], rhs: Int)             -> Tensor[Int]: return and_t(self, to_float64(scalar_int(rhs)))       # a & s (int→f64)

    fn __or__( self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Int]: return or_t (self, rhs)                                # a | b (f64,f64)
    fn __or__( self: Tensor[Float64], rhs: Tensor[Float32]) -> Tensor[Int]: return or_t (self, to_float64(rhs))                    # a | b (f64,f32→f64)
    fn __or__( self: Tensor[Float64], rhs: Tensor[Int])     -> Tensor[Int]: return or_t (self, to_float64(rhs))                    # a | b (f64,int→f64)
    fn __or__( self: Tensor[Float64], rhs: Float64)         -> Tensor[Int]: return or_t (self, scalar_f64(rhs))                     # a | s (f64)
    fn __or__( self: Tensor[Float64], rhs: Float32)         -> Tensor[Int]: return or_t (self, to_float64(scalar_f32(rhs)))         # a | s (f32→f64)
    fn __or__( self: Tensor[Float64], rhs: Int)             -> Tensor[Int]: return or_t (self, to_float64(scalar_int(rhs)))       # a | s (int→f64)

    fn __xor__(self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Int]: return xor_t(self, rhs)                                # a ^ b (f64,f64)
    fn __xor__(self: Tensor[Float64], rhs: Tensor[Float32]) -> Tensor[Int]: return xor_t(self, to_float64(rhs))                    # a ^ b (f64,f32→f64)
    fn __xor__(self: Tensor[Float64], rhs: Tensor[Int])     -> Tensor[Int]: return xor_t(self, to_float64(rhs))                    # a ^ b (f64,int→f64)
    fn __xor__(self: Tensor[Float64], rhs: Float64)         -> Tensor[Int]: return xor_t(self, scalar_f64(rhs))                     # a ^ s (f64)
    fn __xor__(self: Tensor[Float64], rhs: Float32)         -> Tensor[Int]: return xor_t(self, to_float64(scalar_f32(rhs)))         # a ^ s (f32→f64)
    fn __xor__(self: Tensor[Float64], rhs: Int)             -> Tensor[Int]: return xor_t(self, to_float64(scalar_int(rhs)))       # a ^ s (int→f64)
                                                 # ~a


    # =========================
    # Float32 logical overloads (elementwise → Tensor[Int] mask)
    # =========================
    fn __and__(self: Tensor[Float32], rhs: Tensor[Float64]) -> Tensor[Int]: return and_t(to_float64(self), rhs)                    # a(f32→f64) & b(f64)
    fn __and__(self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Int]: return and_t(self, rhs)                                # a & b (f32,f32)
    fn __and__(self: Tensor[Float32], rhs: Tensor[Int])     -> Tensor[Int]: return and_t(self, to_float32(rhs))                    # a & b (f32,int→f32)
    fn __and__(self: Tensor[Float32], rhs: Float64)         -> Tensor[Int]: return and_t(to_float64(self), scalar_f64(rhs))         # a(f32→f64) & s(f64)
    fn __and__(self: Tensor[Float32], rhs: Float32)         -> Tensor[Int]: return and_t(self, scalar_f32(rhs))                     # a & s(f32)
    fn __and__(self: Tensor[Float32], rhs: Int)             -> Tensor[Int]: return and_t(self, to_float32(scalar_int(rhs)))       # a & s(int→f32)

    fn __or__( self: Tensor[Float32], rhs: Tensor[Float64]) -> Tensor[Int]: return or_t (to_float64(self), rhs)                    # a(f32→f64) | b(f64)
    fn __or__( self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Int]: return or_t (self, rhs)                                # a | b (f32,f32)
    fn __or__( self: Tensor[Float32], rhs: Tensor[Int])     -> Tensor[Int]: return or_t (self, to_float32(rhs))                    # a | b (f32,int→f32)
    fn __or__( self: Tensor[Float32], rhs: Float64)         -> Tensor[Int]: return or_t (to_float64(self), scalar_f64(rhs))         # a(f32→f64) | s(f64)
    fn __or__( self: Tensor[Float32], rhs: Float32)         -> Tensor[Int]: return or_t (self, scalar_f32(rhs))                     # a | s(f32)
    fn __or__( self: Tensor[Float32], rhs: Int)             -> Tensor[Int]: return or_t (self, to_float32(scalar_int(rhs)))       # a | s(int→f32)

    fn __xor__(self: Tensor[Float32], rhs: Tensor[Float64]) -> Tensor[Int]: return xor_t(to_float64(self), rhs)                    # a(f32→f64) ^ b(f64)
    fn __xor__(self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Int]: return xor_t(self, rhs)                                # a ^ b (f32,f32)
    fn __xor__(self: Tensor[Float32], rhs: Tensor[Int])     -> Tensor[Int]: return xor_t(self, to_float32(rhs))                    # a ^ b (f32,int→f32)
    fn __xor__(self: Tensor[Float32], rhs: Float64)         -> Tensor[Int]: return xor_t(to_float64(self), scalar_f64(rhs))         # a(f32→f64) ^ s(f64)
    fn __xor__(self: Tensor[Float32], rhs: Float32)         -> Tensor[Int]: return xor_t(self, scalar_f32(rhs))                     # a ^ s(f32)
    fn __xor__(self: Tensor[Float32], rhs: Int)             -> Tensor[Int]: return xor_t(self, to_float32(scalar_int(rhs)))       # a ^ s(int→f32)


    # =========================
    # Int logical overloads (elementwise → Tensor[Int] mask)
    # =========================
    fn __and__(self: Tensor[Int], rhs: Tensor[Float64]) -> Tensor[Int]: return and_t(to_float64(self), rhs)                        # a(int→f64) & b(f64)
    fn __and__(self: Tensor[Int], rhs: Tensor[Float32]) -> Tensor[Int]: return and_t(to_float32(self), rhs)                        # a(int→f32) & b(f32)
    fn __and__(self: Tensor[Int], rhs: Tensor[Int])     -> Tensor[Int]: return and_t(self, rhs)                                    # a & b (int,int)
    fn __and__(self: Tensor[Int], rhs: Float64)         -> Tensor[Int]: return and_t(to_float64(self), scalar_f64(rhs))             # a(int→f64) & s(f64)
    fn __and__(self: Tensor[Int], rhs: Float32)         -> Tensor[Int]: return and_t(to_float32(self), scalar_f32(rhs))             # a(int→f32) & s(f32)
    fn __and__(self: Tensor[Int], rhs: Int)             -> Tensor[Int]: return and_t(self, scalar_int(rhs))                       # a & s(int)

    fn __or__( self: Tensor[Int], rhs: Tensor[Float64]) -> Tensor[Int]: return or_t (to_float64(self), rhs)                        # a(int→f64) | b(f64)
    fn __or__( self: Tensor[Int], rhs: Tensor[Float32]) -> Tensor[Int]: return or_t (to_float32(self), rhs)                        # a(int→f32) | b(f32)
    fn __or__( self: Tensor[Int], rhs: Tensor[Int])     -> Tensor[Int]: return or_t (self, rhs)                                    # a | b (int,int)
    fn __or__( self: Tensor[Int], rhs: Float64)         -> Tensor[Int]: return or_t (to_float64(self), scalar_f64(rhs))             # a(int→f64) | s(f64)
    fn __or__( self: Tensor[Int], rhs: Float32)         -> Tensor[Int]: return or_t (to_float32(self), scalar_f32(rhs))             # a(int→f32) | s(f32)
    fn __or__( self: Tensor[Int], rhs: Int)             -> Tensor[Int]: return or_t (self, scalar_int(rhs))                       # a | s(int)

    fn __xor__(self: Tensor[Int], rhs: Tensor[Float64]) -> Tensor[Int]: return xor_t(to_float64(self), rhs)                        # a(int→f64) ^ b(f64)
    fn __xor__(self: Tensor[Int], rhs: Tensor[Float32]) -> Tensor[Int]: return xor_t(to_float32(self), rhs)                        # a(int→f32) ^ b(f32)
    fn __xor__(self: Tensor[Int], rhs: Tensor[Int])     -> Tensor[Int]: return xor_t(self, rhs)                                    # a ^ b (int,int)
    fn __xor__(self: Tensor[Int], rhs: Float64)         -> Tensor[Int]: return xor_t(to_float64(self), scalar_f64(rhs))             # a(int→f64) ^ s(f64)
    fn __xor__(self: Tensor[Int], rhs: Float32)         -> Tensor[Int]: return xor_t(to_float32(self), scalar_f32(rhs))             # a(int→f32) ^ s(f32)
    fn __xor__(self: Tensor[Int], rhs: Int)             -> Tensor[Int]: return xor_t(self, scalar_int(rhs))                       # a ^ s(int)

    # =========================
    # Float64 reflected logical (result mask: Tensor[Int])
    # =========================
    fn __rand__(self: Tensor[Float64], lhs: Float64) -> Tensor[Int]: return and_t(scalar_f64(lhs), self)                         # s(f64) & a(f64)
    fn __rand__(self: Tensor[Float64], lhs: Float32) -> Tensor[Int]: return and_t(to_float64(scalar_f32(lhs)), self)            # s(f32→f64) & a(f64)
    fn __rand__(self: Tensor[Float64], lhs: Int)     -> Tensor[Int]: return and_t(to_float64(scalar_int(lhs)), self)          # s(int→f64) & a(f64)

    fn __ror__( self: Tensor[Float64], lhs: Float64) -> Tensor[Int]: return or_t (scalar_f64(lhs), self)                        # s(f64) | a(f64)
    fn __ror__( self: Tensor[Float64], lhs: Float32) -> Tensor[Int]: return or_t (to_float64(scalar_f32(lhs)), self)            # s(f32→f64) | a(f64)
    fn __ror__( self: Tensor[Float64], lhs: Int)     -> Tensor[Int]: return or_t (to_float64(scalar_int(lhs)), self)          # s(int→f64) | a(f64)

    fn __rxor__(self: Tensor[Float64], lhs: Float64) -> Tensor[Int]: return xor_t(scalar_f64(lhs), self)                        # s(f64) ^ a(f64)
    fn __rxor__(self: Tensor[Float64], lhs: Float32) -> Tensor[Int]: return xor_t(to_float64(scalar_f32(lhs)), self)            # s(f32→f64) ^ a(f64)
    fn __rxor__(self: Tensor[Float64], lhs: Int)     -> Tensor[Int]: return xor_t(to_float64(scalar_int(lhs)), self)          # s(int→f64) ^ a(f64)


    # =========================
    # Float32 reflected logical (promotion: if lhs is Float64 ⇒ promote self to Float64)
    # =========================
    fn __rand__(self: Tensor[Float32], lhs: Float64) -> Tensor[Int]: return and_t(to_float64(scalar_f64(lhs)), to_float64(self))# s(f64) & a(f32→f64)
    fn __rand__(self: Tensor[Float32], lhs: Float32) -> Tensor[Int]: return and_t(scalar_f32(lhs), self)                        # s(f32) & a(f32)
    fn __rand__(self: Tensor[Float32], lhs: Int)     -> Tensor[Int]: return and_t(to_float32(scalar_int(lhs)), self)          # s(int→f32) & a(f32)

    fn __ror__( self: Tensor[Float32], lhs: Float64) -> Tensor[Int]: return or_t (to_float64(scalar_f64(lhs)), to_float64(self))# s(f64) | a(f32→f64)
    fn __ror__( self: Tensor[Float32], lhs: Float32) -> Tensor[Int]: return or_t (scalar_f32(lhs), self)                        # s(f32) | a(f32)
    fn __ror__( self: Tensor[Float32], lhs: Int)     -> Tensor[Int]: return or_t (to_float32(scalar_int(lhs)), self)          # s(int→f32) | a(f32)

    fn __rxor__(self: Tensor[Float32], lhs: Float64) -> Tensor[Int]: return xor_t(to_float64(scalar_f64(lhs)), to_float64(self))# s(f64) ^ a(f32→f64)
    fn __rxor__(self: Tensor[Float32], lhs: Float32) -> Tensor[Int]: return xor_t(scalar_f32(lhs), self)                        # s(f32) ^ a(f32)
    fn __rxor__(self: Tensor[Float32], lhs: Int)     -> Tensor[Int]: return xor_t(to_float32(scalar_int(lhs)), self)          # s(int→f32) ^ a(f32)


    # =========================
    # Int reflected logical (promotion: lhs Float64 ⇒ f64; lhs Float32 ⇒ f32; else Int)
    # =========================
    fn __rand__(self: Tensor[Int], lhs: Float64) -> Tensor[Int]: return and_t(to_float64(scalar_f64(lhs)), to_float64(self))    # s(f64) & a(int→f64)
    fn __rand__(self: Tensor[Int], lhs: Float32) -> Tensor[Int]: return and_t(to_float32(scalar_f32(lhs)), to_float32(self))    # s(f32) & a(int→f32)
    fn __rand__(self: Tensor[Int], lhs: Int)     -> Tensor[Int]: return and_t(scalar_int(lhs), self)                          # s(int) & a(int)

    fn __ror__( self: Tensor[Int], lhs: Float64) -> Tensor[Int]: return or_t (to_float64(scalar_f64(lhs)), to_float64(self))    # s(f64) | a(int→f64)
    fn __ror__( self: Tensor[Int], lhs: Float32) -> Tensor[Int]: return or_t (to_float32(scalar_f32(lhs)), to_float32(self))    # s(f32) | a(int→f32)
    fn __ror__( self: Tensor[Int], lhs: Int)     -> Tensor[Int]: return or_t (scalar_int(lhs), self)                          # s(int) | a(int)

    fn __rxor__(self: Tensor[Int], lhs: Float64) -> Tensor[Int]: return xor_t(to_float64(scalar_f64(lhs)), to_float64(self))    # s(f64) ^ a(int→f64)
    fn __rxor__(self: Tensor[Int], lhs: Float32) -> Tensor[Int]: return xor_t(to_float32(scalar_f32(lhs)), to_float32(self))    # s(f32) ^ a(int→f32)
    fn __rxor__(self: Tensor[Int], lhs: Int)     -> Tensor[Int]: return xor_t(scalar_int(lhs), self)                          # s(int) ^ a(int)

    # =========================
    # Float64 comparisons → mask(Int)
    # Promotion: Int < Float32 < Float64
    # =========================
    fn __lt__(self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Int]: return lt_t(self, rhs)                               # a < b
    fn __lt__(self: Tensor[Float64], rhs: Float64)        -> Tensor[Int]: return lt_t(self, scalar_f64(rhs))                    # a < s(f64)
    fn __lt__(self: Tensor[Float64], rhs: Float32)        -> Tensor[Int]: return lt_t(self, to_float64(scalar_f32(rhs)))        # a < s(f32→f64)
    fn __lt__(self: Tensor[Float64], rhs: Int)            -> Tensor[Int]: return lt_t(self, to_float64(scalar_int(rhs)))      # a < s(int→f64)

    fn __le__(self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Int]: return le_t(self, rhs)                               # a <= b
    fn __le__(self: Tensor[Float64], rhs: Float64)        -> Tensor[Int]: return le_t(self, scalar_f64(rhs))                    # a <= s(f64)
    fn __le__(self: Tensor[Float64], rhs: Float32)        -> Tensor[Int]: return le_t(self, to_float64(scalar_f32(rhs)))        # a <= s(f32→f64)
    fn __le__(self: Tensor[Float64], rhs: Int)            -> Tensor[Int]: return le_t(self, to_float64(scalar_int(rhs)))      # a <= s(int→f64)

    fn __gt__(self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Int]: return gt_t(self, rhs)                               # a > b
    fn __gt__(self: Tensor[Float64], rhs: Float64)        -> Tensor[Int]: return gt_t(self, scalar_f64(rhs))                    # a > s(f64)
    fn __gt__(self: Tensor[Float64], rhs: Float32)        -> Tensor[Int]: return gt_t(self, to_float64(scalar_f32(rhs)))        # a > s(f32→f64)
    fn __gt__(self: Tensor[Float64], rhs: Int)            -> Tensor[Int]: return gt_t(self, to_float64(scalar_int(rhs)))      # a > s(int→f64)

    fn __ge__(self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Int]: return ge_t(self, rhs)                               # a >= b
    fn __ge__(self: Tensor[Float64], rhs: Float64)        -> Tensor[Int]: return ge_t(self, scalar_f64(rhs))                    # a >= s(f64)
    fn __ge__(self: Tensor[Float64], rhs: Float32)        -> Tensor[Int]: return ge_t(self, to_float64(scalar_f32(rhs)))        # a >= s(f32→f64)
    fn __ge__(self: Tensor[Float64], rhs: Int)            -> Tensor[Int]: return ge_t(self, to_float64(scalar_int(rhs)))      # a >= s(int→f64)

    fn __eq__(self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Int]: return eq_t(self, rhs)                               # a == b
    fn __eq__(self: Tensor[Float64], rhs: Float64)        -> Tensor[Int]: return eq_t(self, scalar_f64(rhs))                    # a == s(f64)
    fn __eq__(self: Tensor[Float64], rhs: Float32)        -> Tensor[Int]: return eq_t(self, to_float64(scalar_f32(rhs)))        # a == s(f32→f64)
    fn __eq__(self: Tensor[Float64], rhs: Int)            -> Tensor[Int]: return eq_t(self, to_float64(scalar_int(rhs)))      # a == s(int→f64)

    fn __ne__(self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Int]: return ne_t(self, rhs)                               # a != b
    fn __ne__(self: Tensor[Float64], rhs: Float64)        -> Tensor[Int]: return ne_t(self, scalar_f64(rhs))                    # a != s(f64)
    fn __ne__(self: Tensor[Float64], rhs: Float32)        -> Tensor[Int]: return ne_t(self, to_float64(scalar_f32(rhs)))        # a != s(f32→f64)
    fn __ne__(self: Tensor[Float64], rhs: Int)            -> Tensor[Int]: return ne_t(self, to_float64(scalar_int(rhs)))      # a != s(int→f64)
    # =========================
    # Float32 comparisons → mask(Int)
    # Promotion: with Float64 ⇒ compare in Float64, with Int ⇒ compare in Float32
    # =========================
    fn __lt__(self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Int]: return lt_t(self, rhs)                               # a < b
    fn __lt__(self: Tensor[Float32], rhs: Float32)        -> Tensor[Int]: return lt_t(self, scalar_f32(rhs))                    # a < s(f32)
    fn __lt__(self: Tensor[Float32], rhs: Float64)        -> Tensor[Int]: return lt_t(to_float64(self), scalar_f64(rhs))        # a(f32→f64) < s(f64)
    fn __lt__(self: Tensor[Float32], rhs: Int)            -> Tensor[Int]: return lt_t(self, to_float32(scalar_int(rhs)))      # a < s(int→f32)

    fn __le__(self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Int]: return le_t(self, rhs)                               # a <= b
    fn __le__(self: Tensor[Float32], rhs: Float32)        -> Tensor[Int]: return le_t(self, scalar_f32(rhs))                    # a <= s(f32)
    fn __le__(self: Tensor[Float32], rhs: Float64)        -> Tensor[Int]: return le_t(to_float64(self), scalar_f64(rhs))        # a(f32→f64) <= s(f64)
    fn __le__(self: Tensor[Float32], rhs: Int)            -> Tensor[Int]: return le_t(self, to_float32(scalar_int(rhs)))      # a <= s(int→f32)

    fn __gt__(self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Int]: return gt_t(self, rhs)                               # a > b
    fn __gt__(self: Tensor[Float32], rhs: Float32)        -> Tensor[Int]: return gt_t(self, scalar_f32(rhs))                    # a > s(f32)
    fn __gt__(self: Tensor[Float32], rhs: Float64)        -> Tensor[Int]: return gt_t(to_float64(self), scalar_f64(rhs))        # a(f32→f64) > s(f64)
    fn __gt__(self: Tensor[Float32], rhs: Int)            -> Tensor[Int]: return gt_t(self, to_float32(scalar_int(rhs)))      # a > s(int→f32)

    fn __ge__(self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Int]: return ge_t(self, rhs)                               # a >= b
    fn __ge__(self: Tensor[Float32], rhs: Float32)        -> Tensor[Int]: return ge_t(self, scalar_f32(rhs))                    # a >= s(f32)
    fn __ge__(self: Tensor[Float32], rhs: Float64)        -> Tensor[Int]: return ge_t(to_float64(self), scalar_f64(rhs))        # a(f32→f64) >= s(f64)
    fn __ge__(self: Tensor[Float32], rhs: Int)            -> Tensor[Int]: return ge_t(self, to_float32(scalar_int(rhs)))      # a >= s(int→f32)

    fn __eq__(self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Int]: return eq_t(self, rhs)                               # a == b
    fn __eq__(self: Tensor[Float32], rhs: Float32)        -> Tensor[Int]: return eq_t(self, scalar_f32(rhs))                    # a == s(f32)
    fn __eq__(self: Tensor[Float32], rhs: Float64)        -> Tensor[Int]: return eq_t(to_float64(self), scalar_f64(rhs))        # a(f32→f64) == s(f64)
    fn __eq__(self: Tensor[Float32], rhs: Int)            -> Tensor[Int]: return eq_t(self, to_float32(scalar_int(rhs)))      # a == s(int→f32)

    fn __ne__(self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Int]: return ne_t(self, rhs)                               # a != b
    fn __ne__(self: Tensor[Float32], rhs: Float32)        -> Tensor[Int]: return ne_t(self, scalar_f32(rhs))                    # a != s(f32)
    fn __ne__(self: Tensor[Float32], rhs: Float64)        -> Tensor[Int]: return ne_t(to_float64(self), scalar_f64(rhs))        # a(f32→f64) != s(f64)
    fn __ne__(self: Tensor[Float32], rhs: Int)            -> Tensor[Int]: return ne_t(self, to_float32(scalar_int(rhs)))      # a != s(int→f32)
    # =========================
    # Int comparisons → mask(Int)
    # Promotion: with Float64 ⇒ compare in Float64; with Float32 ⇒ compare in Float32; else Int
    # =========================
    fn __lt__(self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]: return lt_t(self, rhs)                                        # a < b
    fn __lt__(self: Tensor[Int], rhs: Int)        -> Tensor[Int]: return lt_t(self, scalar_int(rhs))                            # a < s(int)
    fn __lt__(self: Tensor[Int], rhs: Float32)    -> Tensor[Int]: return lt_t(to_float32(self), scalar_f32(rhs))                  # a(int→f32) < s(f32)
    fn __lt__(self: Tensor[Int], rhs: Float64)    -> Tensor[Int]: return lt_t(to_float64(self), scalar_f64(rhs))                  # a(int→f64) < s(f64)

    fn __le__(self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]: return le_t(self, rhs)                                        # a <= b
    fn __le__(self: Tensor[Int], rhs: Int)        -> Tensor[Int]: return le_t(self, scalar_int(rhs))                            # a <= s(int)
    fn __le__(self: Tensor[Int], rhs: Float32)    -> Tensor[Int]: return le_t(to_float32(self), scalar_f32(rhs))                  # a(int→f32) <= s(f32)
    fn __le__(self: Tensor[Int], rhs: Float64)    -> Tensor[Int]: return le_t(to_float64(self), scalar_f64(rhs))                  # a(int→f64) <= s(f64)

    fn __gt__(self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]: return gt_t(self, rhs)                                        # a > b
    fn __gt__(self: Tensor[Int], rhs: Int)        -> Tensor[Int]: return gt_t(self, scalar_int(rhs))                            # a > s(int)
    fn __gt__(self: Tensor[Int], rhs: Float32)    -> Tensor[Int]: return gt_t(to_float32(self), scalar_f32(rhs))                  # a(int→f32) > s(f32)
    fn __gt__(self: Tensor[Int], rhs: Float64)    -> Tensor[Int]: return gt_t(to_float64(self), scalar_f64(rhs))                  # a(int→f64) > s(f64)

    fn __ge__(self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]: return ge_t(self, rhs)                                        # a >= b
    fn __ge__(self: Tensor[Int], rhs: Int)        -> Tensor[Int]: return ge_t(self, scalar_int(rhs))                            # a >= s(int)
    fn __ge__(self: Tensor[Int], rhs: Float32)    -> Tensor[Int]: return ge_t(to_float32(self), scalar_f32(rhs))                  # a(int→f32) >= s(f32)
    fn __ge__(self: Tensor[Int], rhs: Float64)    -> Tensor[Int]: return ge_t(to_float64(self), scalar_f64(rhs))                  # a(int→f64) >= s(f64)

    fn __eq__(self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]: return eq_t(self, rhs)                                        # a == b
    fn __eq__(self: Tensor[Int], rhs: Int)        -> Tensor[Int]: return eq_t(self, scalar_int(rhs))                            # a == s(int)
    fn __eq__(self: Tensor[Int], rhs: Float32)    -> Tensor[Int]: return eq_t(to_float32(self), scalar_f32(rhs))                  # a(int→f32) == s(f32)
    fn __eq__(self: Tensor[Int], rhs: Float64)    -> Tensor[Int]: return eq_t(to_float64(self), scalar_f64(rhs))                  # a(int→f64) == s(f64)

    fn __ne__(self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]: return ne_t(self, rhs)                                        # a != b
    fn __ne__(self: Tensor[Int], rhs: Int)        -> Tensor[Int]: return ne_t(self, scalar_int(rhs))                            # a != s(int)
    fn __ne__(self: Tensor[Int], rhs: Float32)    -> Tensor[Int]: return ne_t(to_float32(self), scalar_f32(rhs))                  # a(int→f32) != s(f32)
    fn __ne__(self: Tensor[Int], rhs: Float64)    -> Tensor[Int]: return ne_t(to_float64(self), scalar_f64(rhs))                  # a(int→f64) != s(f64)

    # =========================
    # Scalar arithmetic with full promotion (Int < Float32 < Float64)
    # Uses only: scalar_int/scalar_f32/scalar_f64, to_int/to_float32/to_float64, add_t/sub_t/mul_t/div_t/mod_t/pow_t
    # =========================

    # -------------------------
    # Tensor[Float64] overloads
    # -------------------------
    fn add_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Float64]: return add_t(self, scalar_f64(s))                    # a + s(f64)
    fn add_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Float64]: return add_t(self, to_float64(scalar_f32(s)))       # a + s(f32→f64)
    fn add_scalar(self: Tensor[Float64], s: Int)     -> Tensor[Float64]: return add_t(self, to_float64(scalar_int(s)))     # a + s(int→f64)

    fn sub_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Float64]: return sub_t(self, scalar_f64(s))                    # a - s(f64)
    fn sub_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Float64]: return sub_t(self, to_float64(scalar_f32(s)))       # a - s(f32→f64)
    fn sub_scalar(self: Tensor[Float64], s: Int)     -> Tensor[Float64]: return sub_t(self, to_float64(scalar_int(s)))     # a - s(int→f64)

    fn mul_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Float64]: return mul_t(self, scalar_f64(s))                    # a * s(f64)
    fn mul_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Float64]: return mul_t(self, to_float64(scalar_f32(s)))       # a * s(f32→f64)
    fn mul_scalar(self: Tensor[Float64], s: Int)     -> Tensor[Float64]: return mul_t(self, to_float64(scalar_int(s)))     # a * s(int→f64)

    fn div_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Float64]: return div_t(self, scalar_f64(s))                    # a / s(f64)
    fn div_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Float64]: return div_t(self, to_float64(scalar_f32(s)))       # a / s(f32→f64)
    fn div_scalar(self: Tensor[Float64], s: Int)     -> Tensor[Float64]: return div_t(self, to_float64(scalar_int(s)))     # a / s(int→f64)

    fn mod_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Float64]: return mod_t(self, scalar_f64(s))                    # a % s(f64)
    fn mod_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Float64]: return mod_t(self, to_float64(scalar_f32(s)))       # a % s(f32→f64)
    fn mod_scalar(self: Tensor[Float64], s: Int)     -> Tensor[Float64]: return mod_t(self, to_float64(scalar_int(s)))     # a % s(int→f64)

    fn pow_scalar(self: Tensor[Float64], p: Float64) -> Tensor[Float64]: return pow_t(self, scalar_f64(p))                    # a ** p(f64)
    fn pow_scalar(self: Tensor[Float64], p: Float32) -> Tensor[Float64]: return pow_t(self, to_float64(scalar_f32(p)))       # a ** p(f32→f64)
    fn pow_scalar(self: Tensor[Float64], p: Int)     -> Tensor[Float64]: return pow_t(self, to_float64(scalar_int(p)))     # a ** p(int→f64)

    # -------------------------
    # Tensor[Float32] overloads
    # -------------------------
    fn add_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Float64]: return add_t(to_float64(self), scalar_f64(s))        # (a→f64) + s(f64)
    fn add_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Float32]: return add_t(self, scalar_f32(s))                     # a + s(f32)
    fn add_scalar(self: Tensor[Float32], s: Int)     -> Tensor[Float32]: return add_t(self, to_float32(scalar_int(s)))      # a + s(int→f32)

    fn sub_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Float64]: return sub_t(to_float64(self), scalar_f64(s))        # (a→f64) - s(f64)
    fn sub_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Float32]: return sub_t(self, scalar_f32(s))                     # a - s(f32)
    fn sub_scalar(self: Tensor[Float32], s: Int)     -> Tensor[Float32]: return sub_t(self, to_float32(scalar_int(s)))      # a - s(int→f32)

    fn mul_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Float64]: return mul_t(to_float64(self), scalar_f64(s))        # (a→f64) * s(f64)
    fn mul_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Float32]: return mul_t(self, scalar_f32(s))                     # a * s(f32)
    fn mul_scalar(self: Tensor[Float32], s: Int)     -> Tensor[Float32]: return mul_t(self, to_float32(scalar_int(s)))      # a * s(int→f32)

    fn div_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Float64]: return div_t(to_float64(self), scalar_f64(s))        # (a→f64) / s(f64)
    fn div_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Float64]: return div_t(self, scalar_f32(s))                     # a / s(f32)
    fn div_scalar(self: Tensor[Float32], s: Int)     -> Tensor[Float64]: return div_t(self, to_float32(scalar_int(s)))      # a / s(int→f32)

    fn mod_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Float64]: return mod_t(to_float64(self), scalar_f64(s))        # (a→f64) % s(f64)
    fn mod_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Float64]: return mod_t(self, scalar_f32(s))                     # a % s(f32)
    fn mod_scalar(self: Tensor[Float32], s: Int)     -> Tensor[Float64]: return mod_t(self, to_float32(scalar_int(s)))      # a % s(int→f32)

    fn pow_scalar(self: Tensor[Float32], p: Float64) -> Tensor[Float64]: return pow_t(to_float64(self), scalar_f64(p))        # (a→f64) ** p(f64)
    fn pow_scalar(self: Tensor[Float32], p: Float32) -> Tensor[Float32]: return pow_t(self, scalar_f32(p))                     # a ** p(f32)
    fn pow_scalar(self: Tensor[Float32], p: Int)     -> Tensor[Float32]: return pow_t(self, to_float32(scalar_int(p)))      # a ** p(int→f32)

    # -------------------------
    # Tensor[Int] overloads
    # -------------------------
    fn add_scalar(self: Tensor[Int], s: Float64) -> Tensor[Float64]: return add_t(to_float64(self), scalar_f64(s))            # (a→f64) + s(f64)
    fn add_scalar(self: Tensor[Int], s: Float32) -> Tensor[Float32]: return add_t(to_float32(self), scalar_f32(s))            # (a→f32) + s(f32)
    fn add_scalar(self: Tensor[Int], s: Int)     -> Tensor[Int]:     return add_t(self, scalar_int(s))                      # a + s(int)

    fn sub_scalar(self: Tensor[Int], s: Float64) -> Tensor[Float64]: return sub_t(to_float64(self), scalar_f64(s))            # (a→f64) - s(f64)
    fn sub_scalar(self: Tensor[Int], s: Float32) -> Tensor[Float32]: return sub_t(to_float32(self), scalar_f32(s))            # (a→f32) - s(f32)
    fn sub_scalar(self: Tensor[Int], s: Int)     -> Tensor[Int]:     return sub_t(self, scalar_int(s))                      # a - s(int)

    fn mul_scalar(self: Tensor[Int], s: Float64) -> Tensor[Float64]: return mul_t(to_float64(self), scalar_f64(s))            # (a→f64) * s(f64)
    fn mul_scalar(self: Tensor[Int], s: Float32) -> Tensor[Float32]: return mul_t(to_float32(self), scalar_f32(s))            # (a→f32) * s(f32)
    fn mul_scalar(self: Tensor[Int], s: Int)     -> Tensor[Int]:     return mul_t(self, scalar_int(s))                      # a * s(int)

    fn div_scalar(self: Tensor[Int], s: Float64) -> Tensor[Float64]: return div_t(to_float64(self), scalar_f64(s))            # (a→f64) / s(f64)
    fn div_scalar(self: Tensor[Int], s: Float32) -> Tensor[Float64]: return div_t(to_float32(self), scalar_f32(s))            # (a→f32) / s(f32)
    fn div_scalar(self: Tensor[Int], s: Int)     -> Tensor[Float64]:     return div_t(self, scalar_int(s))                      # a / s(int)

    fn mod_scalar(self: Tensor[Int], s: Float64) -> Tensor[Float64]: return mod_t(to_float64(self), scalar_f64(s))            # (a→f64) % s(f64)
    fn mod_scalar(self: Tensor[Int], s: Float32) -> Tensor[Float64]: return mod_t(to_float32(self), scalar_f32(s))            # (a→f32) % s(f32)
    fn mod_scalar(self: Tensor[Int], s: Int)     -> Tensor[Float64]:     return mod_t(self, scalar_int(s))                      # a % s(int)

    fn pow_scalar(self: Tensor[Int], p: Float64) -> Tensor[Float64]: return pow_t(to_float64(self), scalar_f64(p))            # (a→f64) ** p(f64)
    fn pow_scalar(self: Tensor[Int], p: Float32) -> Tensor[Float32]: return pow_t(to_float32(self), scalar_f32(p))            # (a→f32) ** p(f32)
    fn pow_scalar(self: Tensor[Int], p: Int)     -> Tensor[Int]:     return pow_t(self, scalar_int(p))                      # a ** p(int)
    # =========================
    # Float64 overloads (tensor-tensor; promotion Int < Float32 < Float64)
    # =========================

    # Arithmetic (tensor)
    fn add   (self: Tensor[Float64], other: Tensor[Float64]) -> Tensor[Float64]: return add_t(self, other)                 # a + b (f64,f64→f64)
    fn add   (self: Tensor[Float64], other: Tensor[Float32]) -> Tensor[Float64]: return add_t(self, to_float64(other))     # a + b (f64,f32→f64)
    fn add   (self: Tensor[Float64], other: Tensor[Int])     -> Tensor[Float64]: return add_t(self, to_float64(other))     # a + b (f64,int→f64)

    fn sub   (self: Tensor[Float64], other: Tensor[Float64]) -> Tensor[Float64]: return sub_t(self, other)                 # a - b (f64,f64→f64)
    fn sub   (self: Tensor[Float64], other: Tensor[Float32]) -> Tensor[Float64]: return sub_t(self, to_float64(other))     # a - b (f64,f32→f64)
    fn sub   (self: Tensor[Float64], other: Tensor[Int])     -> Tensor[Float64]: return sub_t(self, to_float64(other))     # a - b (f64,int→f64)

    fn mul   (self: Tensor[Float64], other: Tensor[Float64]) -> Tensor[Float64]: return mul_t(self, other)                 # a * b (f64,f64→f64)
    fn mul   (self: Tensor[Float64], other: Tensor[Float32]) -> Tensor[Float64]: return mul_t(self, to_float64(other))     # a * b (f64,f32→f64)
    fn mul   (self: Tensor[Float64], other: Tensor[Int])     -> Tensor[Float64]: return mul_t(self, to_float64(other))     # a * b (f64,int→f64)

    fn divide(self: Tensor[Float64], other: Tensor[Float64]) -> Tensor[Float64]: return div_t(self, other)                 # a / b (f64,f64→f64)
    fn divide(self: Tensor[Float64], other: Tensor[Float32]) -> Tensor[Float64]: return div_t(self, to_float64(other))     # a / b (f64,f32→f64)
    fn divide(self: Tensor[Float64], other: Tensor[Int])     -> Tensor[Float64]: return div_t(self, to_float64(other))     # a / b (f64,int→f64)

    fn mod   (self: Tensor[Float64], other: Tensor[Float64]) -> Tensor[Float64]: return mod_t(self, other)                 # a % b (f64,f64→f64)
    fn mod   (self: Tensor[Float64], other: Tensor[Float32]) -> Tensor[Float64]: return mod_t(self, to_float64(other))     # a % b (f64,f32→f64)
    fn mod   (self: Tensor[Float64], other: Tensor[Int])     -> Tensor[Float64]: return mod_t(self, to_float64(other))     # a % b (f64,int→f64)

    fn pow   (self: Tensor[Float64], other: Tensor[Float64]) -> Tensor[Float64]: return pow_t(self, other)                 # a ** b (f64,f64→f64)
    fn pow   (self: Tensor[Float64], other: Tensor[Float32]) -> Tensor[Float64]: return pow_t(self, to_float64(other))     # a ** b (f64,f32→f64)
    fn pow   (self: Tensor[Float64], other: Tensor[Int])     -> Tensor[Float64]: return pow_t(self, to_float64(other))     # a ** b (f64,int→f64)


    # =========================
    # Float32 overloads (tensor-tensor; promotion Int < Float32 < Float64)
    # =========================

    # Arithmetic (tensor)
    fn add   (self: Tensor[Float32], other: Tensor[Float64]) -> Tensor[Float64]: return add_t(to_float64(self), other)     # a + b (f32,f64→f64)
    fn add   (self: Tensor[Float32], other: Tensor[Float32]) -> Tensor[Float32]: return add_t(self, other)                 # a + b (f32,f32→f32)
    fn add   (self: Tensor[Float32], other: Tensor[Int])     -> Tensor[Float32]: return add_t(self, to_float32(other))     # a + b (f32,int→f32)

    fn sub   (self: Tensor[Float32], other: Tensor[Float64]) -> Tensor[Float64]: return sub_t(to_float64(self), other)     # a - b (f32,f64→f64)
    fn sub   (self: Tensor[Float32], other: Tensor[Float32]) -> Tensor[Float32]: return sub_t(self, other)                 # a - b (f32,f32→f32)
    fn sub   (self: Tensor[Float32], other: Tensor[Int])     -> Tensor[Float32]: return sub_t(self, to_float32(other))     # a - b (f32,int→f32)

    fn mul   (self: Tensor[Float32], other: Tensor[Float64]) -> Tensor[Float64]: return mul_t(to_float64(self), other)     # a * b (f32,f64→f64)
    fn mul   (self: Tensor[Float32], other: Tensor[Float32]) -> Tensor[Float32]: return mul_t(self, other)                 # a * b (f32,f32→f32)
    fn mul   (self: Tensor[Float32], other: Tensor[Int])     -> Tensor[Float32]: return mul_t(self, to_float32(other))     # a * b (f32,int→f32)

    fn divide(self: Tensor[Float32], other: Tensor[Float64]) -> Tensor[Float64]: return div_t(to_float64(self), other)     # a / b (f32,f64→f64)
    fn divide(self: Tensor[Float32], other: Tensor[Float32]) -> Tensor[Float32]: return div_t(self, other)                 # a / b (f32,f32→f32)
    fn divide(self: Tensor[Float32], other: Tensor[Int])     -> Tensor[Float32]: return div_t(self, to_float32(other))     # a / b (f32,int→f32)

    fn mod   (self: Tensor[Float32], other: Tensor[Float64]) -> Tensor[Float64]: return mod_t(to_float64(self), other)     # a % b (f32,f64→f64)
    fn mod   (self: Tensor[Float32], other: Tensor[Float32]) -> Tensor[Float64]: return mod_t(self, other)                 # a % b (f32,f32→f32)
    fn mod   (self: Tensor[Float32], other: Tensor[Int])     -> Tensor[Float64]: return mod_t(self, to_float32(other))     # a % b (f32,int→f32)

    fn pow   (self: Tensor[Float32], other: Tensor[Float64]) -> Tensor[Float64]: return pow_t(to_float64(self), other)     # a ** b (f32,f64→f64)
    fn pow   (self: Tensor[Float32], other: Tensor[Float32]) -> Tensor[Float32]: return pow_t(self, other)                 # a ** b (f32,f32→f32)
    fn pow   (self: Tensor[Float32], other: Tensor[Int])     -> Tensor[Float32]: return pow_t(self, to_float32(other))     # a ** b (f32,int→f32)


    # =========================
    # Int overloads (tensor-tensor; promotion Int < Float32 < Float64)
    # =========================

    # Arithmetic (tensor)
    fn add   (self: Tensor[Int], other: Tensor[Float64]) -> Tensor[Float64]: return add_t(to_float64(self), other)         # a + b (int,f64→f64)
    fn add   (self: Tensor[Int], other: Tensor[Float32]) -> Tensor[Float32]: return add_t(to_float32(self), other)         # a + b (int,f32→f32)
    fn add   (self: Tensor[Int], other: Tensor[Int])     -> Tensor[Int]:     return add_t(self, other)                     # a + b (int,int→int)

    fn sub   (self: Tensor[Int], other: Tensor[Float64]) -> Tensor[Float64]: return sub_t(to_float64(self), other)         # a - b (int,f64→f64)
    fn sub   (self: Tensor[Int], other: Tensor[Float32]) -> Tensor[Float32]: return sub_t(to_float32(self), other)         # a - b (int,f32→f32)
    fn sub   (self: Tensor[Int], other: Tensor[Int])     -> Tensor[Int]:     return sub_t(self, other)                     # a - b (int,int→int)

    fn mul   (self: Tensor[Int], other: Tensor[Float64]) -> Tensor[Float64]: return mul_t(to_float64(self), other)         # a * b (int,f64→f64)
    fn mul   (self: Tensor[Int], other: Tensor[Float32]) -> Tensor[Float32]: return mul_t(to_float32(self), other)         # a * b (int,f32→f32)
    fn mul   (self: Tensor[Int], other: Tensor[Int])     -> Tensor[Int]:     return mul_t(self, other)                     # a * b (int,int→int)

    fn divide(self: Tensor[Int], other: Tensor[Float64]) -> Tensor[Float64]: return div_t(to_float64(self), other)         # a / b (int,f64→f64)
    fn divide(self: Tensor[Int], other: Tensor[Float32]) -> Tensor[Float64]: return div_t(to_float32(self), other)         # a / b (int,f32→f32)
    fn divide(self: Tensor[Int], other: Tensor[Int])     -> Tensor[Float64]:     return div_t(self, other)                     # a / b (int,int→int)

    fn mod   (self: Tensor[Int], other: Tensor[Float64]) -> Tensor[Float64]: return mod_t(to_float64(self), other)         # a % b (int,f64→f64)
    fn mod   (self: Tensor[Int], other: Tensor[Float32]) -> Tensor[Float64]: return mod_t(to_float32(self), other)         # a % b (int,f32→f32)
    fn mod   (self: Tensor[Int], other: Tensor[Int])     -> Tensor[Float64]:     return mod_t(self, other)                     # a % b (int,int→int)

    fn pow   (self: Tensor[Int], other: Tensor[Float64]) -> Tensor[Float64]: return pow_t(to_float64(self), other)         # a ** b (int,f64→f64)
    fn pow   (self: Tensor[Int], other: Tensor[Float32]) -> Tensor[Float32]: return pow_t(to_float32(self), other)         # a ** b (int,f32→f32)
    fn pow   (self: Tensor[Int], other: Tensor[Int])     -> Tensor[Float64]:     return pow_t(self, other)                     # a ** b (int,int→int)

    # =========================
    # Float64 overloads (full combos)
    # =========================

    # Logical (bitwise-style over numeric domain; result promoted by rule)
    fn and_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Int]: return to_int(and_t(self, scalar_f64(s)))                                         # a(f64) & s(f64) ⇒ f64
    fn and_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Int]: return to_int(and_t(self, to_float64(scalar_f32(s))))                             # a(f64) & s(f32→f64) ⇒ f64
    fn and_scalar(self: Tensor[Float64], s: Int)     -> Tensor[Int]: return to_int(and_t(self, to_float64(scalar_int(s))))                           # a(f64) & s(int→f64) ⇒ f64

    fn or_scalar (self: Tensor[Float64], s: Float64) -> Tensor[Int]: return to_int(or_t (self, scalar_f64(s)))                                         # a(f64) | s(f64) ⇒ f64
    fn or_scalar (self: Tensor[Float64], s: Float32) -> Tensor[Int]: return to_int(or_t (self, to_float64(scalar_f32(s))))                             # a(f64) | s(f32→f64) ⇒ f64
    fn or_scalar (self: Tensor[Float64], s: Int)     -> Tensor[Int]: return to_int(or_t (self, to_float64(scalar_int(s))))                           # a(f64) | s(int→f64) ⇒ f64

    fn xor_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Int]: return to_int(xor_t(self, scalar_f64(s)))                                         # a(f64) ^ s(f64) ⇒ f64
    fn xor_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Int]: return to_int(xor_t(self, to_float64(scalar_f32(s))))                             # a(f64) ^ s(f32→f64) ⇒ f64
    fn xor_scalar(self: Tensor[Float64], s: Int)     -> Tensor[Int]: return to_int(xor_t(self, to_float64(scalar_int(s))))                           # a(f64) ^ s(int→f64) ⇒ f64

    # Logical boolean ops (mask-style; returns Bool)                                                                    # ~a ⇒ bool mask
    fn logical_and(self: Tensor[Float64], other: Tensor[Float64]) -> Tensor[Bool]: return land_t(self, other)                                     # a(f64) & b(f64) ⇒ bool
    fn logical_and(self: Tensor[Float64], other: Tensor[Float32]) -> Tensor[Bool]: return land_t(self, to_float64(other))                         # a(f64) & b(f32→f64) ⇒ bool
    fn logical_and(self: Tensor[Float64], other: Tensor[Int])     -> Tensor[Bool]: return land_t(self, to_float64(other))                         # a(f64) & b(int→f64) ⇒ bool

    fn logical_or (self: Tensor[Float64], other: Tensor[Float64]) -> Tensor[Bool]: return lor_t (self, other)                                     # a(f64) | b(f64) ⇒ bool
    fn logical_or (self: Tensor[Float64], other: Tensor[Float32]) -> Tensor[Bool]: return lor_t (self, to_float64(other))                         # a(f64) | b(f32→f64) ⇒ bool
    fn logical_or (self: Tensor[Float64], other: Tensor[Int])     -> Tensor[Bool]: return lor_t (self, to_float64(other))                         # a(f64) | b(int→f64) ⇒ bool

    fn logical_xor(self: Tensor[Float64], other: Tensor[Float64]) -> Tensor[Bool]: return lxor_t(self, other)                                     # a(f64) ^ b(f64) ⇒ bool
    fn logical_xor(self: Tensor[Float64], other: Tensor[Float32]) -> Tensor[Bool]: return lxor_t(self, to_float64(other))                         # a(f64) ^ b(f32→f64) ⇒ bool
    fn logical_xor(self: Tensor[Float64], other: Tensor[Int])     -> Tensor[Bool]: return lxor_t(self, to_float64(other))                         # a(f64) ^ b(int→f64) ⇒ bool


    # Logical (bitwise-style over numeric domain; result promoted by rule)
    fn and_bitwise(self: Tensor[Float64], s:Tensor[Float64]) -> Tensor[Int]: return to_int(and_t(self, (s)))                                         # a(f64) & s(f64) ⇒ f64
    fn and_bitwise(self: Tensor[Float64], s:Tensor[Float32]) -> Tensor[Int]: return to_int(and_t(self, to_float64((s))))                             # a(f64) & s(f32→f64) ⇒ f64
    fn and_bitwise(self: Tensor[Float64], s:Tensor[Int])     -> Tensor[Int]: return to_int(and_t(self, to_float64((s))))                           # a(f64) & s(int→f64) ⇒ f64

    fn or_bitwise(self: Tensor[Float64], s: Tensor[Float64]) -> Tensor[Int]: return to_int(or_t (self, (s)))                                         # a(f64) | s(f64) ⇒ f64
    fn or_bitwise(self: Tensor[Float64], s: Tensor[Float32]) -> Tensor[Int]: return to_int(or_t (self, to_float64((s))))                             # a(f64) | s(f32→f64) ⇒ f64
    fn or_bitwise(self: Tensor[Float64], s: Tensor[Int])     -> Tensor[Int]: return to_int(or_t (self, to_float64((s))))                           # a(f64) | s(int→f64) ⇒ f64

    fn xor_bitwise(self: Tensor[Float64], s: Tensor[Float64]) -> Tensor[Int]: return to_int(xor_t(self, (s)))                                         # a(f64) ^ s(f64) ⇒ f64
    fn xor_bitwise(self: Tensor[Float64], s: Tensor[Float32]) -> Tensor[Int]: return to_int(xor_t(self, to_float64((s))))                             # a(f64) ^ s(f32→f64) ⇒ f64
    fn xor_bitwise(self: Tensor[Float64], s: Tensor[Int])     -> Tensor[Int]: return to_int(xor_t(self, to_float64((s))))




    # =========================
    # Float32 overloads (full combos)
    # =========================

    # Numeric bitwise with scalar (result promoted by rule)
    fn and_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Int]: return to_int(and_t(to_float64(self), scalar_f64(s)))                              # a(f32→f64) & s(f64) ⇒ f64
    fn and_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Int]: return to_int(and_t(self, scalar_f32(s)))                                         # a(f32) & s(f32) ⇒ f32
    fn and_scalar(self: Tensor[Float32], s: Int)     -> Tensor[Int]: return to_int(and_t(self, to_float32(scalar_int(s))))                           # a(f32) & s(int→f32) ⇒ f32

    fn or_scalar (self: Tensor[Float32], s: Float64) -> Tensor[Int]: return to_int(or_t (to_float64(self), scalar_f64(s)))                             # a(f32→f64) | s(f64) ⇒ f64
    fn or_scalar (self: Tensor[Float32], s: Float32) -> Tensor[Int]: return to_int(or_t (self, scalar_f32(s)))                                         # a(f32) | s(f32) ⇒ f32
    fn or_scalar (self: Tensor[Float32], s: Int)     -> Tensor[Int]: return to_int(or_t (self, to_float32(scalar_int(s))))                           # a(f32) | s(int→f32) ⇒ f32

    fn xor_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Int]: return to_int(xor_t(to_float64(self), scalar_f64(s)))                             # a(f32→f64) ^ s(f64) ⇒ f64
    fn xor_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Int]: return to_int(xor_t(self, scalar_f32(s)))                                         # a(f32) ^ s(f32) ⇒ f32
    fn xor_scalar(self: Tensor[Float32], s: Int)     -> Tensor[Int]: return to_int(xor_t(self, to_float32(scalar_int(s))))                           # a(f32) ^ s(int→f32) ⇒ f32

    # Boolean logical with tensor (return Bool; promote both args)                                                                # ~a ⇒ bool mask
    fn logical_and(self: Tensor[Float32], other: Tensor[Float64]) -> Tensor[Bool]: return land_t(to_float64(self), other)                         # a(f32→f64) & b(f64) ⇒ bool
    fn logical_and(self: Tensor[Float32], other: Tensor[Float32]) -> Tensor[Bool]: return land_t(self, other)                                     # a(f32) & b(f32) ⇒ bool
    fn logical_and(self: Tensor[Float32], other: Tensor[Int])     -> Tensor[Bool]: return land_t(self, to_float32(other))                         # a(f32) & b(int→f32) ⇒ bool

    fn logical_or (self: Tensor[Float32], other: Tensor[Float64]) -> Tensor[Bool]: return lor_t (to_float64(self), other)                         # a(f32→f64) | b(f64) ⇒ bool
    fn logical_or (self: Tensor[Float32], other: Tensor[Float32]) -> Tensor[Bool]: return lor_t (self, other)                                     # a(f32) | b(f32) ⇒ bool
    fn logical_or (self: Tensor[Float32], other: Tensor[Int])     -> Tensor[Bool]: return lor_t (self, to_float32(other))                         # a(f32) | b(int→f32) ⇒ bool

    fn logical_xor(self: Tensor[Float32], other: Tensor[Float64]) -> Tensor[Bool]: return lxor_t(to_float64(self), other)                         # a(f32→f64) ^ b(f64) ⇒ bool
    fn logical_xor(self: Tensor[Float32], other: Tensor[Float32]) -> Tensor[Bool]: return lxor_t(self, other)                                     # a(f32) ^ b(f32) ⇒ bool
    fn logical_xor(self: Tensor[Float32], other: Tensor[Int])     -> Tensor[Bool]: return lxor_t(self, to_float32(other))                         # a(f32) ^ b(int→f32) ⇒ bool

    # Numeric bitwise with scalar (result promoted by rule)
    fn and_bitwise(self: Tensor[Float32], s: Tensor[Float64]) -> Tensor[Int]: return to_int(and_t(to_float64(self), (s)))                              # a(f32→f64) & s(f64) ⇒ f64
    fn and_bitwise(self: Tensor[Float32], s: Tensor[Float32]) -> Tensor[Int]: return to_int(and_t(self, (s)))                                         # a(f32) & s(f32) ⇒ f32
    fn and_bitwise(self: Tensor[Float32], s: Tensor[Int])     -> Tensor[Int]: return to_int(and_t(self, to_float32((s))))                           # a(f32) & s(int→f32) ⇒ f32

    fn or_bitwise(self: Tensor[Float32], s: Tensor[Float64]) -> Tensor[Int]: return to_int(or_t (to_float64(self), (s)))                             # a(f32→f64) | s(f64) ⇒ f64
    fn or_bitwise(self: Tensor[Float32], s: Tensor[Float32]) -> Tensor[Int]: return to_int(or_t (self, (s)))                                         # a(f32) | s(f32) ⇒ f32
    fn or_bitwise(self: Tensor[Float32], s: Tensor[Int])     -> Tensor[Int]: return to_int(or_t (self, to_float32((s))))                           # a(f32) | s(int→f32) ⇒ f32

    fn xor_bitwise(self: Tensor[Float32], s: Tensor[Float64]) -> Tensor[Int]: return to_int(xor_t(to_float64(self), (s)))                             # a(f32→f64) ^ s(f64) ⇒ f64
    fn xor_bitwise(self: Tensor[Float32], s: Tensor[Float32]) -> Tensor[Int]: return to_int(xor_t(self, (s)))                                         # a(f32) ^ s(f32) ⇒ f32
    fn xor_bitwise(self: Tensor[Float32], s: Tensor[Int])     -> Tensor[Int]: return to_int(xor_t(self, to_float32((s))))                           # a(f32) ^ s(int→f32) ⇒ f32

    # =========================
    # Int overloads (full combos)
    # =========================

    # Numeric bitwise with scalar (result promoted by rule)
    fn and_scalar(self: Tensor[Int], s: Float64) -> Tensor[Int]: return to_int(and_t(to_float64(self), scalar_f64(s)))                                   # a(int→f64) & s(f64) ⇒ f64
    fn and_scalar(self: Tensor[Int], s: Float32) -> Tensor[Int]: return to_int(and_t(to_float32(self), scalar_f32(s)))                                  # a(int→f32) & s(f32) ⇒ f32
    fn and_scalar(self: Tensor[Int], s: Int)     -> Tensor[Int]:     return and_t(self, scalar_int(s))                                           # a(int) & s(int) ⇒ int

    fn or_scalar (self: Tensor[Int], s: Float64) -> Tensor[Int]: return to_int(or_t (to_float64(self), scalar_f64(s)))                                  # a(int→f64) | s(f64) ⇒ f64
    fn or_scalar (self: Tensor[Int], s: Float32) -> Tensor[Int]: return to_int(or_t (to_float32(self), scalar_f32(s)))                                  # a(int→f32) | s(f32) ⇒ f32
    fn or_scalar (self: Tensor[Int], s: Int)     -> Tensor[Int]:     return or_t (self, scalar_int(s))                                           # a(int) | s(int) ⇒ int

    fn xor_scalar(self: Tensor[Int], s: Float64) -> Tensor[Int]: return to_int(xor_t(to_float64(self), scalar_f64(s)))                                   # a(int→f64) ^ s(f64) ⇒ f64
    fn xor_scalar(self: Tensor[Int], s: Float32) -> Tensor[Int]: return to_int(xor_t(to_float32(self), scalar_f32(s)))                                   # a(int→f32) ^ s(f32) ⇒ f32
    fn xor_scalar(self: Tensor[Int], s: Int)     -> Tensor[Int]:     return xor_t(self, scalar_int(s))                                           # a(int) ^ s(int) ⇒ int

    # Boolean logical with tensor (return Bool; promote both args)                                                                     # ~a ⇒ bool mask
    fn logical_and(self: Tensor[Int], other: Tensor[Float64]) -> Tensor[Bool]: return land_t(to_float64(self), other)                              # a(int→f64) & b(f64) ⇒ bool
    fn logical_and(self: Tensor[Int], other: Tensor[Float32]) -> Tensor[Bool]: return land_t(to_float32(self), other)                              # a(int→f32) & b(f32) ⇒ bool
    fn logical_and(self: Tensor[Int], other: Tensor[Int])     -> Tensor[Bool]: return land_t(self, other)                                          # a(int) & b(int) ⇒ bool

    fn logical_or (self: Tensor[Int], other: Tensor[Float64]) -> Tensor[Bool]: return lor_t (to_float64(self), other)                              # a(int→f64) | b(f64) ⇒ bool
    fn logical_or (self: Tensor[Int], other: Tensor[Float32]) -> Tensor[Bool]: return lor_t (to_float32(self), other)                              # a(int→f32) | b(f32) ⇒ bool
    fn logical_or (self: Tensor[Int], other: Tensor[Int])     -> Tensor[Bool]: return lor_t (self, other)                                          # a(int) | b(int) ⇒ bool

    fn logical_xor(self: Tensor[Int], other: Tensor[Float64]) -> Tensor[Bool]: return lxor_t(to_float64(self), other)                              # a(int→f64) ^ b(f64) ⇒ bool
    fn logical_xor(self: Tensor[Int], other: Tensor[Float32]) -> Tensor[Bool]: return lxor_t(to_float32(self), other)                              # a(int→f32) ^ b(f32) ⇒ bool
    fn logical_xor(self: Tensor[Int], other: Tensor[Int])     -> Tensor[Bool]: return lxor_t(self, other)                                          # a(int) ^ b(int) ⇒ bool


    # Numeric bitwise with scalar (result promoted by rule)
    fn and_bitwise(self: Tensor[Int], s: Tensor[Float64]) -> Tensor[Int]: return to_int(and_t(to_float64(self), (s)))                                   # a(int→f64) & s(f64) ⇒ f64
    fn and_bitwise(self: Tensor[Int], s: Tensor[Float32]) -> Tensor[Int]: return to_int(and_t(to_float32(self), (s)))                                  # a(int→f32) & s(f32) ⇒ f32
    fn and_bitwise(self: Tensor[Int], s: Tensor[Int])     -> Tensor[Int]:     return and_t(self, (s))                                           # a(int) & s(int) ⇒ int

    fn or_bitwise(self: Tensor[Int], s: Tensor[Float64]) -> Tensor[Int]: return to_int(or_t (to_float64(self), (s)))                                  # a(int→f64) | s(f64) ⇒ f64
    fn or_bitwise(self: Tensor[Int], s: Tensor[Float32]) -> Tensor[Int]: return to_int(or_t (to_float32(self), (s)))                                  # a(int→f32) | s(f32) ⇒ f32
    fn or_bitwise(self: Tensor[Int], s: Tensor[Int])     -> Tensor[Int]:     return or_t (self, (s))                                           # a(int) | s(int) ⇒ int

    fn xor_bitwise(self: Tensor[Int], s: Tensor[Float64]) -> Tensor[Int]: return to_int(xor_t(to_float64(self), (s)))                                   # a(int→f64) ^ s(f64) ⇒ f64
    fn xor_bitwise(self: Tensor[Int], s: Tensor[Float32]) -> Tensor[Int]: return to_int(xor_t(to_float32(self), (s)))                                   # a(int→f32) ^ s(f32) ⇒ f32
    fn xor_bitwise(self: Tensor[Int], s: Tensor[Int])     -> Tensor[Int]:     return xor_t(self, (s))                                           # a(int) ^ s(int) ⇒ int

    # =========================
    # Float64 overloads — Shifts (full combos, one-liners)
    # Promotion: Int < Float32 < Float64
    # =========================

    # Scalar
    fn lshift_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Float64]: return shl_t(self, scalar_f64(s))                                # a<<s (f64)
    fn lshift_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Float64]: return shl_t(self, to_float64(scalar_f32(s)))                   # a<<s (f32→f64)
    fn lshift_scalar(self: Tensor[Float64], s: Int)     -> Tensor[Float64]: return shl_t(self, to_float64(scalar_int(s)))                 # a<<s (int→f64)
    fn rshift_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Float64]: return shr_t(self, scalar_f64(s))                                # a>>s (f64)
    fn rshift_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Float64]: return shr_t(self, to_float64(scalar_f32(s)))                   # a>>s (f32→f64)
    fn rshift_scalar(self: Tensor[Float64], s: Int)     -> Tensor[Float64]: return shr_t(self, to_float64(scalar_int(s)))                 # a>>s (int→f64)

    # Tensor
    fn lshift(self: Tensor[Float64], other: Tensor[Float64]) -> Tensor[Float64]: return shl_t(self, other)                                  # a<<b (f64,f64)
    fn lshift(self: Tensor[Float64], other: Tensor[Float32]) -> Tensor[Float64]: return shl_t(self, to_float64(other))                     # a<<b (f64,f32→f64)
    fn lshift(self: Tensor[Float64], other: Tensor[Int])     -> Tensor[Float64]: return shl_t(self, to_float64(other))                     # a<<b (f64,int→f64)
    fn rshift(self: Tensor[Float64], other: Tensor[Float64]) -> Tensor[Float64]: return shr_t(self, other)                                  # a>>b (f64,f64)
    fn rshift(self: Tensor[Float64], other: Tensor[Float32]) -> Tensor[Float64]: return shr_t(self, to_float64(other))                     # a>>b (f64,f32→f64)
    fn rshift(self: Tensor[Float64], other: Tensor[Int])     -> Tensor[Float64]: return shr_t(self, to_float64(other))                     # a>>b (f64,int→f64)



    fn not_bitwise(self: Tensor[Float64]) -> Tensor[Int]: return not_t(self)
    fn not_bitwise(self: Tensor[Float32]) -> Tensor[Int]:return not_t(self)
    fn not_bitwise(self: Tensor[Int])     -> Tensor[Int]:return not_t(self)

    # Logical boolean ops (mask-style; returns Bool)
    fn logical_not(self: Tensor[Float64]) -> Tensor[Bool]: return lnot_t(self)
    fn logical_not(self: Tensor[Float32]) -> Tensor[Bool]: return lnot_t(self)
    fn logical_not(self: Tensor[Int]) -> Tensor[Bool]: return lnot_t(self)
    fn logical_not(self: Tensor[Bool]) -> Tensor[Bool]: return lnot_t(self)

    fn logical_and(self: Tensor[Bool], other: Tensor[Bool]) -> Tensor[Bool]: return land_t(self, other)                                     # a(f64) & b(f64) ⇒ bool

    fn logical_or (self: Tensor[Bool], other: Tensor[Bool]) -> Tensor[Bool]: return lor_t (self, other)                                     # a(f64) | b(f64) ⇒ bool

    fn logical_xor(self: Tensor[Bool], other: Tensor[Bool]) -> Tensor[Bool]: return lxor_t(self, other)                                     # a(f64) ^ b(f64) ⇒ bool

    # =========================
    # Float32 overloads — Shifts (full combos, one-liners)
    # Promotion: Int < Float32 < Float64
    # =========================

    # Scalar (note: any f64 ⇒ promote to f64)
    fn lshift_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Float64]: return shl_t(to_float64(self), scalar_f64(s))                   # a<<s (self→f64, s f64)
    fn lshift_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Float32]: return shl_t(self, scalar_f32(s))                               # a<<s (f32)
    fn lshift_scalar(self: Tensor[Float32], s: Int)     -> Tensor[Float32]: return shl_t(self, to_float32(scalar_int(s)))                # a<<s (int→f32)
    fn rshift_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Float64]: return shr_t(to_float64(self), scalar_f64(s))                   # a>>s (self→f64, s f64)
    fn rshift_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Float32]: return shr_t(self, scalar_f32(s))                               # a>>s (f32)
    fn rshift_scalar(self: Tensor[Float32], s: Int)     -> Tensor[Float32]: return shr_t(self, to_float32(scalar_int(s)))                # a>>s (int→f32)

    # Tensor (any f64 ⇒ result f64; else if any f32 ⇒ f32)
    fn lshift(self: Tensor[Float32], other: Tensor[Float64]) -> Tensor[Float64]: return shl_t(to_float64(self), other)                     # a<<b (→f64)
    fn lshift(self: Tensor[Float32], other: Tensor[Float32]) -> Tensor[Float32]: return shl_t(self, other)                                 # a<<b (f32)
    fn lshift(self: Tensor[Float32], other: Tensor[Int])     -> Tensor[Float32]: return shl_t(self, to_float32(other))                     # a<<b (int→f32)
    fn rshift(self: Tensor[Float32], other: Tensor[Float64]) -> Tensor[Float64]: return shr_t(to_float64(self), other)                     # a>>b (→f64)
    fn rshift(self: Tensor[Float32], other: Tensor[Float32]) -> Tensor[Float32]: return shr_t(self, other)                                 # a>>b (f32)
    fn rshift(self: Tensor[Float32], other: Tensor[Int])     -> Tensor[Float32]: return shr_t(self, to_float32(other))                     # a>>b (int→f32)


    # =========================
    # Int overloads — Shifts (full combos, one-liners)
    # Promotion: Int < Float32 < Float64
    # =========================

    # Scalar (promote to largest of {self,s})
    fn lshift_scalar(self: Tensor[Int], s: Float64) -> Tensor[Float64]: return shl_t(to_float64(self), scalar_f64(s))                       # a<<s (→f64)
    fn lshift_scalar(self: Tensor[Int], s: Float32) -> Tensor[Float32]: return shl_t(to_float32(self), scalar_f32(s))                      # a<<s (→f32)
    fn lshift_scalar(self: Tensor[Int], s: Int)     -> Tensor[Int]:     return shl_t(self, scalar_int(s))                                # a<<s (int)
    fn rshift_scalar(self: Tensor[Int], s: Float64) -> Tensor[Float64]: return shr_t(to_float64(self), scalar_f64(s))                       # a>>s (→f64)
    fn rshift_scalar(self: Tensor[Int], s: Float32) -> Tensor[Float32]: return shr_t(to_float32(self), scalar_f32(s))                      # a>>s (→f32)
    fn rshift_scalar(self: Tensor[Int], s: Int)     -> Tensor[Int]:     return shr_t(self, scalar_int(s))                                # a>>s (int)

    # Tensor (promote to largest of {self,other})
    fn lshift(self: Tensor[Int], other: Tensor[Float64]) -> Tensor[Float64]: return shl_t(to_float64(self), other)                        # a<<b (→f64)
    fn lshift(self: Tensor[Int], other: Tensor[Float32]) -> Tensor[Float32]: return shl_t(to_float32(self), other)                        # a<<b (→f32)
    fn lshift(self: Tensor[Int], other: Tensor[Int])     -> Tensor[Int]:     return shl_t(self, other)                                    # a<<b (int)
    fn rshift(self: Tensor[Int], other: Tensor[Float64]) -> Tensor[Float64]: return shr_t(to_float64(self), other)                        # a>>b (→f64)
    fn rshift(self: Tensor[Int], other: Tensor[Float32]) -> Tensor[Float32]: return shr_t(to_float32(self), other)                        # a>>b (→f32)
    fn rshift(self: Tensor[Int], other: Tensor[Int])     -> Tensor[Int]:     return shr_t(self, other)                                    # a>>b (int)
    # =========================
    # Float64 comparisons (→ mask Int) — full scalar/tensor combos
    # Promotion: Int < Float32 < Float64
    # =========================

    # Scalars
    fn lt_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Int]: return lt_t(self, scalar_f64(s))                                # a < s(f64)
    fn lt_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Int]: return lt_t(self, to_float64(scalar_f32(s)))                   # a < s(f32→f64)
    fn lt_scalar(self: Tensor[Float64], s: Int)     -> Tensor[Int]: return lt_t(self, to_float64(scalar_int(s)))                 # a < s(int→f64)

    fn le_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Int]: return le_t(self, scalar_f64(s))                                # a <= s(f64)
    fn le_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Int]: return le_t(self, to_float64(scalar_f32(s)))                   # a <= s(f32→f64)
    fn le_scalar(self: Tensor[Float64], s: Int)     -> Tensor[Int]: return le_t(self, to_float64(scalar_int(s)))                 # a <= s(int→f64)

    fn gt_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Int]: return gt_t(self, scalar_f64(s))                                # a > s(f64)
    fn gt_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Int]: return gt_t(self, to_float64(scalar_f32(s)))                   # a > s(f32→f64)
    fn gt_scalar(self: Tensor[Float64], s: Int)     -> Tensor[Int]: return gt_t(self, to_float64(scalar_int(s)))                 # a > s(int→f64)

    fn ge_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Int]: return ge_t(self, scalar_f64(s))                                # a >= s(f64)
    fn ge_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Int]: return ge_t(self, to_float64(scalar_f32(s)))                   # a >= s(f32→f64)
    fn ge_scalar(self: Tensor[Float64], s: Int)     -> Tensor[Int]: return ge_t(self, to_float64(scalar_int(s)))                 # a >= s(int→f64)

    fn eq_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Int]: return eq_t(self, scalar_f64(s))                                # a == s(f64)
    fn eq_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Int]: return eq_t(self, to_float64(scalar_f32(s)))                   # a == s(f32→f64)
    fn eq_scalar(self: Tensor[Float64], s: Int)     -> Tensor[Int]: return eq_t(self, to_float64(scalar_int(s)))                 # a == s(int→f64)

    fn ne_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Int]: return ne_t(self, scalar_f64(s))                                # a != s(f64)
    fn ne_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Int]: return ne_t(self, to_float64(scalar_f32(s)))                   # a != s(f32→f64)
    fn ne_scalar(self: Tensor[Float64], s: Int)     -> Tensor[Int]: return ne_t(self, to_float64(scalar_int(s)))                 # a != s(int→f64)

    # Tensors
    fn lt(self: Tensor[Float64], other: Tensor[Float64]) -> Tensor[Int]: return lt_t(self, other)                                  # a < b(f64)
    fn lt(self: Tensor[Float64], other: Tensor[Float32]) -> Tensor[Int]: return lt_t(self, to_float64(other))                     # a < b(f32→f64)
    fn lt(self: Tensor[Float64], other: Tensor[Int])     -> Tensor[Int]: return lt_t(self, to_float64(other))                     # a < b(int→f64)

    fn le(self: Tensor[Float64], other: Tensor[Float64]) -> Tensor[Int]: return le_t(self, other)                                  # a <= b(f64)
    fn le(self: Tensor[Float64], other: Tensor[Float32]) -> Tensor[Int]: return le_t(self, to_float64(other))                     # a <= b(f32→f64)
    fn le(self: Tensor[Float64], other: Tensor[Int])     -> Tensor[Int]: return le_t(self, to_float64(other))                     # a <= b(int→f64)

    fn gt(self: Tensor[Float64], other: Tensor[Float64]) -> Tensor[Int]: return gt_t(self, other)                                  # a > b(f64)
    fn gt(self: Tensor[Float64], other: Tensor[Float32]) -> Tensor[Int]: return gt_t(self, to_float64(other))                     # a > b(f32→f64)
    fn gt(self: Tensor[Float64], other: Tensor[Int])     -> Tensor[Int]: return gt_t(self, to_float64(other))                     # a > b(int→f64)

    fn ge(self: Tensor[Float64], other: Tensor[Float64]) -> Tensor[Int]: return ge_t(self, other)                                  # a >= b(f64)
    fn ge(self: Tensor[Float64], other: Tensor[Float32]) -> Tensor[Int]: return ge_t(self, to_float64(other))                     # a >= b(f32→f64)
    fn ge(self: Tensor[Float64], other: Tensor[Int])     -> Tensor[Int]: return ge_t(self, to_float64(other))                     # a >= b(int→f64)

    fn eq(self: Tensor[Float64], other: Tensor[Float64]) -> Tensor[Int]: return eq_t(self, other)                                  # a == b(f64)
    fn eq(self: Tensor[Float64], other: Tensor[Float32]) -> Tensor[Int]: return eq_t(self, to_float64(other))                     # a == b(f32→f64)
    fn eq(self: Tensor[Float64], other: Tensor[Int])     -> Tensor[Int]: return eq_t(self, to_float64(other))                     # a == b(int→f64)

    fn ne(self: Tensor[Float64], other: Tensor[Float64]) -> Tensor[Int]: return ne_t(self, other)                                  # a != b(f64)
    fn ne(self: Tensor[Float64], other: Tensor[Float32]) -> Tensor[Int]: return ne_t(self, to_float64(other))                     # a != b(f32→f64)
    fn ne(self: Tensor[Float64], other: Tensor[Int])     -> Tensor[Int]: return ne_t(self, to_float64(other))                     # a != b(int→f64)
    # =========================
    # Float32 comparisons (→ mask Int) — full scalar/tensor combos
    # Promotion: Int < Float32 < Float64
    # =========================

    # Scalars
    fn lt_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Int]: return lt_t(to_float64(self), scalar_f64(s))                   # a(f32→f64) < s(f64)
    fn lt_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Int]: return lt_t(self, scalar_f32(s))                               # a(f32) < s(f32)
    fn lt_scalar(self: Tensor[Float32], s: Int)     -> Tensor[Int]: return lt_t(self, to_float32(scalar_int(s)))                 # a(f32) < s(int→f32)

    fn le_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Int]: return le_t(to_float64(self), scalar_f64(s))                   # a(f32→f64) <= s(f64)
    fn le_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Int]: return le_t(self, scalar_f32(s))                               # a(f32) <= s(f32)
    fn le_scalar(self: Tensor[Float32], s: Int)     -> Tensor[Int]: return le_t(self, to_float32(scalar_int(s)))                 # a(f32) <= s(int→f32)

    fn gt_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Int]: return gt_t(to_float64(self), scalar_f64(s))                   # a(f32→f64) > s(f64)
    fn gt_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Int]: return gt_t(self, scalar_f32(s))                               # a(f32) > s(f32)
    fn gt_scalar(self: Tensor[Float32], s: Int)     -> Tensor[Int]: return gt_t(self, to_float32(scalar_int(s)))                 # a(f32) > s(int→f32)

    fn ge_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Int]: return ge_t(to_float64(self), scalar_f64(s))                   # a(f32→f64) >= s(f64)
    fn ge_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Int]: return ge_t(self, scalar_f32(s))                               # a(f32) >= s(f32)
    fn ge_scalar(self: Tensor[Float32], s: Int)     -> Tensor[Int]: return ge_t(self, to_float32(scalar_int(s)))                 # a(f32) >= s(int→f32)

    fn eq_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Int]: return eq_t(to_float64(self), scalar_f64(s))                   # a(f32→f64) == s(f64)
    fn eq_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Int]: return eq_t(self, scalar_f32(s))                               # a(f32) == s(f32)
    fn eq_scalar(self: Tensor[Float32], s: Int)     -> Tensor[Int]: return eq_t(self, to_float32(scalar_int(s)))                 # a(f32) == s(int→f32)

    fn ne_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Int]: return ne_t(to_float64(self), scalar_f64(s))                   # a(f32→f64) != s(f64)
    fn ne_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Int]: return ne_t(self, scalar_f32(s))                               # a(f32) != s(f32)
    fn ne_scalar(self: Tensor[Float32], s: Int)     -> Tensor[Int]: return ne_t(self, to_float32(scalar_int(s)))                 # a(f32) != s(int→f32)

    # Tensors
    fn lt(self: Tensor[Float32], other: Tensor[Float64]) -> Tensor[Int]: return lt_t(to_float64(self), other)                     # a(f32→f64) < b(f64)
    fn lt(self: Tensor[Float32], other: Tensor[Float32]) -> Tensor[Int]: return lt_t(self, other)                                 # a(f32) < b(f32)
    fn lt(self: Tensor[Float32], other: Tensor[Int])     -> Tensor[Int]: return lt_t(self, to_float32(other))                     # a(f32) < b(int→f32)

    fn le(self: Tensor[Float32], other: Tensor[Float64]) -> Tensor[Int]: return le_t(to_float64(self), other)                     # a(f32→f64) <= b(f64)
    fn le(self: Tensor[Float32], other: Tensor[Float32]) -> Tensor[Int]: return le_t(self, other)                                 # a(f32) <= b(f32)
    fn le(self: Tensor[Float32], other: Tensor[Int])     -> Tensor[Int]: return le_t(self, to_float32(other))                     # a(f32) <= b(int→f32)

    fn gt(self: Tensor[Float32], other: Tensor[Float64]) -> Tensor[Int]: return gt_t(to_float64(self), other)                     # a(f32→f64) > b(f64)
    fn gt(self: Tensor[Float32], other: Tensor[Float32]) -> Tensor[Int]: return gt_t(self, other)                                 # a(f32) > b(f32)
    fn gt(self: Tensor[Float32], other: Tensor[Int])     -> Tensor[Int]: return gt_t(self, to_float32(other))                     # a(f32) > b(int→f32)

    fn ge(self: Tensor[Float32], other: Tensor[Float64]) -> Tensor[Int]: return ge_t(to_float64(self), other)                     # a(f32→f64) >= b(f64)
    fn ge(self: Tensor[Float32], other: Tensor[Float32]) -> Tensor[Int]: return ge_t(self, other)                                 # a(f32) >= b(f32)
    fn ge(self: Tensor[Float32], other: Tensor[Int])     -> Tensor[Int]: return ge_t(self, to_float32(other))                     # a(f32) >= b(int→f32)

    fn eq(self: Tensor[Float32], other: Tensor[Float64]) -> Tensor[Int]: return eq_t(to_float64(self), other)                     # a(f32→f64) == b(f64)
    fn eq(self: Tensor[Float32], other: Tensor[Float32]) -> Tensor[Int]: return eq_t(self, other)                                 # a(f32) == b(f32)
    fn eq(self: Tensor[Float32], other: Tensor[Int])     -> Tensor[Int]: return eq_t(self, to_float32(other))                     # a(f32) == b(int→f32)

    fn ne(self: Tensor[Float32], other: Tensor[Float64]) -> Tensor[Int]: return ne_t(to_float64(self), other)                     # a(f32→f64) != b(f64)
    fn ne(self: Tensor[Float32], other: Tensor[Float32]) -> Tensor[Int]: return ne_t(self, other)                                 # a(f32) != b(f32)
    fn ne(self: Tensor[Float32], other: Tensor[Int])     -> Tensor[Int]: return ne_t(self, to_float32(other))                     # a(f32) != b(int→f32)
    # =========================
    # Int comparisons (→ mask Int) — full scalar/tensor combos
    # Promotion: Int < Float32 < Float64
    # =========================

    # Scalars
    fn lt_scalar(self: Tensor[Int], s: Float64) -> Tensor[Int]: return lt_t(to_float64(self), scalar_f64(s))                        # a(int→f64) < s(f64)
    fn lt_scalar(self: Tensor[Int], s: Float32) -> Tensor[Int]: return lt_t(to_float32(self), scalar_f32(s))                        # a(int→f32) < s(f32)
    fn lt_scalar(self: Tensor[Int], s: Int)     -> Tensor[Int]: return lt_t(self, scalar_int(s))                                  # a(int) < s(int)

    fn le_scalar(self: Tensor[Int], s: Float64) -> Tensor[Int]: return le_t(to_float64(self), scalar_f64(s))                        # a(int→f64) <= s(f64)
    fn le_scalar(self: Tensor[Int], s: Float32) -> Tensor[Int]: return le_t(to_float32(self), scalar_f32(s))                        # a(int→f32) <= s(f32)
    fn le_scalar(self: Tensor[Int], s: Int)     -> Tensor[Int]: return le_t(self, scalar_int(s))                                  # a(int) <= s(int)

    fn gt_scalar(self: Tensor[Int], s: Float64) -> Tensor[Int]: return gt_t(to_float64(self), scalar_f64(s))                        # a(int→f64) > s(f64)
    fn gt_scalar(self: Tensor[Int], s: Float32) -> Tensor[Int]: return gt_t(to_float32(self), scalar_f32(s))                        # a(int→f32) > s(f32)
    fn gt_scalar(self: Tensor[Int], s: Int)     -> Tensor[Int]: return gt_t(self, scalar_int(s))                                  # a(int) > s(int)

    fn ge_scalar(self: Tensor[Int], s: Float64) -> Tensor[Int]: return ge_t(to_float64(self), scalar_f64(s))                        # a(int→f64) >= s(f64)
    fn ge_scalar(self: Tensor[Int], s: Float32) -> Tensor[Int]: return ge_t(to_float32(self), scalar_f32(s))                        # a(int→f32) >= s(f32)
    fn ge_scalar(self: Tensor[Int], s: Int)     -> Tensor[Int]: return ge_t(self, scalar_int(s))                                  # a(int) >= s(int)

    fn eq_scalar(self: Tensor[Int], s: Float64) -> Tensor[Int]: return eq_t(to_float64(self), scalar_f64(s))                        # a(int→f64) == s(f64)
    fn eq_scalar(self: Tensor[Int], s: Float32) -> Tensor[Int]: return eq_t(to_float32(self), scalar_f32(s))                        # a(int→f32) == s(f32)
    fn eq_scalar(self: Tensor[Int], s: Int)     -> Tensor[Int]: return eq_t(self, scalar_int(s))                                  # a(int) == s(int)

    fn ne_scalar(self: Tensor[Int], s: Float64) -> Tensor[Int]: return ne_t(to_float64(self), scalar_f64(s))                        # a(int→f64) != s(f64)
    fn ne_scalar(self: Tensor[Int], s: Float32) -> Tensor[Int]: return ne_t(to_float32(self), scalar_f32(s))                        # a(int→f32) != s(f32)
    fn ne_scalar(self: Tensor[Int], s: Int)     -> Tensor[Int]: return ne_t(self, scalar_int(s))                                  # a(int) != s(int)

    # Tensors
    fn lt(self: Tensor[Int], other: Tensor[Float64]) -> Tensor[Int]: return lt_t(to_float64(self), other)                          # a(int→f64) < b(f64)
    fn lt(self: Tensor[Int], other: Tensor[Float32]) -> Tensor[Int]: return lt_t(to_float32(self), other)                          # a(int→f32) < b(f32)
    fn lt(self: Tensor[Int], other: Tensor[Int])     -> Tensor[Int]: return lt_t(self, other)                                      # a(int) < b(int)

    fn le(self: Tensor[Int], other: Tensor[Float64]) -> Tensor[Int]: return le_t(to_float64(self), other)                          # a(int→f64) <= b(f64)
    fn le(self: Tensor[Int], other: Tensor[Float32]) -> Tensor[Int]: return le_t(to_float32(self), other)                          # a(int→f32) <= b(f32)
    fn le(self: Tensor[Int], other: Tensor[Int])     -> Tensor[Int]: return le_t(self, other)                                      # a(int) <= b(int)

    fn gt(self: Tensor[Int], other: Tensor[Float64]) -> Tensor[Int]: return gt_t(to_float64(self), other)                          # a(int→f64) > b(f64)
    fn gt(self: Tensor[Int], other: Tensor[Float32]) -> Tensor[Int]: return gt_t(to_float32(self), other)                          # a(int→f32) > b(f32)
    fn gt(self: Tensor[Int], other: Tensor[Int])     -> Tensor[Int]: return gt_t(self, other)                                      # a(int) > b(int)

    fn ge(self: Tensor[Int], other: Tensor[Float64]) -> Tensor[Int]: return ge_t(to_float64(self), other)                          # a(int→f64) >= b(f64)
    fn ge(self: Tensor[Int], other: Tensor[Float32]) -> Tensor[Int]: return ge_t(to_float32(self), other)                          # a(int→f32) >= b(f32)
    fn ge(self: Tensor[Int], other: Tensor[Int])     -> Tensor[Int]: return ge_t(self, other)                                      # a(int) >= b(int)

    fn eq(self: Tensor[Int], other: Tensor[Float64]) -> Tensor[Int]: return eq_t(to_float64(self), other)                          # a(int→f64) == b(f64)
    fn eq(self: Tensor[Int], other: Tensor[Float32]) -> Tensor[Int]: return eq_t(to_float32(self), other)                          # a(int→f32) == b(f32)
    fn eq(self: Tensor[Int], other: Tensor[Int])     -> Tensor[Int]: return eq_t(self, other)                                      # a(int) == b(int)

    fn ne(self: Tensor[Int], other: Tensor[Float64]) -> Tensor[Int]: return ne_t(to_float64(self), other)                          # a(int→f64) != b(f64)
    fn ne(self: Tensor[Int], other: Tensor[Float32]) -> Tensor[Int]: return ne_t(to_float32(self), other)                          # a(int→f32) != b(f32)
    fn ne(self: Tensor[Int], other: Tensor[Int])     -> Tensor[Int]: return ne_t(self, other)                                      # a(int) != b(int)


    fn to_float64(self: Tensor[Float64]) -> Tensor[Float64]: return to_float64(self)
    fn to_float64(self: Tensor[Float32]) -> Tensor[Float64]: return to_float64(self)
    fn to_float64(self: Tensor[Int])     -> Tensor[Float64]: return to_float64(self)

    fn to_float32(self: Tensor[Float64]) -> Tensor[Float32]: return to_float32(self)
    fn to_float32(self: Tensor[Float32]) -> Tensor[Float32]: return to_float32(self)
    fn to_float32(self: Tensor[Int])     -> Tensor[Float32]: return to_float32(self)

    fn to_int(self: Tensor[Float64])     -> Tensor[Int]:     return to_int(self)
    fn to_int(self: Tensor[Float32])     -> Tensor[Int]:     return to_int(self)
    fn to_int(self: Tensor[Int])         -> Tensor[Int]:     return to_int(self)




    fn view(self ,shape: List[Int]) -> Tensor[T]:        return view(self,shape)

    fn dtype_name(self: Tensor[Bool])    -> String: return "Bool"
    fn dtype_name(self: Tensor[Int])     -> String: return "Int"
    fn dtype_name(self: Tensor[UInt])    -> String: return "UInt"
    fn dtype_name(self: Tensor[Int8])    -> String: return "Int8"
    fn dtype_name(self: Tensor[Int16])   -> String: return "Int16"
    fn dtype_name(self: Tensor[Int32])   -> String: return "Int32"
    fn dtype_name(self: Tensor[Int64])   -> String: return "Int64"
    fn dtype_name(self: Tensor[UInt8])   -> String: return "UInt8"
    fn dtype_name(self: Tensor[UInt16])  -> String: return "UInt16"
    fn dtype_name(self: Tensor[UInt32])  -> String: return "UInt32"
    fn dtype_name(self: Tensor[UInt64])  -> String: return "UInt64"
    fn dtype_name(self: Tensor[Float32]) -> String: return "Float32"
    fn dtype_name(self: Tensor[Float64]) -> String: return "Float64"

    fn is_contiguous(self: Tensor[Float64]) -> Bool:    return is_row_major_contiguous(self._shape, self._strides)
    fn is_contiguous(self: Tensor[Float32]) -> Bool:    return is_row_major_contiguous(self._shape, self._strides)
    fn is_contiguous(self: Tensor[Int]) -> Bool:        return is_row_major_contiguous(self._shape, self._strides)


    fn sigmoid(self: Tensor[Float64]) -> Tensor[Float64]: return sigmoid_t(self)
    fn sigmoid(self: Tensor[Float32]) -> Tensor[Float64]: return sigmoid_t(self)
    fn sigmoid(self: Tensor[Int])     -> Tensor[Float64]: return sigmoid_t(self)

    @always_inline
    fn scatter_add(self: Tensor[Int], dim: Int, index: Tensor[Int], src: Tensor[Int]) -> Tensor[Int]:
        if dim == 0: return scatter_add_dim0_int(self, index, src)
        if dim == 1: return scatter_add_dim1_int(self, index, src)
        return self.copy()

    @always_inline
    fn scatter_add(self: Tensor[Float32], dim: Int, index: Tensor[Int], src: Tensor[Float32]) -> Tensor[Float32]:
        if dim == 0: return scatter_add_dim0_f32(self, index, src)
        if dim == 1: return scatter_add_dim1_f32(self, index, src)
        return self.copy()

    @always_inline
    fn scatter_add(self: Tensor[Float64], dim: Int, index: Tensor[Int], src: Tensor[Float64]) -> Tensor[Float64]:
        if dim == 0: return scatter_add_dim0_f64(self, index, src)
        if dim == 1: return scatter_add_dim1_f64(self, index, src)
        return self.copy()

    @always_inline
    fn scatter(self,dim_in: Int,index: Tensor[Int],src: Tensor[T]) -> Tensor[T]:
        return scatter(self,dim_in,index,src)

    fn contiguous(self: Tensor[Float64]) -> Tensor[Float64]:
        if is_row_major_contiguous(self._shape, self._strides):
            return self.copy()
        var flat = self.flatten()
        var shp = copy_list_int(self._shape)
        var strides = compute_row_major_strides(shp)
        return Tensor[Float64](flat._data, shp, strides, 0)

    fn contiguous(self: Tensor[Float32]) -> Tensor[Float32]:
        if is_row_major_contiguous(self._shape, self._strides):
            return self.copy()
        var flat = self.flatten()
        var shp = copy_list_int(self._shape)
        var strides = compute_row_major_strides(shp)
        return Tensor[Float32](flat._data, shp, strides, 0)

    fn contiguous(self: Tensor[Int]) -> Tensor[Int]:
        if is_row_major_contiguous(self._shape, self._strides):
            return self.copy()
        var flat = self.flatten()
        var shp = copy_list_int(self._shape)
        var strides = compute_row_major_strides(shp)
        return Tensor[Int](flat._data, shp, strides, 0)



        # ---------------- High-performance appends on Tensor._data ---------------- #
    @always_inline
    fn _reserve_extra(mut self, extra: Int) -> None:
        # if capacity for 'extra' more elements.
        if extra <= 0: return
        self._data.reserve(len(self._data) + extra)

    @always_inline
    fn append(mut self, x: T) -> None:
        # Append a single element to the underlying storage.
        self._data.append(x)

    @always_inline
    fn append_list(mut self, xs: List[T]) -> None:
        # Append entire list 'xs' into underlying storage with unroll-8.
        var n = len(xs)
        if n == 0: return
        self._reserve_extra(n)
        var i = 0
        var lim = (n // 8) * 8
        while i < lim:
            self._data.append(xs[i + 0]); self._data.append(xs[i + 1])
            self._data.append(xs[i + 2]); self._data.append(xs[i + 3])
            self._data.append(xs[i + 4]); self._data.append(xs[i + 5])
            self._data.append(xs[i + 6]); self._data.append(xs[i + 7])
            i += 8
        while i < n:
            self._data.append(xs[i])
            i += 1

    @always_inline
    fn append_repeat(mut self, val: T, count: Int) -> None:
        # Append 'count' copies of 'val' with unroll-8.
        if count <= 0: return
        self._reserve_extra(count)
        var k = 0
        var lim = (count // 8) * 8
        while k < lim:
            self._data.append(val); self._data.append(val)
            self._data.append(val); self._data.append(val)
            self._data.append(val); self._data.append(val)
            self._data.append(val); self._data.append(val)
            k += 8
        while k < count:
            self._data.append(val)
            k += 1

    @always_inline
    fn append_strided(mut self, src: List[T], base: Int, n: Int, step: Int = 1) -> None:
        # Append 'n' items from 'src', starting at 'base' with stride 'step'.
        # Useful for building from non-contiguous layouts (printing/slicing).
        if n <= 0: return
        self._reserve_extra(n)
        var i = 0
        var idx = base
        var lim = (n // 8) * 8
        while i < lim:
            self._data.append(src[idx + 0 * step]); self._data.append(src[idx + 1 * step])
            self._data.append(src[idx + 2 * step]); self._data.append(src[idx + 3 * step])
            self._data.append(src[idx + 4 * step]); self._data.append(src[idx + 5 * step])
            self._data.append(src[idx + 6 * step]); self._data.append(src[idx + 7 * step])
            idx += 8 * step
            i += 8
        while i < n:
            self._data.append(src[idx])
            idx += step
            i += 1

    # (Optional) finalize helpers for builders
    @always_inline
    fn set_shape_row_major(mut self, shape: List[Int]) -> None:
        # Set shape and recompute row-major strides; keeps current _data as-is.
        var shp = shape.copy()
        self._shape = shp
        # compute_row_major_strides(..) must exist in your module:
        self._strides = compute_row_major_strides(self._shape)
        self._offset = 0

    fn gather(self, axis: Int, index: Tensor[Int]) -> Tensor[T]:
        return gather(self, axis, index)

    @always_inline
    fn uniform_(mut self: Tensor[Float64], low: Float64, high: Float64, seed: Optional[Int] = None) -> Tensor[Float64]:
        var s: UInt64
        if seed is None:
            s = UInt64(0xC6BC279692B5C323)
        else:
            s = UInt64(seed.value())
        var rng = XorShift64(s)

        var n = len(self._data)
        var i = 0
        var span = high - low
        while i < n:
            var u = rng.next_unit_f64()
            self._data[i] = low + span * u
            i += 1
        return self.copy()


    @always_inline
    fn uniform_(mut self: Tensor[Float32], low: Float32, high: Float32, seed: Optional[Int] = None) -> Tensor[Float32]:
        var s: UInt64
        if seed is None:
            s = UInt64(0x94D049BB133111EB)
        else:
            s = UInt64(seed.value())
        var rng = XorShift64(s)

        var n = len(self._data)
        var i = 0
        var span = low
        span = high - low
        while i < n:
            var u = rng.next_unit_f64()
            self._data[i] = low + Float32(Float64(span) * u)
            i += 1
        return self.copy()



    # =========================
    # Bernoulli — Float32
    # =========================
    @always_inline
    fn bernoulli(self: Tensor[Float32], seed: Optional[Int] = None) -> Tensor[Float32]:
        var out = self.copy()
        var s: UInt64
        if seed is None:
            s = UInt64(0x94D049BB133111EB)
        else:
            s = UInt64(seed.value())
        var rng = XorShift64(s)

        var n = len(out._data)
        var i = 0
        while i < n:
            var p = Float64(self._data[i])
            if p < 0.0: p = 0.0
            if p > 1.0: p = 1.0
            var u = rng.next_unit_f64()
            if u < p:
                out._data[i] = Float32(1.0)
            else:
                out._data[i] = Float32(0.0)
            i += 1
        return out.copy()



    # =========================
    # Bernoulli — Float64
    # =========================
    @always_inline
    fn bernoulli(self: Tensor[Float64], seed: Optional[Int] = None) -> Tensor[Float64]:
        var out = self.copy()  # same dtype/shape
        var s: UInt64
        if seed is None:
            s = UInt64(0xC0FFEE0012345678)
        else:
            s = UInt64(seed.value())
        var rng = XorShift64(s)

        var n = len(out._data)
        var i = 0
        while i < n:
            var p = self._data[i]
            if p < 0.0: p = 0.0
            if p > 1.0: p = 1.0
            var u = rng.next_unit_f64()
            if u < p:
                out._data[i] = 1.0
            else:
                out._data[i] = 0.0
            i += 1
        return out.copy()


    @always_inline
    fn multinomial(
        self: Tensor[Float64],
        num_samples: Int,
        replacement: Bool = True,
        seed: Optional[Int] = None
    ) -> Tensor[Int]:
        var rank = len(self._shape)

        # safe seed
        var s: UInt64
        if seed is None:
            s = UInt64(0xC0FFEE0012345678)
        else:
            s = UInt64(seed.value())
        var rng = XorShift64(s)

        # rank==0 → []
        if rank == 0:
            var shp = List[Int]()               # shape []
            return empty_tensor_with[Int](to_int_from_f64, Optional[List[Int]](shp.copy()))

        # num_samples <= 0 → shape [0] یا [batch,0]
        if num_samples <= 0:
            var shp0 = List[Int]()
            if rank == 1:
                shp0.append(0)
            else:
                shp0.append(self._shape[0])
                shp0.append(0)
            return empty_tensor_with[Int](to_int_from_f64, Optional[List[Int]](shp0.copy()))

        if rank == 1:
            var k = 0
            if len(self._shape) > 0:
                k = self._shape[0]
            if (replacement == False) and (num_samples > k):
                var shp_err = List[Int](); shp_err.append(0)
                return empty_tensor_with[Int](to_int_from_f64, Optional[List[Int]](shp_err.copy()))

            var out_shape = List[Int](); out_shape.append(num_samples)
            var out = empty_tensor_with[Int](to_int_from_f64, Optional[List[Int]](out_shape.copy()))

            var cdf = List[Float64]()
            _build_cdf_1d(self, 0, k, cdf)

            var i = 0
            while i < num_samples:
                var u = rng.next_unit_f64()
                var idx = _search_cdf(cdf, u)
                out._data[i] = idx
                i += 1
            return out.copy()
        else:
            # [batch, classes...] → classes = product(dims[1:])
            var batch = self._shape[0]
            var classes = 1
            var d = 1
            while d < len(self._shape):
                classes = classes * self._shape[d]
                d += 1

            if (replacement == False) and (num_samples > classes):
                var shp_err = List[Int](); shp_err.append(batch); shp_err.append(0)
                return empty_tensor_with[Int](to_int_from_f64, Optional[List[Int]](shp_err.copy()))

            var out_shape = List[Int](); out_shape.append(batch); out_shape.append(num_samples.copy())
            var out = empty_tensor_with[Int](to_int_from_f64, Optional[List[Int]](out_shape.copy()))

            var cdf = List[Float64]()
            var b = 0
            while b < batch:
                var row_start = b * classes
                _build_cdf_1d(self, row_start, classes, cdf)

                var t = 0
                while t < num_samples:
                    var u = rng.next_unit_f64()
                    var idx = _search_cdf(cdf, u)
                    out._data[b * num_samples + t] = idx
                    t += 1
                b += 1
            return out.copy()


    @always_inline
    fn one_hot(self: Tensor[Int], depth: Int) -> Tensor[Int]:
        # Works for any rank; flattens logical positions row-major.
        if depth <= 0:
            # Return an empty last-dim when depth<=0 (defensive).
            var shp = copy_ints(self._shape); shp.append(0)
            var strd = compute_row_major_strides(shp)
            return Tensor[Int](List[Int](), shp, strd, 0)

        # Fast path for rank==0 (scalar) → shape [depth]
        if len(self._shape) == 0:
            var cls = 0
            if len(self._data) != 0:
                cls = self._data[0]
            var buf = List[Int]()
            buf.reserve(depth)
            var d = 0
            while d < depth:
                buf.append(1 if (cls >= 0 and cls < depth and d == cls) else 0)
                d += 1
            var shp = List[Int]()
            shp.append(depth)
            var strd = compute_row_major_strides(shp)
            return Tensor[Int](buf, shp, strd, 0)

        # General path
        return one_hot_core_indices(self._data, self._shape, depth)


    # ------------------------------
    # Method on Tensor[Float64]
    # ------------------------------
    @always_inline
    fn one_hot(self: Tensor[Float64], depth: Int) -> Tensor[Int]:
        var shp = copy_ints(self._shape)
        var n = 1
        var i = 0
        while i < len(shp):
            n = n * shp[i]
            i += 1

        var ids = List[Int]()
        ids.reserve(n)
        i = 0
        var lim = len(self._data)
        if n < len(self._data):  lim = n
        while i < lim:
            ids.append(Int(self._data[i]))
            i += 1

        return one_hot_core_indices(ids, shp, depth)
    # ------------------------------
    # Method on Tensor[Float32]
    # ------------------------------
    @always_inline
    fn one_hot(self: Tensor[Float32], depth: Int) -> Tensor[Int]:
        var shp = copy_ints(self._shape)
        var n = 1
        var i = 0
        while i < len(shp):
            n = n * shp[i]
            i += 1

        var ids = List[Int]()
        ids.reserve(n)
        i = 0
        var lim = len(self._data)
        if n < len(self._data):  lim = n
        while i < lim:
            ids.append(Int(self._data[i]))
            i += 1

        return one_hot_core_indices(ids, shp, depth)

    # -------------------------------------------------------------------
    # to_list(): flattened row-major list (respects strides and offset)
    # -------------------------------------------------------------------
    @always_inline
    fn to_list(self: Tensor[Float64]) -> List[List[Float64]]:
        # Return a Python-style nested list for rank<=2.
        # For higher ranks, return a single-row flattened logical view.
        var r = len(self._shape)

        # Rank-0 (scalar) -> [[value]]
        if r == 0:
            var row = List[Float64]()
            row.append(self._data[self._offset])
            var outer0 = List[List[Float64]]()
            outer0.append(row.copy())
            return outer0.copy()

        # Rank-1 -> [ [a0, a1, ...] ]
        if r == 1:
            var n0 = self._shape[0]
            var s0 = self._strides[0]
            var base = self._offset

            var row1 = List[Float64]()
            var i = 0
            while i < n0:
                row1.append(self._data[base + i * s0])
                i += 1

            var outer1 = List[List[Float64]]()
            outer1.append(row1.copy())
            return outer1.copy()

        # Rank-2 -> [ [row0...], [row1...], ... ]
        if r == 2:
            var n0 = self._shape[0]
            var n1 = self._shape[1]
            var s0 = self._strides[0]
            var s1 = self._strides[1]
            var base = self._offset

            var out2 = List[List[Float64]]()
            var i = 0
            while i < n0:
                var row2 = List[Float64]()
                var j = 0
                while j < n1:
                    var lin = base + i * s0 + j * s1
                    row2.append(self._data[lin])
                    j += 1
                out2.append(row2.copy())
                i += 1
            return out2.copy()

        # Fallback for rank >= 3: build a logical flat row using strides
        var total = 1
        var d = 0
        while d < r:
            total = total * self._shape[d]
            d += 1

        var idx = List[Int]()
        idx.reserve(r)
        d = 0
        while d < r:
            idx.append(0)
            d += 1

        var flat = List[Float64]()
        flat.reserve(total)

        var done = False
        while True:
            var lin = self._offset
            d = 0
            while d < r:
                lin = lin + idx[d] * self._strides[d]
                d += 1
            flat.append(self._data[lin])

            # increment multi-index
            d = r - 1
            while True:
                if d < 0:
                    done = True
                    break
                idx[d] = idx[d] + 1
                if idx[d] < self._shape[d]:
                    break
                idx[d] = 0
                d = d - 1
            if done:
                break

        var outerN = List[List[Float64]]()
        outerN.append(flat.copy())
        return outerN.copy()


    @always_inline
    fn to_list(self: Tensor[Float32]) -> List[Float32]:
        var shp = self._shape.copy()
        var n = numel_shape(shp)
        var out = List[Float32]()
        out.reserve(n)

        var idx = List[Int]()
        var i = 0
        while i < n:
            unravel_index(i, shp, idx)
            var li = self._offset + lin_index(idx, self._strides)
            out.append(self._data[li])
            i += 1
        return out.copy()

    @always_inline
    fn to_list(self: Tensor[Int]) -> List[Int]:
        var shp = self._shape.copy()
        var n = numel_shape(shp)
        var out = List[Int]()
        out.reserve(n)

        var idx = List[Int]()
        var i = 0
        while i < n:
            unravel_index(i, shp, idx)
            var li = self._offset + lin_index(idx, self._strides)
            out.append(self._data[li])
            i += 1
        return out.copy()

    @always_inline
    fn to_list(self: Tensor[Bool]) -> List[Bool]:
        var shp = self._shape.copy()
        var n = numel_shape(shp)
        var out = List[Bool]()
        out.reserve(n)

        var idx = List[Int]()
        var i = 0
        while i < n:
            unravel_index(i, shp, idx)
            var li = self._offset + lin_index(idx, self._strides)
            out.append(self._data[li])
            i += 1
        return out.copy()

    @always_inline
    fn to_list(self: Tensor[String]) -> List[String]:
        var shp = self._shape.copy()
        var n = numel_shape(shp)
        var out = List[String]()
        out.reserve(n)

        var idx = List[Int]()
        var i = 0
        while i < n:
            unravel_index(i, shp, idx)
            var li = self._offset + lin_index(idx, self._strides)
            out.append(self._data[li])
            i += 1
        return out.copy()

    # -------------------------------------------------------------------
    # fill(): in-place set all elements in the current view
    # -------------------------------------------------------------------

    @always_inline
    fn fill(mut self: Tensor[Int], v: Int) -> None:
        fill[Int](self, v)

    @always_inline
    fn fill(mut self: Tensor[Float32], v: Float32) -> None:
        fill[Float32](self, v)

    @always_inline
    fn fill(mut self: Tensor[Float64], v: Float64) -> None:
        fill[Float64](self, v)



    fn masked_select(self, mask: Tensor[Int]) -> Tensor[T]:
        return masked_select(self, mask)


    fn masked_fill(mut self, mask: Tensor[Int],value: T):
        masked_fill(self, mask,value)

    @always_inline
    fn boolean_select(
        self: Tensor[T], mask: Tensor[Int]
    ) -> Tensor[T]:
        # scalar mask
        if len(mask._shape) == 0:
            var out_data = List[T]()
            if mask._data[mask._offset] != 0:
                # select all elements, flattened
                var n = 1
                var i = 0
                while i < len(self._shape):
                    n = n * self._shape[i]
                    i += 1
                out_data.reserve(n)
                if n > 0:
                    var rm_x = _row_major_multipliers(self._shape)
                    var k = 0
                    while k < n:
                        var offx = _offset_from_linear(self._shape, self._strides, self._offset, rm_x, k)
                        out_data.append(self._data[offx])
                        k += 1
            var shp = List[Int](); shp.append(len(out_data))
            var strd = compute_row_major_strides(shp)
            return Tensor[T](out_data, shp, strd, 0)

        # same-shape mask
        var r = len(self._shape)
        var r_m = len(mask._shape)

        # fast return on empty
        var nself = 1
        var i = 0
        while i < r:
            nself = nself * self._shape[i]
            i += 1
        if nself == 0:
            var empty = List[T]()
            var shp0 = List[Int](); shp0.append(0)
            var str0 = compute_row_major_strides(shp0)
            return Tensor[T](empty, shp0, str0, 0)

        # rows-major linear walks
        var rm_x = _row_major_multipliers(self._shape)
        var rm_m = _row_major_multipliers(mask._shape)

        var out_data = List[T]()
        out_data.reserve(nself)

        var lin = 0
        while lin < nself:
            var offx = _offset_from_linear(self._shape, self._strides, self._offset, rm_x, lin)

            var offm = 0
            if r_m == r:
                offm = _offset_from_linear(mask._shape, mask._strides, mask._offset, rm_m, lin)
            else:
                # fallback: treat non-matching mask as scalar false
                offm = mask._offset

            if mask._data[offm] != 0:
                out_data.append(self._data[offx])

            lin += 1

        var shp = List[Int](); shp.append(len(out_data))
        var strd = compute_row_major_strides(shp)
        return Tensor[T](out_data, shp, strd, 0)


    # In Tensor[T] impl block:

    @always_inline
    fn clamp(mut self: Tensor[Int], min_opt: Optional[Int], max_opt: Optional[Int]) -> None:
        clamp(self, min_opt, max_opt)
        return

    @always_inline
    fn clamp(mut self: Tensor[Float32], min_opt: Optional[Float32], max_opt: Optional[Float32]) -> None:
        clamp(self, min_opt, max_opt)
        return

    @always_inline
    fn clamp(mut self: Tensor[Float64], min_opt: Optional[Float64], max_opt: Optional[Float64]) -> None:
        clamp(self, min_opt, max_opt)
        return


    # Int -> Int
    @always_inline
    fn clamped(self: Tensor[Int], mn: Int, mx: Int) -> Tensor[Int]:
        var out = self.copy()
        out.clamp(some_i(mn), some_i(mx))
        return out.copy()

    # Int -> Float64 (promotion)
    @always_inline
    fn clamped(self: Tensor[Int], mn: Float64, mx: Float64) -> Tensor[Float64]:
        var yf64 = to_float64(self)
        yf64.clamp(some_f64(mn), some_f64(mx))
        return yf64.copy()

    # Float32 -> Float32
    @always_inline
    fn clamped(self: Tensor[Float32], mn: Float32, mx: Float32) -> Tensor[Float32]:
        var out = self.copy()
        out.clamp(some_f32(mn), some_f32(mx))
        return out.copy()

    # Float64 -> Float64
    @always_inline
    fn clamped(self: Tensor[Float64], mn: Float64, mx: Float64) -> Tensor[Float64]:
        var out = self.copy()
        out.clamp(some_f64(mn), some_f64(mx))
        return out.copy()


    @always_inline
    fn solve(self: Tensor[Float64], b: Tensor[Float64]) -> Tensor[Float64]:
        # Promote to Float64, solve, then cast back to Float32.
        var Af64 = to_float64(self)
        var bf64 = to_float64(b)
        var xf64 = solve(Af64,bf64)
        return  xf64.copy()

    @always_inline
    fn solve(self: Tensor[Float32], b: Tensor[Float32]) -> Tensor[Float32]:
        # Promote to Float64, solve, then cast back to Float32.
        var Af64 = to_float64(self)
        var bf64 = to_float64(b)
        var xf64 = solve(Af64,bf64)
        return to_float32(xf64)

    @always_inline
    fn solve(self: Tensor[Int], b: Tensor[Int]) -> Tensor[Float64]:
        # Promote to Float64 and solve in Float64; return Float64 for numerical safety.
        var Af64 = to_float64(self)
        var bf64 = to_float64(b)
        return solve(Af64,bf64)

    # Mixed-type overloads (common in demos)

    @always_inline
    fn solve(self: Tensor[Float64], b: Tensor[Int]) -> Tensor[Float64]:
        return solve(self,to_float64(b))

    @always_inline
    fn solve(self: Tensor[Float64], b: Tensor[Float32]) -> Tensor[Float64]:
        return solve(self,to_float64(b))
    @always_inline
    fn solve(self: Tensor[Float32], b: Tensor[Int]) -> Tensor[Float64]:
        return solve(to_float64(self),to_float64(b))

    @always_inline
    fn solve(self: Tensor[Float32], b: Tensor[Float64]) -> Tensor[Float64]:
        return solve(to_float64(self),b)

    @always_inline
    fn solve(self: Tensor[Int], b: Tensor[Float64]) -> Tensor[Float64]:
        return solve(to_float64(self),b)

    @always_inline
    fn solve(self: Tensor[Int], b: Tensor[Float32]) -> Tensor[Float64]:
        return solve(to_float64(self),to_float64(b))


    # Promotion overloads
    @always_inline
    fn inv(self: Tensor[Float32]) -> Tensor[Float64]:
        return inv(to_float64(self))
    @always_inline
    fn inv(self: Tensor[Float64]) -> Tensor[Float64]:
        return inv((self))

    @always_inline
    fn inv(self: Tensor[Int]) -> Tensor[Float64]:
        return inv(to_float64(self))

    # Promotion overloads
    @always_inline
    fn qr(self: Tensor[Float32]) -> (Tensor[Float64], Tensor[Float64]):
        return qr(to_float64(self))

    @always_inline
    fn qr(self: Tensor[Int]) -> (Tensor[Float64], Tensor[Float64]):
        return qr(to_float64(self))
    @always_inline
    fn qr(self: Tensor[Float64]) -> (Tensor[Float64], Tensor[Float64]):
        return qr((self))

    # Promotions
    @always_inline
    fn svd(self: Tensor[Float32]) -> (Tensor[Float64], Tensor[Float64], Tensor[Float64]):
        return svd(to_float64(self))

    @always_inline
    fn svd(self: Tensor[Int]) -> (Tensor[Float64], Tensor[Float64], Tensor[Float64]):
        return svd(to_float64(self))
    # Promotions
    @always_inline
    fn svd(self: Tensor[Float64]) -> (Tensor[Float64], Tensor[Float64], Tensor[Float64]):
        return svd((self))


    # Promotions
    @always_inline
    fn cholesky(self: Tensor[Float32]) -> Tensor[Float64]:
        return cholesky(to_float64(self))

    @always_inline
    fn cholesky(self: Tensor[Int]) -> Tensor[Float64]:
        return cholesky(to_float64(self))
    @always_inline
    fn cholesky(self: Tensor[Float64]) -> Tensor[Float64]:
        return  cholesky(self)


    @staticmethod
    @always_inline
    fn cat(self: List[Tensor[T]], dim: Int) -> Tensor[T]:
        return cat(self, dim)


    @always_inline
    fn split_sizes(self: Tensor[T], sizes: List[Int], dim: Int) -> List[Tensor[T]]:
        return split_sizes(self, sizes, dim)

    @always_inline
    fn chunk(self: Tensor[T], chunks: Int, dim: Int) -> List[Tensor[T]]:
        return chunk(self, chunks, dim)

    @always_inline
    fn unbind(self: Tensor[T], dim: Int) -> List[Tensor[T]]:
        return unbind(self, dim)


    @always_inline
    fn row(self: Tensor[T], i: Int) -> Tensor[T]:
        return row(self, i)

    fn numel(self) -> Int:
        return numel(self._shape)

    fn reshape(self, new_shape: List[Int]) -> Tensor[T]:
        return reshape(self, new_shape)

    fn reshape_infer(self, new_shape: List[Int]) -> Tensor[T]:
        return reshape_infer(self, new_shape)

    fn reshape_like(self, other: Tensor[T]) -> Tensor[T]:
        return reshape_like(self, other)

    # Expand/truncate selector list to exactly ndim, fill full-axis as needed
    @always_inline
    fn pad_full_axes(self, sels: List[IndexSel]) -> List[IndexSel]:
        var out = List[IndexSel]()
        var d = len(self._shape)
        var i = 0
        while i < len(sels) and i < d:
            out.append(clone_sel(sels[i]))
            i = i + 1
        while i < d:
            out.append(self.all_axis())
            i = i + 1
        return out.copy()

    fn equal(self: Tensor[Int], other: Tensor[Int]) -> Tensor[Int]:
      return self.__eq__(other)
    fn equal(self: Tensor[Float64], other: Tensor[Float64]) -> Tensor[Int]:
      return self.__eq__(other)
    fn equal(self: Tensor[Float32], other: Tensor[Float32]) -> Tensor[Int]:
      return self.__eq__(other)


     # ---------- helpers (private) ----------
    @always_inline
    fn _compute_strides_(self, shape: List[Int]) -> List[Int]:
        # Row-major contiguous strides
        var n = len(shape)
        var strides = List[Int]()
        strides.reserve(n)
        var i = 0
        while i < n:
            strides.push_back(0)
            i = i + 1
        if n == 0:
            return strides
        strides[n - 1] = 1
        var j = n - 2
        while j >= 0:
            strides[j] = strides[j + 1] * shape[j + 1]
            if j == 0: break
            j = j - 1
        return strides

    @always_inline
    fn _offset_(self, strides: List[Int], idx: List[Int]) -> Int:
        var n = len(idx)
        var off = 0
        var i = 0
        while i < n:
            off = off + idx[i] * strides[i]
            i = i + 1
        return off

   # ---------- transpose 2D ----------
    fn transpose2d(x: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
        var shp = x.shape()
        var rank = len(shp)
        if rank != 2:
            var same = tensor.zeros(shp)   # same shape, zeros
            # copy data
            var n = len(x._data)
            var t = 0
            while t < n:
                same._data[t] = x._data[t]
                t = t + 1
            return same.copy()

        var h = shp[0]
        var w = shp[1]
        var out = tensor.zeros([w, h])

        var i = 0
        while i < h:
            var j = 0
            while j < w:
                # out[j, i] = x[i, j]
                out._data[j * h + i] = x._data[i * w + j]
                j = j + 1
            i = i + 1
        return out.copy()






@always_inline
fn _normalize_bounds_f64(min_opt: Optional[Float64], max_opt: Optional[Float64]) -> (Optional[Float64], Optional[Float64]):
    if (min_opt is not None) and (max_opt is not None):
        var a = min_opt.value()
        var b = max_opt.value()
        if a > b:
            var t = a
            a = b
            b = t
        return (Optional[Float64](a), Optional[Float64](b))
    return (min_opt, max_opt)

@always_inline
fn _normalize_bounds_f32(min_opt: Optional[Float32], max_opt: Optional[Float32]) -> (Optional[Float32], Optional[Float32]):
    if (min_opt is not None) and (max_opt is not None):
        var a = min_opt.value()
        var b = max_opt.value()
        if a > b:
            var t = a
            a = b
            b = t
        return (Optional[Float32](a), Optional[Float32](b))
    return (min_opt, max_opt)

@always_inline
fn _normalize_bounds_i(min_opt: Optional[Int], max_opt: Optional[Int]) -> (Optional[Int], Optional[Int]):
    if (min_opt is not None) and (max_opt is not None):
        var a = min_opt.value()
        var b = max_opt.value()
        if a > b:
            var t = a
            a = b
            b = t
        return (Optional[Int](a), Optional[Int](b))
    return (min_opt, max_opt)


# Internal: build normalized CDF for a 1D slice [start, start+K)
@always_inline
fn _build_cdf_1d(
    p: Tensor[Float64],
    start: Int,
    k: Int,
    mut out_cdf: List[Float64]
) -> Float64:
    out_cdf.clear()
    out_cdf.reserve(k)

    var sum_p = 0.0
    var i = 0
    while i < k:
        var v = p._data[start + i]
        if v < 0.0:
            v = 0.0
        sum_p += v
        out_cdf.append(sum_p)
        i += 1

    if sum_p <= 0.0:
        # Fallback: uniform probs if all zeros/negatives
        out_cdf.clear()
        var j = 0
        while j < k:
            out_cdf.append(Float64(j + 1))
            j += 1
        sum_p = Float64(k)
    else:
        # Normalize to 1.0 (prevent drift)
        var t = 0
        while t < k:
            out_cdf[t] = out_cdf[t] / sum_p
            t += 1

    return sum_p




# Internal: binary search over CDF in [0,1)
@always_inline
fn _search_cdf(cdf: List[Float64], u: Float64) -> Int:
    var lo = 0
    var hi = len(cdf) - 1
    while lo < hi:
        var mid = (lo + hi) >> 1
        if u > cdf[mid]:
            lo = mid + 1
        else:
            hi = mid
    return lo


@always_inline
fn _fmt_int(x: Int) -> String:
    return x.__str__()


@always_inline
fn _fmt_i16(x: Int16) -> String:
    return x.__str__()

@always_inline
fn _fmt_i32(x: Int32) -> String:
    return x.__str__()

@always_inline
fn _fmt_i64(x: Int64) -> String:
    return x.__str__()

@always_inline
fn _fmt_f32(x: Float32) -> String:
    return x.__str__()

@always_inline
fn _fmt_f64(x: Float64) -> String:
    return x.__str__()

@always_inline
fn _fmt_bool(x: Bool) -> String:
    return x.__str__()

@always_inline
fn _fmt_str(x: String) -> String:
    return x
@always_inline
fn _row_major_multipliers(shape: List[Int]) -> List[Int]:
    var r = len(shape)
    var rm = List[Int]()
    rm.reserve(r)
    var i = 0
    while i < r:
        rm.append(0)
        i += 1
    var acc = 1
    var k = r - 1
    while k >= 0:
        rm[k] = acc
        acc = acc * shape[k]
        k -= 1
    return rm.copy()

@always_inline
fn _offset_from_linear(
    shape: List[Int], strides: List[Int], off: Int, rm: List[Int], lin: Int
) -> Int:
    var r = len(shape)
    var o = off
    var d = 0
    while d < r:
        var s = 0
        if shape[d] != 0 and rm[d] != 0:
            s = (lin // rm[d]) % shape[d]
        o = o + s * strides[d]
        d += 1
    return o
