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
 
 
from momijo.tensor.transform import reshape,transpose
from momijo.tensor.broadcast import broadcast_shapes,can_broadcast_shapes
 
from momijo.tensor.broadcast import matmul as _matmul_free
from momijo.tensor.broadcast import tensordot as _tensordot_free
from momijo.tensor.math import mean as _mean_free

from momijo.tensor.math import sum as _sum_free 
from momijo.tensor.math import *
from momijo.tensor.cast import *
from momijo.tensor.creation import scalar64,scalar32,scalar_int
from momijo.tensor.indexing import *
from momijo.tensor.transform import flatten,view ,clamp_int
# ========================= slice primitives =========================
# --- SliceSpec and IndexSel as tuples (value types) ---
 




# =============================== Tensor ===============================
struct Tensor[T: ImplicitlyCopyable & Copyable & Movable](Copyable, Movable):
    var _data: List[T]
    var _shape: List[Int]
    var _strides: List[Int]
    var _offset: Int

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
    fn all(self) -> IndexSel:
        # Full axis: start=0, stop=dim will be normalized later per-axis
        return make_slice_sel((0, 0, 1))  # stop will be filled from axis dim

    @always_inline
    fn take(self, js: List[Int]) -> IndexSel:
        return make_fancy_sel(js)

    # Expand/truncate selector list to exactly ndim, fill full-axis as needed
    @always_inline
    fn pad_full_axes(self, sels: List[IndexSel]) -> List[IndexSel]:
        var out = List[IndexSel]()
        var d = len(self._shape)
        var i = 0
        while i < len(sels) and i < d:
            out.append(sels[i].copy())
            i = i + 1
        while i < d:
            out.append(self.all())
            i = i + 1
        return out.copy()

    # Normalize a slice selector against dim and (possibly) negative step.
    # Returns (start, stop, step) normalized s.t. resulting axis length >= 0.
    @always_inline
    fn _normalize_slice(self, dim: Int, st0: Int, sp0: Int, step0: Int) -> (Int, Int, Int):
        var step = step0
        if step == 0:
            step = 1

        var start = st0
        var stop  = sp0

        if step > 0:
            # Positive step defaults (NumPy-like):
            # If stop token was empty in parsing (represented here as 0), take stop = dim.
            # NOTE: This also makes "0:" behave as full to end, which is desirable.
            if stop == 0:
                stop = dim

            # Normalize negatives
            if start < 0: start = dim + start
            if stop  < 0: stop  = dim + stop

            # Clamp into [0, dim]
            start = clamp_int(start, 0, dim)
            stop  = clamp_int(stop,  0, dim)

        else:
            # Negative step defaults:
            # If start token was empty => start = dim - 1
            # If stop token was empty  => stop  = -1 (exclusive, one before 0)
            if start == 0 and st0 == 0:
                start = dim - 1
            if stop == 0 and sp0 == 0:
                stop = -1

            # Normalize negatives
            if start < 0: start = dim + start
            if stop  < 0: stop  = dim + stop

            # Clamp allowing -1 on stop and [-1, dim-1] on start
            start = clamp_int(start, -1, dim - 1)
            stop  = clamp_int(stop,  -1, dim)

        return (start, stop, step)

    # =========================== Core: select_view ===========================
    # Build a ZERO-COPY view if possible (only indices and slices). Fancy indices force copy.
    fn select_view(self, sels_in: List[IndexSel]) -> Tensor[T]:
        var sels = self.pad_full_axes(sels_in)
        var ndim = len(self._shape)

        var new_shape = List[Int]()
        var new_strides = List[Int]()
        var new_off = self._offset

        var ax = 0
        while ax < ndim:
            var dim = self._shape[ax]
            if is_index(sels[ax]):
                var ii = wrap_axis_index(get_index(sels[ax]), dim)
                new_off = new_off + ii * self._strides[ax]
                # drop axis
            elif is_slice(sels[ax]):
                var (st, sp, stp) = get_slice(sels[ax])
                var (ns, np, nstep) = self._normalize_slice(dim, st, sp, stp)
                var out_len = axis_len_from_slice(ns, np, nstep)
                new_off = new_off + ns * self._strides[ax]
                new_shape.append(out_len)
                new_strides.append(self._strides[ax] * nstep)
            else:
                # Fancy index ⇒ cannot be a pure view; fall back to copy path
                return self.select_copy(sels)
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
        # If selection is viewable (no fancy), return zero-copy view; else copy.
        var viewable = True
        var sels = self.pad_full_axes(sels_in)
        var i = 0
        while i < len(sels):
            if is_fancy(sels[i]):
                viewable = False
                break
            i = i + 1
        if viewable:
            return self.select_view(sels)
        return self.select_copy(sels)

    # =========================== __getitem__ family ===========================

    # NumPy-like: single Int on an N-D tensor drops axis-0 and returns a ZERO-COPY VIEW.
    # - If rank==1, the result is a 0-D tensor (shape=[]). If you want a plain scalar,
    #   call `get_nd([i])` or add a separate helper, but we keep operator semantics uniform.
    fn __getitem__(self, i: Int) -> Tensor[T]:
        var d = len(self._shape)
        if d == 0:
            # Return an empty 0-D view
            var shp = List[Int]()
            var strd = List[Int]()
            return self._make_view(shp, strd, self._offset)
        var ii = wrap_axis_index(i, self._shape[0])
        var new_off = self._offset + ii * self._strides[0]
        # drop axis-0
        var new_shape = List[Int]()
        var new_strides = List[Int]()
        var ax = 1
        while ax < d:
            new_shape.append(self._shape[ax])
            new_strides.append(self._strides[ax])
            ax = ax + 1
        return self._make_view(new_shape, new_strides, new_off)

    # Convenience setter: single-Int on axis-0; sets the selected sub-array to a scalar.
    fn __setitem__(mut self, i: Int, value: T) -> None:
        var sel0 = make_index_sel(i)
        var d = len(self._shape)
        var sels = List[IndexSel]()
        sels.reserve(d)
        sels.append(sel0.copy())
        var ax = 1
        while ax < d:
            sels.append(self.all())
            ax = ax + 1
        self.select_set_scalar(sels, value)


    # String parser: supports tokens per axis separated by commas.
    # Forms: ":", "1", "-1", "1:3", "::2", "5:1:-1". (No per-axis fancy list in string.)
    fn _parse_selector_string(self, spec: String) -> List[IndexSel]:
        var sels = List[IndexSel]()
        var tokens = spec.split(",")
        var n = len(tokens)
        var ax = 0
        while ax < n:
            var raw = trim_ascii(to_string_owned(tokens[ax]))
            if len(raw) == 0 or raw == ":":
                sels.append(self.all())
                ax = ax + 1
                continue
            var colon_pos = raw.find(":")
            if colon_pos >= 0:
                var parts = raw.split(":")
                var p0 = String("")
                var p1 = String("")
                var p2 = String("")
                if len(parts) >= 1: p0 = trim_ascii(to_string_owned(parts[0]))
                if len(parts) >= 2: p1 = trim_ascii(to_string_owned(parts[1]))
                if len(parts) >= 3: p2 = trim_ascii(to_string_owned(parts[2]))
                var st = 0
                var sp = 0
                var stp = 1
                if len(p2) != 0:
                    var (ok_st, vst) = parse_int_safe(p2)
                    if ok_st: stp = vst
                    if stp == 0: stp = 1
                if len(p0) != 0:
                    var (ok0, v0) = parse_int_safe(p0)
                    if ok0: st = v0
                if len(p1) != 0:
                    var (ok1, v1) = parse_int_safe(p1)
                    if ok1: sp = v1
                sels.append(make_slice_sel((st, sp, stp)))
            else:
                var ii = 0
                var (ok_i, v_i) = parse_int_safe(raw)
                if ok_i: ii = v_i
                sels.append(make_index_sel(ii))
            ax = ax + 1
        return sels.copy()


    # Get via string (prefers view)
    fn __getitem__(self, spec: String) -> Tensor[T]:
        var sels = self._parse_selector_string(spec)
        return self.select(sels)

    # Apply one selector to axis 0, full on others (prefers view)
    fn __getitem__(self, sel: IndexSel) -> Tensor[T]:
        var d = len(self._shape)
        var sels = List[IndexSel]()
        sels.reserve(d)
        sels.append(sel.copy())
        var ax = 1
        while ax < d:
            sels.append(self.all())
            ax = ax + 1
        return self.select(sels)

    # Fancy indexing on axis 0: x[[i0,i1,...]]
    fn __getitem__(self, idxs: List[Int]) -> Tensor[T]:
        var sel0 = make_fancy_sel(idxs)
        return self.__getitem__(sel0)

    @always_inline
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

    # 2-int → view (rank-2 drop). If exactly rank==2, this is a 0-D tensor.
    fn __getitem__(self: Tensor[T], i0: Int, i1: Int) -> Tensor[T]:
        var idxs = List[Int](); idxs.append(i0); idxs.append(i1)
        return self._view_after_prefix_ints(idxs)

    # 3-int
    fn __getitem__(self: Tensor[T], i0: Int, i1: Int, i2: Int) -> Tensor[T]:
        var idxs = List[Int](); idxs.append(i0); idxs.append(i1); idxs.append(i2)
        return self._view_after_prefix_ints(idxs)

    # 4-int
    fn __getitem__(self: Tensor[T], i0: Int, i1: Int, i2: Int, i3: Int) -> Tensor[T]:
        var idxs = List[Int]()
        idxs.append(i0); idxs.append(i1); idxs.append(i2); idxs.append(i3)
        return self._view_after_prefix_ints(idxs)

    # 5-int
    fn __getitem__(self: Tensor[T], i0: Int, i1: Int, i2: Int, i3: Int, i4: Int) -> Tensor[T]:
        var idxs = List[Int]()
        idxs.append(i0); idxs.append(i1); idxs.append(i2); idxs.append(i3); idxs.append(i4)
        return self._view_after_prefix_ints(idxs)


    # Mixed IndexSel up to 5D (prefers view)
    fn __getitem__(self, a: IndexSel, b: IndexSel) -> Tensor[T]:
        var sels = List[IndexSel]()
        sels.append(a.copy()); sels.append(b.copy())
        return self.select(sels)

    fn __getitem__(self, a: IndexSel, b: IndexSel, c: IndexSel) -> Tensor[T]:
        var sels = List[IndexSel]()
        sels.append(a.copy()); sels.append(b.copy()); sels.append(c.copy())
        return self.select(sels)

    fn __getitem__(self, a: IndexSel, b: IndexSel, c: IndexSel, d: IndexSel) -> Tensor[T]:
        var sels = List[IndexSel]()
        sels.append(a.copy()); sels.append(b.copy()); sels.append(c.copy()); sels.append(d.copy())
        return self.select(sels)

    fn __getitem__(self, a: IndexSel, b: IndexSel, c: IndexSel, d: IndexSel, e: IndexSel) -> Tensor[T]:
        var sels = List[IndexSel]()
        sels.append(a.copy()); sels.append(b.copy()); sels.append(c.copy()); sels.append(d.copy()); sels.append(e.copy())
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


    # ---------------- convenience setitem overloads ----------------

    fn __setitem__(mut self, spec: String, value: T) -> None:
        var sels = self._parse_selector_string(spec)
        self.select_set_scalar(sels, value)

    fn __setitem__(mut self, sel: IndexSel, value: T) -> None:
        var d = len(self._shape)
        var sels = List[IndexSel]()
        sels.reserve(d)
        sels.append(sel.copy())
        var ax = 1
        while ax < d:
            sels.append(self.all())
            ax = ax + 1
        self.select_set_scalar(sels, value)

    fn __setitem__(mut self, spec: String, src: Tensor[T]) -> None:
        var sels = self._parse_selector_string(spec)
        self.select_set_tensor(sels, src)

    fn __setitem__(mut self, sel: IndexSel, src: Tensor[T]) -> None:
        var d = len(self._shape)
        var sels = List[IndexSel]()
        sels.reserve(d)
        sels.append(sel.copy())
        var ax = 1
        while ax < d:
            sels.append(self.all())
            ax = ax + 1
        self.select_set_tensor(sels, src)

    fn __setitem__(mut self, idxs: List[Int], src: Tensor[T]) -> None:
        var s0 = make_fancy_sel(idxs)
        self.__setitem__(s0, src)

    # ---------- Introspection ---------- 

    fn len(self) -> Int:
        return numel(self._shape)

    # rank (number of dimensions)
    fn ndim(self) -> Int:
        return len(self._shape)

    fn strides(self) -> List[Int]:
        return self._strides.copy()

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

    # --------------------------- slice builders ------------------------
    fn full(self, axis: Int) -> SliceSpec:
        var dim = self._shape[normalize_axis(axis, len(self._shape))]
        return SliceSpec(start=0, stop=dim, step=1)

    fn range(self, axis: Int, start: Int, stop: Int, step: Int = 1) -> SliceSpec:
        var d = len(self._shape)
        var ax = normalize_axis(axis, d)
        var dim = self._shape[ax]
        var s = wrap_index(start, dim)
        var e = stop
        if e < 0:
            e = dim + e
        if e > dim:
            e = dim
        if s < 0:
            s = 0
        if s > dim:
            s = dim
        if e < 0:
            e = 0
        return SliceSpec(start=s, stop=e, step=step)

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
        return out

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
 
 

    fn numel(self) -> Int:
        return numel(self._shape)

    fn reshape(self, new_shape: List[Int]) -> Tensor[T]:
        return reshape(self, new_shape)

    fn reshape_infer(self, new_shape: List[Int]) -> Tensor[T]:
        return reshape_infer(self, new_shape)

    fn resize_like_with_pad(self, new_tensor: Tensor[T]) -> Tensor[T]:
        return resize_like_with_pad(self, new_shape)

    fn transpose(self, new_shape: List[Int]) -> Tensor[T]:
        return transpose(self, new_shape)
 

    fn matmul(self: Tensor[Float64], x: Tensor[Float64]) -> Tensor[Float64]:
        return _matmul_free(self, x)

    # Tensor[Int] @ Tensor[Int] -> Tensor[Float64]  (upcast then call Float64 impl)
    fn matmul(self: Tensor[Int], x: Tensor[Int]) -> Tensor[Float64]:
        var Af = astype_f64_from_int(self)
        var xf = astype_f64_from_int(x)
        return _matmul_free(Af, xf)


    # Float64 x Float64
    fn tensordot(self: Tensor[Float64], B: Tensor[Float64], axis: Int = 1) -> Tensor[Float64]:
        return _tensordot_free(self, B, axis)

    
    fn tensordot(self: Tensor[Int], B: Tensor[Int], axis: Int = 1) -> Tensor[Float64]:
        var Af = astype_f64_from_int(self)
        var Bf = astype_f64_from_int(B)
        return _tensordot_free(Af, Bf, axis)

     # Inside: struct Tensor[T: ImplicitlyCopyable & Copyable & Movable]:
    fn sum(self: Tensor[Float64], axis: Optional[Int] = None, keepdims: Bool = False) -> Tensor[Float64]:
        return _sum_free(self, axis, keepdims)
        # Inside: struct Tensor[T: ImplicitlyCopyable & Copyable & Movable]:
    fn sum(self: Tensor[Int], axis: Optional[Int], keepdims: Bool = False) -> Tensor[Float64]:
        var f = astype_f64_from_int(self)
        return _sum_free(f, axis, keepdims)

 

    # Tensor[Float64] -> Float64
    fn mean(self: Tensor[Float64], axis: Int = 1) -> Tensor[Float64]:
        var ax = Optional[Int](axis)      # Some(axis)
        return _mean_free(self, ax, False)

    # Also handy to have a whole-tensor mean overload
    fn mean(self: Tensor[Float64]) -> Tensor[Float64]:
        var none = Optional[Int]()        # None
        return _mean_free(self, none, False)

    # Tensor[Int] -> upcast to Float64 then call the Float64 free mean
    fn mean(self: Tensor[Int], axis: Int = 1) -> Tensor[Float64]:
        var A = astype_f64_from_int(self)
        var ax = Optional[Int](axis)
        return _mean_free(A, ax, False)

    fn mean(self: Tensor[Int]) -> Tensor[Float64]:
        var A = astype_f64_from_int(self)
        var none = Optional[Int]()
        return _mean_free(A, none, False)


 

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

    # ---- Int → Int (already have the Int unique pipeline) ----
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
         
    @always_inline
    fn unique(self: Tensor[Int]) -> UniqueResult:
        return tensor_unique_int(self)


    @always_inline
    fn sort(self: Tensor[Int]) -> Tensor[Int]:
        return tensor_sort_int(self)
 

    @always_inline
    fn bincount(self: Tensor[Int]) -> Tensor[Int]:
        return tensor_bincount_int(self)

    @always_inline
    fn histogram(self: Tensor[Int], bins: List[Int]) -> UniqueResult:
        return tensor_histogram_int(self, bins)

    @always_inline
    fn digitize(self: Tensor[Int], edges: List[Int]) -> Tensor[Int]:
        return tensor_digitize_int(self, edges)
 

    @always_inline
    fn sum(self: Tensor[Int]) -> Int:
        return sum1d_unrolled(self)

    @always_inline
    fn sum(self: Tensor[Float64]) -> Float64:
        return sum1d_unrolled(self) 

    @always_inline
    fn sum(self: Tensor[Float32]) -> Float32:
        return sum1d_unrolled(self) 
 


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
    fn clip(self: Tensor[Float64], lo: Float64, hi: Float64) -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(self, scalar64(hi)), self, scalar64(hi)), scalar64(lo)), where_f64(le_t(self, scalar64(hi)), self, scalar64(hi)), scalar64(lo))     # clip f64,f64
    fn clip(self: Tensor[Float64], lo: Float64, hi: Float32) -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(self, to_float64(scalar32(hi))), self, to_float64(scalar32(hi))), scalar64(lo)), where_f64(le_t(self, to_float64(scalar32(hi))), self, to_float64(scalar32(hi))), scalar64(lo)) # clip f64,f32→f64
    fn clip(self: Tensor[Float64], lo: Float64, hi: Int)     -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(self, to_float64(scalar_int(hi))), self, to_float64(scalar_int(hi))), scalar64(lo)), where_f64(le_t(self, to_float64(scalar_int(hi))), self, to_float64(scalar_int(hi))), scalar64(lo)) # clip f64,int→f64
    fn clip(self: Tensor[Float64], lo: Float32, hi: Float64) -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(self, scalar64(hi)), self, scalar64(hi)), to_float64(scalar32(lo))), where_f64(le_t(self, scalar64(hi)), self, scalar64(hi)), to_float64(scalar32(lo)))                                 # clip f32→f64,f64
    fn clip(self: Tensor[Float64], lo: Float32, hi: Float32) -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(self, to_float64(scalar32(hi))), self, to_float64(scalar32(hi))), to_float64(scalar32(lo))), where_f64(le_t(self, to_float64(scalar32(hi))), self, to_float64(scalar32(hi))), to_float64(scalar32(lo))) # clip f32→f64,f32→f64
    fn clip(self: Tensor[Float64], lo: Float32, hi: Int)     -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(self, to_float64(scalar_int(hi))), self, to_float64(scalar_int(hi))), to_float64(scalar32(lo))), where_f64(le_t(self, to_float64(scalar_int(hi))), self, to_float64(scalar_int(hi))), to_float64(scalar32(lo)))     # clip f32→f64,int→f64
    fn clip(self: Tensor[Float64], lo: Int,     hi: Float64) -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(self, scalar64(hi)), self, scalar64(hi)), to_float64(scalar_int(lo))), where_f64(le_t(self, scalar64(hi)), self, scalar64(hi)), to_float64(scalar_int(lo)))                                 # clip int→f64,f64
    fn clip(self: Tensor[Float64], lo: Int,     hi: Float32) -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(self, to_float64(scalar32(hi))), self, to_float64(scalar32(hi))), to_float64(scalar_int(lo))), where_f64(le_t(self, to_float64(scalar32(hi))), self, to_float64(scalar32(hi))), to_float64(scalar_int(lo)))         # clip int→f64,f32→f64
    fn clip(self: Tensor[Float64], lo: Int,     hi: Int)     -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(self, to_float64(scalar_int(hi))), self, to_float64(scalar_int(hi))), to_float64(scalar_int(lo))), where_f64(le_t(self, to_float64(scalar_int(hi))), self, to_float64(scalar_int(hi))), to_float64(scalar_int(lo)))         # clip int→f64,int→f64

    fn minimum_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Float64]: return where_f64(le_t(self, scalar64(s)), self, scalar64(s))                                   # min f64
    fn minimum_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Float64]: return where_f64(le_t(self, to_float64(scalar32(s))), self, to_float64(scalar32(s)))           # min f32→f64
    fn minimum_scalar(self: Tensor[Float64], s: Int)     -> Tensor[Float64]: return where_f64(le_t(self, to_float64(scalar_int(s))), self, to_float64(scalar_int(s)))       # min int→f64

    fn maximum_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Float64]: return where_f64(ge_t(self, scalar64(s)), self, scalar64(s))                                   # max f64
    fn maximum_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Float64]: return where_f64(ge_t(self, to_float64(scalar32(s))), self, to_float64(scalar32(s)))           # max f32→f64
    fn maximum_scalar(self: Tensor[Float64], s: Int)     -> Tensor[Float64]: return where_f64(ge_t(self, to_float64(scalar_int(s))), self, to_float64(scalar_int(s)))       # max int→f64

    fn min(self: Tensor[Float64]) -> Float64: return min(self)                                   # min f64
    fn max(self: Tensor[Float64]) -> Float64: return max(self)                                   # max f64
    

 
    fn can_broadcast_with(self: Tensor[Float64], other: Tensor[Float64]) -> Bool:
        return can_broadcast_shapes(self._shape, other._shape)
 
    fn can_broadcast_with(self: Tensor[Float64], other: Tensor[Float32]) -> Bool:
        return can_broadcast_shapes(self._shape, other._shape)
 
    fn can_broadcast_with(self: Tensor[Float64], other: Tensor[Int]) -> Bool:
        return can_broadcast_shapes(self._shape, other._shape)

    fn can_broadcast_with(self: Tensor[Float32], other: Tensor[Float64]) -> Bool:
        return can_broadcast_shapes(self._shape, other._shape)

    fn can_broadcast_with(self: Tensor[Float32], other: Tensor[Float32]) -> Bool:
        return can_broadcast_shapes(self._shape, other._shape)

    fn can_broadcast_with(self: Tensor[Float32], other: Tensor[Int]) -> Bool:
        return can_broadcast_shapes(self._shape, other._shape)

    fn can_broadcast_with(self: Tensor[Int], other: Tensor[Float64]) -> Bool:
        return can_broadcast_shapes(self._shape, other._shape)

    fn can_broadcast_with(self: Tensor[Int], other: Tensor[Float32]) -> Bool:
        return can_broadcast_shapes(self._shape, other._shape)

    fn can_broadcast_with(self: Tensor[Int], other: Tensor[Int]) -> Bool:
        return can_broadcast_shapes(self._shape, other._shape)

    # ----------------------
    # self: Tensor[Float32]
    # ----------------------
    # clip → if any Float64 is present, output Float64; else output Float32
    fn clip(self: Tensor[Float32], lo: Float64, hi: Float64) -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(to_float64(self), scalar64(hi)), to_float64(self), scalar64(hi)), scalar64(lo)), where_f64(le_t(to_float64(self), scalar64(hi)), to_float64(self), scalar64(hi)), scalar64(lo))                         # f64,f64 ⇒ f64
    fn clip(self: Tensor[Float32], lo: Float64, hi: Float32) -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(to_float64(self), to_float64(scalar32(hi))), to_float64(self), to_float64(scalar32(hi))), scalar64(lo)), where_f64(le_t(to_float64(self), to_float64(scalar32(hi))), to_float64(self), to_float64(scalar32(hi))), scalar64(lo)) # f64,f32 ⇒ f64
    fn clip(self: Tensor[Float32], lo: Float64, hi: Int)     -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(to_float64(self), to_float64(scalar_int(hi))), to_float64(self), to_float64(scalar_int(hi))), scalar64(lo)), where_f64(le_t(to_float64(self), to_float64(scalar_int(hi))), to_float64(self), to_float64(scalar_int(hi))), scalar64(lo))     # f64,int ⇒ f64
    fn clip(self: Tensor[Float32], lo: Float32, hi: Float64) -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(to_float64(self), scalar64(hi)), to_float64(self), scalar64(hi)), to_float64(scalar32(lo))), where_f64(le_t(to_float64(self), scalar64(hi)), to_float64(self), scalar64(hi)), to_float64(scalar32(lo)))                 # f32,f64 ⇒ f64
    fn clip(self: Tensor[Float32], lo: Int,     hi: Float64) -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(to_float64(self), scalar64(hi)), to_float64(self), scalar64(hi)), to_float64(scalar_int(lo))), where_f64(le_t(to_float64(self), scalar64(hi)), to_float64(self), scalar64(hi)), to_float64(scalar_int(lo)))                 # int,f64 ⇒ f64

    fn clip(self: Tensor[Float32], lo: Float32, hi: Float32) -> Tensor[Float32]: return where_f32(ge_t(where_f32(le_t(self, scalar32(hi)), self, scalar32(hi)), scalar32(lo)), where_f32(le_t(self, scalar32(hi)), self, scalar32(hi)), scalar32(lo))     # f32,f32 ⇒ f32
    fn clip(self: Tensor[Float32], lo: Float32, hi: Int)     -> Tensor[Float32]: return where_f32(ge_t(where_f32(le_t(self, to_float32(scalar_int(hi))), self, to_float32(scalar_int(hi))), scalar32(lo)), where_f32(le_t(self, to_float32(scalar_int(hi))), self, to_float32(scalar_int(hi))), scalar32(lo))                         # f32,int ⇒ f32
    fn clip(self: Tensor[Float32], lo: Int,     hi: Float32) -> Tensor[Float32]: return where_f32(ge_t(where_f32(le_t(self, scalar32(hi)), self, scalar32(hi)), to_float32(scalar_int(lo))), where_f32(le_t(self, scalar32(hi)), self, scalar32(hi)), to_float32(scalar_int(lo)))                                             # int,f32 ⇒ f32
    fn clip(self: Tensor[Float32], lo: Int,     hi: Int)     -> Tensor[Float32]: return where_f32(ge_t(where_f32(le_t(self, to_float32(scalar_int(hi))), self, to_float32(scalar_int(hi))), to_float32(scalar_int(lo))), where_f32(le_t(self, to_float32(scalar_int(hi))), self, to_float32(scalar_int(hi))), to_float32(scalar_int(lo)))         # int,int ⇒ f32? (promotion says Float32 since self is Float32)

    # min/max → if s is Float64 ⇒ Float64; else ⇒ Float32
    fn minimum_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Float64]: return where_f64(le_t(to_float64(self), scalar64(s)), to_float64(self), scalar64(s))                           # min s=f64 ⇒ f64
    fn maximum_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Float64]: return where_f64(ge_t(to_float64(self), scalar64(s)), to_float64(self), scalar64(s))                           # max s=f64 ⇒ f64
    fn minimum_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Float32]: return where_f32(le_t(self, scalar32(s)), self, scalar32(s))                                                  # min s=f32 ⇒ f32
    fn maximum_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Float32]: return where_f32(ge_t(self, scalar32(s)), self, scalar32(s))                                                  # max s=f32 ⇒ f32
    fn minimum_scalar(self: Tensor[Float32], s: Int)     -> Tensor[Float32]: return where_f32(le_t(self, to_float32(scalar_int(s))), self, to_float32(scalar_int(s)))                      # min s=int ⇒ f32
    fn maximum_scalar(self: Tensor[Float32], s: Int)     -> Tensor[Float32]: return where_f32(ge_t(self, to_float32(scalar_int(s))), self, to_float32(scalar_int(s)))                      # max s=int ⇒ f32


    fn min(self: Tensor[Float32]) -> Float32: return min(self)                                   
    fn max(self: Tensor[Float32]) -> Float32: return max(self)       
    # ----------------------
    # self: Tensor[Int]
    # ----------------------
    # clip → if any Float64 ⇒ Float64; else if any Float32 ⇒ Float32; else ⇒ Int
    # ⇒ 9 overloads grouped by target

    # target Float64 (any Float64 present)
    fn clip(self: Tensor[Int], lo: Float64, hi: Float64) -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(to_float64(self), scalar64(hi)), to_float64(self), scalar64(hi)), scalar64(lo)), where_f64(le_t(to_float64(self), scalar64(hi)), to_float64(self), scalar64(hi)), scalar64(lo))                                 # f64,f64 ⇒ f64
    fn clip(self: Tensor[Int], lo: Float64, hi: Float32) -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(to_float64(self), to_float64(scalar32(hi))), to_float64(self), to_float64(scalar32(hi))), scalar64(lo)), where_f64(le_t(to_float64(self), to_float64(scalar32(hi))), to_float64(self), to_float64(scalar32(hi))), scalar64(lo)) # f64,f32 ⇒ f64
    fn clip(self: Tensor[Int], lo: Float64, hi: Int)     -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(to_float64(self), to_float64(scalar_int(hi))), to_float64(self), to_float64(scalar_int(hi))), scalar64(lo)), where_f64(le_t(to_float64(self), to_float64(scalar_int(hi))), to_float64(self), to_float64(scalar_int(hi))), scalar64(lo))     # f64,int ⇒ f64
    fn clip(self: Tensor[Int], lo: Float32, hi: Float64) -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(to_float64(self), scalar64(hi)), to_float64(self), scalar64(hi)), to_float64(scalar32(lo))), where_f64(le_t(to_float64(self), scalar64(hi)), to_float64(self), scalar64(hi)), to_float64(scalar32(lo)))                 # f32,f64 ⇒ f64
    fn clip(self: Tensor[Int], lo: Int,     hi: Float64) -> Tensor[Float64]: return where_f64(ge_t(where_f64(le_t(to_float64(self), scalar64(hi)), to_float64(self), scalar64(hi)), to_float64(scalar_int(lo))), where_f64(le_t(to_float64(self), scalar64(hi)), to_float64(self), scalar64(hi)), to_float64(scalar_int(lo)))                 # int,f64 ⇒ f64

    # target Float32 (no Float64, but Float32 present)
    fn clip(self: Tensor[Int], lo: Float32, hi: Float32) -> Tensor[Float32]: return where_f32(ge_t(where_f32(le_t(to_float32(self), scalar32(hi)), to_float32(self), scalar32(hi)), scalar32(lo)), where_f32(le_t(to_float32(self), scalar32(hi)), to_float32(self), scalar32(hi)), scalar32(lo))                                 # f32,f32 ⇒ f32
    fn clip(self: Tensor[Int], lo: Float32, hi: Int)     -> Tensor[Float32]: return where_f32(ge_t(where_f32(le_t(to_float32(self), to_float32(scalar_int(hi))), to_float32(self), to_float32(scalar_int(hi))), scalar32(lo)), where_f32(le_t(to_float32(self), to_float32(scalar_int(hi))), to_float32(self), to_float32(scalar_int(hi))), scalar32(lo)) # f32,int ⇒ f32
    fn clip(self: Tensor[Int], lo: Int,     hi: Float32) -> Tensor[Float32]: return where_f32(ge_t(where_f32(le_t(to_float32(self), scalar32(hi)), to_float32(self), scalar32(hi)), to_float32(scalar_int(lo))), where_f32(le_t(to_float32(self), scalar32(hi)), to_float32(self), scalar32(hi)), to_float32(scalar_int(lo)))                 # int,f32 ⇒ f32

    # target Int (all Int)
    fn clip(self: Tensor[Int], lo: Int,     hi: Int)     -> Tensor[Int]: return where_int(ge_t(where_int(le_t(self, scalar_int(hi)), self, scalar_int(hi)), scalar_int(lo)), where_int(le_t(self, scalar_int(hi)), self, scalar_int(hi)), scalar_int(lo)) # int,int ⇒ int

    # min/max → result dtype is max(self, s)
    fn minimum_scalar(self: Tensor[Int], s: Float64) -> Tensor[Float64]: return where_f64(le_t(to_float64(self), scalar64(s)), to_float64(self), scalar64(s))             # min s=f64 ⇒ f64
    fn maximum_scalar(self: Tensor[Int], s: Float64) -> Tensor[Float64]: return where_f64(ge_t(to_float64(self), scalar64(s)), to_float64(self), scalar64(s))             # max s=f64 ⇒ f64
    fn minimum_scalar(self: Tensor[Int], s: Float32) -> Tensor[Float32]: return where_f32(le_t(to_float32(self), scalar32(s)), to_float32(self), scalar32(s))             # min s=f32 ⇒ f32
    fn maximum_scalar(self: Tensor[Int], s: Float32) -> Tensor[Float32]: return where_f32(ge_t(to_float32(self), scalar32(s)), to_float32(self), scalar32(s))             # max s=f32 ⇒ f32
    fn minimum_scalar(self: Tensor[Int], s: Int)     -> Tensor[Int]:     return where_int(le_t(self, scalar_int(s)), self, scalar_int(s))                               # min s=int ⇒ int
    fn maximum_scalar(self: Tensor[Int], s: Int)     -> Tensor[Int]:     return where_int(ge_t(self, scalar_int(s)), self, scalar_int(s))                               # max s=int ⇒ int

    fn min(self: Tensor[Int]) -> Int: return min(self)                                    
    fn max(self: Tensor[Int]) -> Int: return max(self)       


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
    fn __add__(self: Tensor[Float64], rhs: Float64)         -> Tensor[Float64]: return add_t(self, scalar64(rhs))                    # a + s(f64)
    fn __add__(self: Tensor[Float64], rhs: Float32)         -> Tensor[Float64]: return add_t(self, to_float64(scalar32(rhs)))        # a + s(f32→f64)
    fn __add__(self: Tensor[Float64], rhs: Int)             -> Tensor[Float64]: return add_t(self, to_float64(scalar_int(rhs)))      # a + s(int→f64)

    # - (result: Float64)
    fn __sub__(self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Float64]: return sub_t(self, rhs)                                # a - b(f64)
    fn __sub__(self: Tensor[Float64], rhs: Tensor[Float32]) -> Tensor[Float64]: return sub_t(self, to_float64(rhs))                   # a - b(f32→f64)
    fn __sub__(self: Tensor[Float64], rhs: Tensor[Int])     -> Tensor[Float64]: return sub_t(self, to_float64(rhs))                   # a - b(int→f64)
    fn __sub__(self: Tensor[Float64], rhs: Float64)         -> Tensor[Float64]: return sub_t(self, scalar64(rhs))                    # a - s(f64)
    fn __sub__(self: Tensor[Float64], rhs: Float32)         -> Tensor[Float64]: return sub_t(self, to_float64(scalar32(rhs)))        # a - s(f32→f64)
    fn __sub__(self: Tensor[Float64], rhs: Int)             -> Tensor[Float64]: return sub_t(self, to_float64(scalar_int(rhs)))      # a - s(int→f64)

    # * (result: Float64)
    fn __mul__(self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Float64]: return mul_t(self, rhs)                                # a * b(f64)
    fn __mul__(self: Tensor[Float64], rhs: Tensor[Float32]) -> Tensor[Float64]: return mul_t(self, to_float64(rhs))                   # a * b(f32→f64)
    fn __mul__(self: Tensor[Float64], rhs: Tensor[Int])     -> Tensor[Float64]: return mul_t(self, to_float64(rhs))                   # a * b(int→f64)
    fn __mul__(self: Tensor[Float64], rhs: Float64)         -> Tensor[Float64]: return mul_t(self, scalar64(rhs))                    # a * s(f64)
    fn __mul__(self: Tensor[Float64], rhs: Float32)         -> Tensor[Float64]: return mul_t(self, to_float64(scalar32(rhs)))        # a * s(f32→f64)
    fn __mul__(self: Tensor[Float64], rhs: Int)             -> Tensor[Float64]: return mul_t(self, to_float64(scalar_int(rhs)))      # a * s(int→f64)

    # / (result: Float64)
    fn __truediv__(self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Float64]: return div_t(self, rhs)                           # a / b(f64)
    fn __truediv__(self: Tensor[Float64], rhs: Tensor[Float32]) -> Tensor[Float64]: return div_t(self, to_float64(rhs))              # a / b(f32→f64)
    fn __truediv__(self: Tensor[Float64], rhs: Tensor[Int])     -> Tensor[Float64]: return div_t(self, to_float64(rhs))              # a / b(int→f64)
    fn __truediv__(self: Tensor[Float64], rhs: Float64)         -> Tensor[Float64]: return div_t(self, scalar64(rhs))               # a / s(f64)
    fn __truediv__(self: Tensor[Float64], rhs: Float32)         -> Tensor[Float64]: return div_t(self, to_float64(scalar32(rhs)))   # a / s(f32→f64)
    fn __truediv__(self: Tensor[Float64], rhs: Int)             -> Tensor[Float64]: return div_t(self, to_float64(scalar_int(rhs))) # a / s(int→f64)

    # % (result: Float64)
    fn __mod__(self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Float64]: return mod_t(self, rhs)                               # a % b(f64)
    fn __mod__(self: Tensor[Float64], rhs: Tensor[Float32]) -> Tensor[Float64]: return mod_t(self, to_float64(rhs))                  # a % b(f32→f64)
    fn __mod__(self: Tensor[Float64], rhs: Tensor[Int])     -> Tensor[Float64]: return mod_t(self, to_float64(rhs))                  # a % b(int→f64)
    fn __mod__(self: Tensor[Float64], rhs: Float64)         -> Tensor[Float64]: return mod_t(self, scalar64(rhs))                   # a % s(f64)
    fn __mod__(self: Tensor[Float64], rhs: Float32)         -> Tensor[Float64]: return mod_t(self, to_float64(scalar32(rhs)))       # a % s(f32→f64)
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
    fn __add__(self: Tensor[Float32], rhs: Float64)         -> Tensor[Float64]: return add_t(to_float64(self), scalar64(rhs))       # a(f32→f64) + s(f64)
    fn __add__(self: Tensor[Float32], rhs: Float32)         -> Tensor[Float32]: return add_t(self, scalar32(rhs))                   # a + s(f32)
    fn __add__(self: Tensor[Float32], rhs: Int)             -> Tensor[Float32]: return add_t(self, to_float32(scalar_int(rhs)))     # a + s(int→f32)

    # - (result: Float64 if rhs has Float64, else Float32)
    fn __sub__(self: Tensor[Float32], rhs: Tensor[Float64]) -> Tensor[Float64]: return sub_t(to_float64(self), rhs)                  # a(f32→f64) - b(f64)
    fn __sub__(self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Float32]: return sub_t(self, rhs)                              # a - b(f32)
    fn __sub__(self: Tensor[Float32], rhs: Tensor[Int])     -> Tensor[Float32]: return sub_t(self, to_float32(rhs))                  # a - b(int→f32)
    fn __sub__(self: Tensor[Float32], rhs: Float64)         -> Tensor[Float64]: return sub_t(to_float64(self), scalar64(rhs))       # a(f32→f64) - s(f64)
    fn __sub__(self: Tensor[Float32], rhs: Float32)         -> Tensor[Float32]: return sub_t(self, scalar32(rhs))                   # a - s(f32)
    fn __sub__(self: Tensor[Float32], rhs: Int)             -> Tensor[Float32]: return sub_t(self, to_float32(scalar_int(rhs)))     # a - s(int→f32)

    # * (result: Float64 if rhs has Float64, else Float32)
    fn __mul__(self: Tensor[Float32], rhs: Tensor[Float64]) -> Tensor[Float64]: return mul_t(to_float64(self), rhs)                  # a(f32→f64) * b(f64)
    fn __mul__(self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Float32]: return mul_t(self, rhs)                              # a * b(f32)
    fn __mul__(self: Tensor[Float32], rhs: Tensor[Int])     -> Tensor[Float32]: return mul_t(self, to_float32(rhs))                  # a * b(int→f32)
    fn __mul__(self: Tensor[Float32], rhs: Float64)         -> Tensor[Float64]: return mul_t(to_float64(self), scalar64(rhs))       # a(f32→f64) * s(f64)
    fn __mul__(self: Tensor[Float32], rhs: Float32)         -> Tensor[Float32]: return mul_t(self, scalar32(rhs))                   # a * s(f32)
    fn __mul__(self: Tensor[Float32], rhs: Int)             -> Tensor[Float32]: return mul_t(self, to_float32(scalar_int(rhs)))     # a * s(int→f32)

    # / (result: Float64 always)
    fn __truediv__(self: Tensor[Float32], rhs: Tensor[Float64]) -> Tensor[Float64]: return div_t(to_float64(self), rhs)              # a(f32→f64) / b(f64)
    fn __truediv__(self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Float64]: return div_t(to_float64(self), to_float64(rhs))  # a(f32→f64) / b(f32→f64)
    fn __truediv__(self: Tensor[Float32], rhs: Tensor[Int])     -> Tensor[Float64]: return div_t(to_float64(self), to_float64(rhs))  # a(f32→f64) / b(int→f64)
    fn __truediv__(self: Tensor[Float32], rhs: Float64)         -> Tensor[Float64]: return div_t(to_float64(self), scalar64(rhs))   # a(f32→f64) / s(f64)
    fn __truediv__(self: Tensor[Float32], rhs: Float32)         -> Tensor[Float64]: return div_t(to_float64(self), to_float64(scalar32(rhs))) # a(f32→f64) / s(f32→f64)
    fn __truediv__(self: Tensor[Float32], rhs: Int)             -> Tensor[Float64]: return div_t(to_float64(self), to_float64(scalar_int(rhs))) # a(f32→f64) / s(int→f64)

    # % (result: Float64 always)
    fn __mod__(self: Tensor[Float32], rhs: Tensor[Float64]) -> Tensor[Float64]: return mod_t(to_float64(self), rhs)                  # a(f32→f64) % b(f64)
    fn __mod__(self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Float64]: return mod_t(to_float64(self), to_float64(rhs))      # a(f32→f64) % b(f32→f64)
    fn __mod__(self: Tensor[Float32], rhs: Tensor[Int])     -> Tensor[Float64]: return mod_t(to_float64(self), to_float64(rhs))      # a(f32→f64) % b(int→f64)
    fn __mod__(self: Tensor[Float32], rhs: Float64)         -> Tensor[Float64]: return mod_t(to_float64(self), scalar64(rhs))       # a(f32→f64) % s(f64)
    fn __mod__(self: Tensor[Float32], rhs: Float32)         -> Tensor[Float64]: return mod_t(to_float64(self), to_float64(scalar32(rhs))) # a(f32→f64) % s(f32→f64)
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
    fn __add__(self: Tensor[Int], rhs: Float64)         -> Tensor[Float64]: return add_t(to_float64(self), scalar64(rhs))         # a(int→f64) + s(f64)
    fn __add__(self: Tensor[Int], rhs: Float32)         -> Tensor[Float32]: return add_t(to_float32(self), scalar32(rhs))         # a(int→f32) + s(f32)
    fn __add__(self: Tensor[Int], rhs: Int)             -> Tensor[Int]:     return add_t(self, scalar_int(rhs))                   # a + s(int)

    # - (promotion)
    fn __sub__(self: Tensor[Int], rhs: Tensor[Float64]) -> Tensor[Float64]: return sub_t(to_float64(self), rhs)                     # a(int→f64) - b(f64)
    fn __sub__(self: Tensor[Int], rhs: Tensor[Float32]) -> Tensor[Float32]: return sub_t(to_float32(self), rhs)                    # a(int→f32) - b(f32)
    fn __sub__(self: Tensor[Int], rhs: Tensor[Int])     -> Tensor[Int]:     return sub_t(self, rhs)                                # a - b(int)
    fn __sub__(self: Tensor[Int], rhs: Float64)         -> Tensor[Float64]: return sub_t(to_float64(self), scalar64(rhs))         # a(int→f64) - s(f64)
    fn __sub__(self: Tensor[Int], rhs: Float32)         -> Tensor[Float32]: return sub_t(to_float32(self), scalar32(rhs))         # a(int→f32) - s(f32)
    fn __sub__(self: Tensor[Int], rhs: Int)             -> Tensor[Int]:     return sub_t(self, scalar_int(rhs))                   # a - s(int)

    # * (promotion)
    fn __mul__(self: Tensor[Int], rhs: Tensor[Float64]) -> Tensor[Float64]: return mul_t(to_float64(self), rhs)                     # a(int→f64) * b(f64)
    fn __mul__(self: Tensor[Int], rhs: Tensor[Float32]) -> Tensor[Float32]: return mul_t(to_float32(self), rhs)                    # a(int→f32) * b(f32)
    fn __mul__(self: Tensor[Int], rhs: Tensor[Int])     -> Tensor[Int]:     return mul_t(self, rhs)                                # a * b(int)
    fn __mul__(self: Tensor[Int], rhs: Float64)         -> Tensor[Float64]: return mul_t(to_float64(self), scalar64(rhs))         # a(int→f64) * s(f64)
    fn __mul__(self: Tensor[Int], rhs: Float32)         -> Tensor[Float32]: return mul_t(to_float32(self), scalar32(rhs))         # a(int→f32) * s(f32)
    fn __mul__(self: Tensor[Int], rhs: Int)             -> Tensor[Int]:     return mul_t(self, scalar_int(rhs))                   # a * s(int)

    # / (Float64 result)
    fn __truediv__(self: Tensor[Int], rhs: Tensor[Float64]) -> Tensor[Float64]: return div_t(to_float64(self), rhs)                # a(int→f64) / b(f64)
    fn __truediv__(self: Tensor[Int], rhs: Tensor[Float32]) -> Tensor[Float64]: return div_t(to_float64(self), to_float64(rhs))    # a(int→f64) / b(f32→f64)
    fn __truediv__(self: Tensor[Int], rhs: Tensor[Int])     -> Tensor[Float64]: return div_t(to_float64(self), to_float64(rhs))    # a(int→f64) / b(int→f64)
    fn __truediv__(self: Tensor[Int], rhs: Float64)         -> Tensor[Float64]: return div_t(to_float64(self), scalar64(rhs))     # a(int→f64) / s(f64)
    fn __truediv__(self: Tensor[Int], rhs: Float32)         -> Tensor[Float64]: return div_t(to_float64(self), to_float64(scalar32(rhs))) # a(int→f64) / s(f32→f64)
    fn __truediv__(self: Tensor[Int], rhs: Int)             -> Tensor[Float64]: return div_t(to_float64(self), to_float64(scalar_int(rhs))) # a(int→f64) / s(int→f64)

    # % (Float64 result)
    fn __mod__(self: Tensor[Int], rhs: Tensor[Float64]) -> Tensor[Float64]: return mod_t(to_float64(self), rhs)                    # a(int→f64) % b(f64)
    fn __mod__(self: Tensor[Int], rhs: Tensor[Float32]) -> Tensor[Float64]: return mod_t(to_float64(self), to_float64(rhs))        # a(int→f64) % b(f32→f64)
    fn __mod__(self: Tensor[Int], rhs: Tensor[Int])     -> Tensor[Float64]: return mod_t(to_float64(self), to_float64(rhs))        # a(int→f64) % b(int→f64)
    fn __mod__(self: Tensor[Int], rhs: Float64)         -> Tensor[Float64]: return mod_t(to_float64(self), scalar64(rhs))         # a(int→f64) % s(f64)
    fn __mod__(self: Tensor[Int], rhs: Float32)         -> Tensor[Float64]: return mod_t(to_float64(self), to_float64(scalar32(rhs))) # a(int→f64) % s(f32→f64)
    fn __mod__(self: Tensor[Int], rhs: Int)             -> Tensor[Float64]: return mod_t(to_float64(self), to_float64(scalar_int(rhs))) # a(int→f64) % s(int→f64)
    # =========================
    # Float64 — Reflected arithmetic (lhs ⊕ self)
    # =========================
    fn __radd__(self: Tensor[Float64], lhs: Float64) -> Tensor[Float64]: return add_t(scalar64(lhs), self)                         # s(f64) + a
    fn __radd__(self: Tensor[Float64], lhs: Float32) -> Tensor[Float64]: return add_t(to_float64(scalar32(lhs)), self)            # s(f32→f64) + a
    fn __radd__(self: Tensor[Float64], lhs: Int)     -> Tensor[Float64]: return add_t(to_float64(scalar_int(lhs)), self)          # s(int→f64) + a

    fn __rsub__(self: Tensor[Float64], lhs: Float64) -> Tensor[Float64]: return sub_t(scalar64(lhs), self)                         # s(f64) - a
    fn __rsub__(self: Tensor[Float64], lhs: Float32) -> Tensor[Float64]: return sub_t(to_float64(scalar32(lhs)), self)            # s(f32→f64) - a
    fn __rsub__(self: Tensor[Float64], lhs: Int)     -> Tensor[Float64]: return sub_t(to_float64(scalar_int(lhs)), self)          # s(int→f64) - a

    fn __rmul__(self: Tensor[Float64], lhs: Float64) -> Tensor[Float64]: return mul_t(scalar64(lhs), self)                         # s(f64) * a
    fn __rmul__(self: Tensor[Float64], lhs: Float32) -> Tensor[Float64]: return mul_t(to_float64(scalar32(lhs)), self)            # s(f32→f64) * a
    fn __rmul__(self: Tensor[Float64], lhs: Int)     -> Tensor[Float64]: return mul_t(to_float64(scalar_int(lhs)), self)          # s(int→f64) * a

    fn __rtruediv__(self: Tensor[Float64], lhs: Float64) -> Tensor[Float64]: return div_t(scalar64(lhs), self)                    # s(f64) / a
    fn __rtruediv__(self: Tensor[Float64], lhs: Float32) -> Tensor[Float64]: return div_t(to_float64(scalar32(lhs)), self)       # s(f32→f64) / a
    fn __rtruediv__(self: Tensor[Float64], lhs: Int)     -> Tensor[Float64]: return div_t(to_float64(scalar_int(lhs)), self)     # s(int→f64) / a

    fn __rmod__(self: Tensor[Float64], lhs: Float64) -> Tensor[Float64]: return mod_t(scalar64(lhs), self)                        # s(f64) % a
    fn __rmod__(self: Tensor[Float64], lhs: Float32) -> Tensor[Float64]: return mod_t(to_float64(scalar32(lhs)), self)           # s(f32→f64) % a
    fn __rmod__(self: Tensor[Float64], lhs: Int)     -> Tensor[Float64]: return mod_t(to_float64(scalar_int(lhs)), self)         # s(int→f64) % a


    # =========================
    # Float32 — Reflected arithmetic (lhs ⊕ self)
    # Note: for / and % we return Float64 (project convention).
    # =========================
    fn __radd__(self: Tensor[Float32], lhs: Float64) -> Tensor[Float64]: return add_t(scalar64(lhs), to_float64(self))            # s(f64) + a(f32→f64)
    fn __radd__(self: Tensor[Float32], lhs: Float32) -> Tensor[Float32]: return add_t(scalar32(lhs), self)                        # s(f32) + a
    fn __radd__(self: Tensor[Float32], lhs: Int)     -> Tensor[Float32]: return add_t(to_float32(scalar_int(lhs)), self)          # s(int→f32) + a

    fn __rsub__(self: Tensor[Float32], lhs: Float64) -> Tensor[Float64]: return sub_t(scalar64(lhs), to_float64(self))            # s(f64) - a(f32→f64)
    fn __rsub__(self: Tensor[Float32], lhs: Float32) -> Tensor[Float32]: return sub_t(scalar32(lhs), self)                        # s(f32) - a
    fn __rsub__(self: Tensor[Float32], lhs: Int)     -> Tensor[Float32]: return sub_t(to_float32(scalar_int(lhs)), self)          # s(int→f32) - a

    fn __rmul__(self: Tensor[Float32], lhs: Float64) -> Tensor[Float64]: return mul_t(scalar64(lhs), to_float64(self))            # s(f64) * a(f32→f64)
    fn __rmul__(self: Tensor[Float32], lhs: Float32) -> Tensor[Float32]: return mul_t(scalar32(lhs), self)                        # s(f32) * a
    fn __rmul__(self: Tensor[Float32], lhs: Int)     -> Tensor[Float32]: return mul_t(to_float32(scalar_int(lhs)), self)          # s(int→f32) * a

    fn __rtruediv__(self: Tensor[Float32], lhs: Float64) -> Tensor[Float64]: return div_t(scalar64(lhs), to_float64(self))        # s(f64) / a(f32→f64)
    fn __rtruediv__(self: Tensor[Float32], lhs: Float32) -> Tensor[Float64]: return div_t(to_float64(scalar32(lhs)), to_float64(self)) # s(f32→f64) / a(f32→f64)
    fn __rtruediv__(self: Tensor[Float32], lhs: Int)     -> Tensor[Float64]: return div_t(to_float64(scalar_int(lhs)), to_float64(self)) # s(int→f64) / a(f32→f64)

    fn __rmod__(self: Tensor[Float32], lhs: Float64) -> Tensor[Float64]: return mod_t(scalar64(lhs), to_float64(self))            # s(f64) % a(f32→f64)
    fn __rmod__(self: Tensor[Float32], lhs: Float32) -> Tensor[Float64]: return mod_t(to_float64(scalar32(lhs)), to_float64(self))# s(f32→f64) % a(f32→f64)
    fn __rmod__(self: Tensor[Float32], lhs: Int)     -> Tensor[Float64]: return mod_t(to_float64(scalar_int(lhs)), to_float64(self)) # s(int→f64) % a(f32→f64)


    # =========================
    # Int — Reflected arithmetic (lhs ⊕ self)
    # Note: for / and % we return Float64 (project convention).
    # =========================
    fn __radd__(self: Tensor[Int], lhs: Float64) -> Tensor[Float64]: return add_t(scalar64(lhs), to_float64(self))                # s(f64) + a(int→f64)
    fn __radd__(self: Tensor[Int], lhs: Float32) -> Tensor[Float32]: return add_t(scalar32(lhs), to_float32(self))                # s(f32) + a(int→f32)
    fn __radd__(self: Tensor[Int], lhs: Int)     -> Tensor[Int]:     return add_t(scalar_int(lhs), self)                          # s(int) + a

    fn __rsub__(self: Tensor[Int], lhs: Float64) -> Tensor[Float64]: return sub_t(scalar64(lhs), to_float64(self))                # s(f64) - a(int→f64)
    fn __rsub__(self: Tensor[Int], lhs: Float32) -> Tensor[Float32]: return sub_t(scalar32(lhs), to_float32(self))                # s(f32) - a(int→f32)
    fn __rsub__(self: Tensor[Int], lhs: Int)     -> Tensor[Int]:     return sub_t(scalar_int(lhs), self)                          # s(int) - a

    fn __rmul__(self: Tensor[Int], lhs: Float64) -> Tensor[Float64]: return mul_t(scalar64(lhs), to_float64(self))                # s(f64) * a(int→f64)
    fn __rmul__(self: Tensor[Int], lhs: Float32) -> Tensor[Float32]: return mul_t(scalar32(lhs), to_float32(self))                # s(f32) * a(int→f32)
    fn __rmul__(self: Tensor[Int], lhs: Int)     -> Tensor[Int]:     return mul_t(scalar_int(lhs), self)                          # s(int) * a

    fn __rtruediv__(self: Tensor[Int], lhs: Float64) -> Tensor[Float64]: return div_t(scalar64(lhs), to_float64(self))            # s(f64) / a(int→f64)
    fn __rtruediv__(self: Tensor[Int], lhs: Float32) -> Tensor[Float64]: return div_t(to_float64(scalar32(lhs)), to_float64(self))# s(f32→f64) / a(int→f64)
    fn __rtruediv__(self: Tensor[Int], lhs: Int)     -> Tensor[Float64]: return div_t(to_float64(scalar_int(lhs)), to_float64(self)) # s(int→f64) / a(int→f64)

    fn __rmod__(self: Tensor[Int], lhs: Float64) -> Tensor[Float64]: return mod_t(scalar64(lhs), to_float64(self))                # s(f64) % a(int→f64)
    fn __rmod__(self: Tensor[Int], lhs: Float32) -> Tensor[Float64]: return mod_t(to_float64(scalar32(lhs)), to_float64(self))    # s(f32→f64) % a(int→f64)
    fn __rmod__(self: Tensor[Int], lhs: Int)     -> Tensor[Float64]: return mod_t(to_float64(scalar_int(lhs)), to_float64(self))  # s(int→f64) % a(int→f64)
    # =========================
    # In-place arithmetic — Float64 (full scalar combos)
    # Promotion policy for in-place: keep self dtype (convert rhs up to Float64)
    # =========================
    fn __iadd__(mut self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Float64]: self = add_t(self, rhs); return self                           # a += b
    fn __iadd__(mut self: Tensor[Float64], rhs: Float64)        -> Tensor[Float64]: self = add_t(self, scalar64(rhs)); return self                  # a += s(f64)
    fn __iadd__(mut self: Tensor[Float64], rhs: Float32)        -> Tensor[Float64]: self = add_t(self, to_float64(scalar32(rhs))); return self      # a += s(f32→f64)
    fn __iadd__(mut self: Tensor[Float64], rhs: Int)            -> Tensor[Float64]: self = add_t(self, to_float64(scalar_int(rhs))); return self    # a += s(int→f64)

    fn __isub__(mut self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Float64]: self = sub_t(self, rhs); return self                           # a -= b
    fn __isub__(mut self: Tensor[Float64], rhs: Float64)        -> Tensor[Float64]: self = sub_t(self, scalar64(rhs)); return self                  # a -= s(f64)
    fn __isub__(mut self: Tensor[Float64], rhs: Float32)        -> Tensor[Float64]: self = sub_t(self, to_float64(scalar32(rhs))); return self      # a -= s(f32→f64)
    fn __isub__(mut self: Tensor[Float64], rhs: Int)            -> Tensor[Float64]: self = sub_t(self, to_float64(scalar_int(rhs))); return self    # a -= s(int→f64)

    fn __imul__(mut self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Float64]: self = mul_t(self, rhs); return self                           # a *= b
    fn __imul__(mut self: Tensor[Float64], rhs: Float64)        -> Tensor[Float64]: self = mul_t(self, scalar64(rhs)); return self                  # a *= s(f64)
    fn __imul__(mut self: Tensor[Float64], rhs: Float32)        -> Tensor[Float64]: self = mul_t(self, to_float64(scalar32(rhs))); return self      # a *= s(f32→f64)
    fn __imul__(mut self: Tensor[Float64], rhs: Int)            -> Tensor[Float64]: self = mul_t(self, to_float64(scalar_int(rhs))); return self    # a *= s(int→f64)

    fn __itruediv__(mut self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Float64]: self = div_t(self, rhs); return self                       # a /= b
    fn __itruediv__(mut self: Tensor[Float64], rhs: Float64)        -> Tensor[Float64]: self = div_t(self, scalar64(rhs)); return self              # a /= s(f64)
    fn __itruediv__(mut self: Tensor[Float64], rhs: Float32)        -> Tensor[Float64]: self = div_t(self, to_float64(scalar32(rhs))); return self  # a /= s(f32→f64)
    fn __itruediv__(mut self: Tensor[Float64], rhs: Int)            -> Tensor[Float64]: self = div_t(self, to_float64(scalar_int(rhs))); return self# a /= s(int→f64)

    fn __imod__(mut self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Float64]: self = mod_t(self, rhs); return self                           # a %= b
    fn __imod__(mut self: Tensor[Float64], rhs: Float64)        -> Tensor[Float64]: self = mod_t(self, scalar64(rhs)); return self                  # a %= s(f64)
    fn __imod__(mut self: Tensor[Float64], rhs: Float32)        -> Tensor[Float64]: self = mod_t(self, to_float64(scalar32(rhs))); return self      # a %= s(f32→f64)
    fn __imod__(mut self: Tensor[Float64], rhs: Int)            -> Tensor[Float64]: self = mod_t(self, to_float64(scalar_int(rhs))); return self    # a %= s(int→f64)


    # =========================
    # In-place arithmetic — Float32 (full scalar combos)
    # Promotion policy for in-place: keep self dtype (convert rhs up to Float32)
    # =========================
    fn __iadd__(mut self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Float32]: self = add_t(self, rhs); return self                           # a += b
    fn __iadd__(mut self: Tensor[Float32], rhs: Float32)        -> Tensor[Float32]: self = add_t(self, scalar32(rhs)); return self                  # a += s(f32)
    fn __iadd__(mut self: Tensor[Float32], rhs: Float64)        -> Tensor[Float32]: self = add_t(self, to_float32(scalar64(rhs))); return self      # a += s(f64→f32)
    fn __iadd__(mut self: Tensor[Float32], rhs: Int)            -> Tensor[Float32]: self = add_t(self, to_float32(scalar_int(rhs))); return self    # a += s(int→f32)

    fn __isub__(mut self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Float32]: self = sub_t(self, rhs); return self                           # a -= b
    fn __isub__(mut self: Tensor[Float32], rhs: Float32)        -> Tensor[Float32]: self = sub_t(self, scalar32(rhs)); return self                  # a -= s(f32)
    fn __isub__(mut self: Tensor[Float32], rhs: Float64)        -> Tensor[Float32]: self = sub_t(self, to_float32(scalar64(rhs))); return self      # a -= s(f64→f32)
    fn __isub__(mut self: Tensor[Float32], rhs: Int)            -> Tensor[Float32]: self = sub_t(self, to_float32(scalar_int(rhs))); return self    # a -= s(int→f32)

    fn __imul__(mut self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Float32]: self = mul_t(self, rhs); return self                           # a *= b
    fn __imul__(mut self: Tensor[Float32], rhs: Float32)        -> Tensor[Float32]: self = mul_t(self, scalar32(rhs)); return self                  # a *= s(f32)
    fn __imul__(mut self: Tensor[Float32], rhs: Float64)        -> Tensor[Float32]: self = mul_t(self, to_float32(scalar64(rhs))); return self      # a *= s(f64→f32)
    fn __imul__(mut self: Tensor[Float32], rhs: Int)            -> Tensor[Float32]: self = mul_t(self, to_float32(scalar_int(rhs))); return self    # a *= s(int→f32)

    fn __itruediv__(mut self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Float32]: self = div_t(self, rhs); return self                       # a /= b
    fn __itruediv__(mut self: Tensor[Float32], rhs: Float32)        -> Tensor[Float32]: self = div_t(self, scalar32(rhs)); return self              # a /= s(f32)
    fn __itruediv__(mut self: Tensor[Float32], rhs: Float64)        -> Tensor[Float32]: self = div_t(self, to_float32(scalar64(rhs))); return self  # a /= s(f64→f32)
    fn __itruediv__(mut self: Tensor[Float32], rhs: Int)            -> Tensor[Float32]: self = div_t(self, to_float32(scalar_int(rhs))); return self# a /= s(int→f32)

    fn __imod__(mut self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Float32]: self = mod_t(self, rhs); return self                           # a %= b
    fn __imod__(mut self: Tensor[Float32], rhs: Float32)        -> Tensor[Float32]: self = mod_t(self, scalar32(rhs)); return self                  # a %= s(f32)
    fn __imod__(mut self: Tensor[Float32], rhs: Float64)        -> Tensor[Float32]: self = mod_t(self, to_float32(scalar64(rhs))); return self      # a %= s(f64→f32)
    fn __imod__(mut self: Tensor[Float32], rhs: Int)            -> Tensor[Float32]: self = mod_t(self, to_float32(scalar_int(rhs))); return self    # a %= s(int→f32)


    # =========================
    # In-place arithmetic — Int (full scalar combos)
    # Promotion policy for in-place: keep self dtype (convert rhs up to Int)
    # NOTE: true-division/mod on Int keeps Int here (uses div_t/mod_t for Int domain)
    # =========================
    fn __iadd__(mut self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]: self = add_t(self, rhs); return self                                 # a += b
    fn __iadd__(mut self: Tensor[Int], rhs: Int)        -> Tensor[Int]: self = add_t(self, scalar_int(rhs)); return self                     # a += s(int)
    fn __iadd__(mut self: Tensor[Int], rhs: Float32)    -> Tensor[Int]: self = add_t(self, to_int(scalar32(rhs))); return self               # a += s(f32→int)
    fn __iadd__(mut self: Tensor[Int], rhs: Float64)    -> Tensor[Int]: self = add_t(self, to_int(scalar64(rhs))); return self               # a += s(f64→int)

    fn __isub__(mut self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]: self = sub_t(self, rhs); return self                                 # a -= b
    fn __isub__(mut self: Tensor[Int], rhs: Int)        -> Tensor[Int]: self = sub_t(self, scalar_int(rhs)); return self                     # a -= s(int)
    fn __isub__(mut self: Tensor[Int], rhs: Float32)    -> Tensor[Int]: self = sub_t(self, to_int(scalar32(rhs))); return self               # a -= s(f32→int)
    fn __isub__(mut self: Tensor[Int], rhs: Float64)    -> Tensor[Int]: self = sub_t(self, to_int(scalar64(rhs))); return self               # a -= s(f64→int)

    fn __imul__(mut self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]: self = mul_t(self, rhs); return self                                 # a *= b
    fn __imul__(mut self: Tensor[Int], rhs: Int)        -> Tensor[Int]: self = mul_t(self, scalar_int(rhs)); return self                     # a *= s(int)
    fn __imul__(mut self: Tensor[Int], rhs: Float32)    -> Tensor[Int]: self = mul_t(self, to_int(scalar32(rhs))); return self               # a *= s(f32→int)
    fn __imul__(mut self: Tensor[Int], rhs: Float64)    -> Tensor[Int]: self = mul_t(self, to_int(scalar64(rhs))); return self               # a *= s(f64→int)

    fn __itruediv__(mut self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]: self = div_t(self, rhs); return self                             # a /= b (int domain)
    fn __itruediv__(mut self: Tensor[Int], rhs: Int)        -> Tensor[Int]: self = div_t(self, scalar_int(rhs)); return self                 # a /= s(int)
    fn __itruediv__(mut self: Tensor[Int], rhs: Float32)    -> Tensor[Int]: self = div_t(self, to_int(scalar32(rhs))); return self           # a /= s(f32→int)
    fn __itruediv__(mut self: Tensor[Int], rhs: Float64)    -> Tensor[Int]: self = div_t(self, to_int(scalar64(rhs))); return self           # a /= s(f64→int)

    fn __imod__(mut self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]: self = mod_t(self, rhs); return self                                 # a %= b (int domain)
    fn __imod__(mut self: Tensor[Int], rhs: Int)        -> Tensor[Int]: self = mod_t(self, scalar_int(rhs)); return self                     # a %= s(int)
    fn __imod__(mut self: Tensor[Int], rhs: Float32)    -> Tensor[Int]: self = mod_t(self, to_int(scalar32(rhs))); return self               # a %= s(f32→int)
    fn __imod__(mut self: Tensor[Int], rhs: Float64)    -> Tensor[Int]: self = mod_t(self, to_int(scalar64(rhs))); return self               # a %= s(f64→int)
    # =========================
    # Power overloads — full combos, promotion Int < Float32 < Float64
    # Uses only: pow_t, to_int, to_float32, to_float64, scalar_int/scalar32/scalar64
    # =========================

    # -------------------------
    # Tensor[Float64]
    # -------------------------
    fn __pow__(self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Float64]: return pow_t(self, rhs)                                         # a(f64) ** b(f64)
    fn __pow__(self: Tensor[Float64], rhs: Float64)        -> Tensor[Float64]: return pow_t(self, scalar64(rhs))                              # a(f64) ** s(f64)
    fn __pow__(self: Tensor[Float64], rhs: Float32)        -> Tensor[Float64]: return pow_t(self, to_float64(scalar32(rhs)))                  # a(f64) ** s(f32→f64)
    fn __pow__(self: Tensor[Float64], rhs: Int)            -> Tensor[Float64]: return pow_t(self, to_float64(scalar_int(rhs)))                # a(f64) ** s(int→f64)

    fn __rpow__(self: Tensor[Float64], lhs: Float64)       -> Tensor[Float64]: return pow_t(scalar64(lhs), self)                              # s(f64) ** a(f64)
    fn __rpow__(self: Tensor[Float64], lhs: Float32)       -> Tensor[Float64]: return pow_t(to_float64(scalar32(lhs)), self)                  # s(f32→f64) ** a(f64)
    fn __rpow__(self: Tensor[Float64], lhs: Int)           -> Tensor[Float64]: return pow_t(to_float64(scalar_int(lhs)), self)                # s(int→f64) ** a(f64)

    fn __ipow__(mut self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Float64]: self = pow_t(self, rhs); return self                      # a **= b(f64)
    fn __ipow__(mut self: Tensor[Float64], rhs: Float64)        -> Tensor[Float64]: self = pow_t(self, scalar64(rhs)); return self            # a **= s(f64)
    fn __ipow__(mut self: Tensor[Float64], rhs: Float32)        -> Tensor[Float64]: self = pow_t(self, to_float64(scalar32(rhs))); return self# a **= s(f32→f64)
    fn __ipow__(mut self: Tensor[Float64], rhs: Int)            -> Tensor[Float64]: self = pow_t(self, to_float64(scalar_int(rhs))); return self# a **= s(int→f64)

    # -------------------------
    # Tensor[Float32]
    # -------------------------
    # result Float64 if rhs/lhs is Float64 or Tensor[Float64]
    fn __pow__(self: Tensor[Float32], rhs: Tensor[Float64]) -> Tensor[Float64]: return pow_t(to_float64(self), rhs)                            # a(f32→f64) ** b(f64)
    fn __pow__(self: Tensor[Float32], rhs: Float64)        -> Tensor[Float64]: return pow_t(to_float64(self), scalar64(rhs))                 # a(f32→f64) ** s(f64)
    fn __rpow__(self: Tensor[Float32], lhs: Float64)       -> Tensor[Float64]: return pow_t(scalar64(lhs), to_float64(self))                 # s(f64) ** a(f32→f64)

    # result Float32 otherwise
    fn __pow__(self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Float32]: return pow_t(self, rhs)                                       # a(f32) ** b(f32)
    fn __pow__(self: Tensor[Float32], rhs: Float32)        -> Tensor[Float32]: return pow_t(self, scalar32(rhs))                             # a(f32) ** s(f32)
    fn __pow__(self: Tensor[Float32], rhs: Int)            -> Tensor[Float32]: return pow_t(self, to_float32(scalar_int(rhs)))               # a(f32) ** s(int→f32)

    fn __rpow__(self: Tensor[Float32], lhs: Float32)       -> Tensor[Float32]: return pow_t(scalar32(lhs), self)                             # s(f32) ** a(f32)
    fn __rpow__(self: Tensor[Float32], lhs: Int)           -> Tensor[Float32]: return pow_t(to_float32(scalar_int(lhs)), self)               # s(int→f32) ** a(f32)

    # in-place only when result dtype remains Float32
    fn __ipow__(mut self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Float32]: self = pow_t(self, rhs); return self                     # a **= b(f32)
    fn __ipow__(mut self: Tensor[Float32], rhs: Float32)        -> Tensor[Float32]: self = pow_t(self, scalar32(rhs)); return self           # a **= s(f32)
    fn __ipow__(mut self: Tensor[Float32], rhs: Int)            -> Tensor[Float32]: self = pow_t(self, to_float32(scalar_int(rhs))); return self # a **= s(int→f32)

    # -------------------------
    # Tensor[Int]
    # -------------------------
    # result Float64 if rhs/lhs is Float64 or Tensor[Float64]
    fn __pow__(self: Tensor[Int], rhs: Tensor[Float64]) -> Tensor[Float64]: return pow_t(to_float64(self), rhs)                                # a(int→f64) ** b(f64)
    fn __pow__(self: Tensor[Int], rhs: Float64)        -> Tensor[Float64]: return pow_t(to_float64(self), scalar64(rhs))                     # a(int→f64) ** s(f64)
    fn __rpow__(self: Tensor[Int], lhs: Float64)       -> Tensor[Float64]: return pow_t(scalar64(lhs), to_float64(self))                     # s(f64) ** a(int→f64)

    # result Float32 if no Float64 present but Float32/ Tensor[Float32] present
    fn __pow__(self: Tensor[Int], rhs: Tensor[Float32]) -> Tensor[Float32]: return pow_t(to_float32(self), rhs)                               # a(int→f32) ** b(f32)
    fn __pow__(self: Tensor[Int], rhs: Float32)        -> Tensor[Float32]: return pow_t(to_float32(self), scalar32(rhs))                    # a(int→f32) ** s(f32)
    fn __rpow__(self: Tensor[Int], lhs: Float32)       -> Tensor[Float32]: return pow_t(scalar32(lhs), to_float32(self))                    # s(f32) ** a(int→f32)

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
    fn __and__(self: Tensor[Float64], rhs: Float64)         -> Tensor[Int]: return and_t(self, scalar64(rhs))                     # a & s (f64)
    fn __and__(self: Tensor[Float64], rhs: Float32)         -> Tensor[Int]: return and_t(self, to_float64(scalar32(rhs)))         # a & s (f32→f64)
    fn __and__(self: Tensor[Float64], rhs: Int)             -> Tensor[Int]: return and_t(self, to_float64(scalar_int(rhs)))       # a & s (int→f64)

    fn __or__( self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Int]: return or_t (self, rhs)                                # a | b (f64,f64)
    fn __or__( self: Tensor[Float64], rhs: Tensor[Float32]) -> Tensor[Int]: return or_t (self, to_float64(rhs))                    # a | b (f64,f32→f64)
    fn __or__( self: Tensor[Float64], rhs: Tensor[Int])     -> Tensor[Int]: return or_t (self, to_float64(rhs))                    # a | b (f64,int→f64)
    fn __or__( self: Tensor[Float64], rhs: Float64)         -> Tensor[Int]: return or_t (self, scalar64(rhs))                     # a | s (f64)
    fn __or__( self: Tensor[Float64], rhs: Float32)         -> Tensor[Int]: return or_t (self, to_float64(scalar32(rhs)))         # a | s (f32→f64)
    fn __or__( self: Tensor[Float64], rhs: Int)             -> Tensor[Int]: return or_t (self, to_float64(scalar_int(rhs)))       # a | s (int→f64)

    fn __xor__(self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Int]: return xor_t(self, rhs)                                # a ^ b (f64,f64)
    fn __xor__(self: Tensor[Float64], rhs: Tensor[Float32]) -> Tensor[Int]: return xor_t(self, to_float64(rhs))                    # a ^ b (f64,f32→f64)
    fn __xor__(self: Tensor[Float64], rhs: Tensor[Int])     -> Tensor[Int]: return xor_t(self, to_float64(rhs))                    # a ^ b (f64,int→f64)
    fn __xor__(self: Tensor[Float64], rhs: Float64)         -> Tensor[Int]: return xor_t(self, scalar64(rhs))                     # a ^ s (f64)
    fn __xor__(self: Tensor[Float64], rhs: Float32)         -> Tensor[Int]: return xor_t(self, to_float64(scalar32(rhs)))         # a ^ s (f32→f64)
    fn __xor__(self: Tensor[Float64], rhs: Int)             -> Tensor[Int]: return xor_t(self, to_float64(scalar_int(rhs)))       # a ^ s (int→f64)
                                                 # ~a


    # =========================
    # Float32 logical overloads (elementwise → Tensor[Int] mask)
    # =========================
    fn __and__(self: Tensor[Float32], rhs: Tensor[Float64]) -> Tensor[Int]: return and_t(to_float64(self), rhs)                    # a(f32→f64) & b(f64)
    fn __and__(self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Int]: return and_t(self, rhs)                                # a & b (f32,f32)
    fn __and__(self: Tensor[Float32], rhs: Tensor[Int])     -> Tensor[Int]: return and_t(self, to_float32(rhs))                    # a & b (f32,int→f32)
    fn __and__(self: Tensor[Float32], rhs: Float64)         -> Tensor[Int]: return and_t(to_float64(self), scalar64(rhs))         # a(f32→f64) & s(f64)
    fn __and__(self: Tensor[Float32], rhs: Float32)         -> Tensor[Int]: return and_t(self, scalar32(rhs))                     # a & s(f32)
    fn __and__(self: Tensor[Float32], rhs: Int)             -> Tensor[Int]: return and_t(self, to_float32(scalar_int(rhs)))       # a & s(int→f32)

    fn __or__( self: Tensor[Float32], rhs: Tensor[Float64]) -> Tensor[Int]: return or_t (to_float64(self), rhs)                    # a(f32→f64) | b(f64)
    fn __or__( self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Int]: return or_t (self, rhs)                                # a | b (f32,f32)
    fn __or__( self: Tensor[Float32], rhs: Tensor[Int])     -> Tensor[Int]: return or_t (self, to_float32(rhs))                    # a | b (f32,int→f32)
    fn __or__( self: Tensor[Float32], rhs: Float64)         -> Tensor[Int]: return or_t (to_float64(self), scalar64(rhs))         # a(f32→f64) | s(f64)
    fn __or__( self: Tensor[Float32], rhs: Float32)         -> Tensor[Int]: return or_t (self, scalar32(rhs))                     # a | s(f32)
    fn __or__( self: Tensor[Float32], rhs: Int)             -> Tensor[Int]: return or_t (self, to_float32(scalar_int(rhs)))       # a | s(int→f32)

    fn __xor__(self: Tensor[Float32], rhs: Tensor[Float64]) -> Tensor[Int]: return xor_t(to_float64(self), rhs)                    # a(f32→f64) ^ b(f64)
    fn __xor__(self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Int]: return xor_t(self, rhs)                                # a ^ b (f32,f32)
    fn __xor__(self: Tensor[Float32], rhs: Tensor[Int])     -> Tensor[Int]: return xor_t(self, to_float32(rhs))                    # a ^ b (f32,int→f32)
    fn __xor__(self: Tensor[Float32], rhs: Float64)         -> Tensor[Int]: return xor_t(to_float64(self), scalar64(rhs))         # a(f32→f64) ^ s(f64)
    fn __xor__(self: Tensor[Float32], rhs: Float32)         -> Tensor[Int]: return xor_t(self, scalar32(rhs))                     # a ^ s(f32)
    fn __xor__(self: Tensor[Float32], rhs: Int)             -> Tensor[Int]: return xor_t(self, to_float32(scalar_int(rhs)))       # a ^ s(int→f32)
 

    # =========================
    # Int logical overloads (elementwise → Tensor[Int] mask)
    # =========================
    fn __and__(self: Tensor[Int], rhs: Tensor[Float64]) -> Tensor[Int]: return and_t(to_float64(self), rhs)                        # a(int→f64) & b(f64)
    fn __and__(self: Tensor[Int], rhs: Tensor[Float32]) -> Tensor[Int]: return and_t(to_float32(self), rhs)                        # a(int→f32) & b(f32)
    fn __and__(self: Tensor[Int], rhs: Tensor[Int])     -> Tensor[Int]: return and_t(self, rhs)                                    # a & b (int,int)
    fn __and__(self: Tensor[Int], rhs: Float64)         -> Tensor[Int]: return and_t(to_float64(self), scalar64(rhs))             # a(int→f64) & s(f64)
    fn __and__(self: Tensor[Int], rhs: Float32)         -> Tensor[Int]: return and_t(to_float32(self), scalar32(rhs))             # a(int→f32) & s(f32)
    fn __and__(self: Tensor[Int], rhs: Int)             -> Tensor[Int]: return and_t(self, scalar_int(rhs))                       # a & s(int)

    fn __or__( self: Tensor[Int], rhs: Tensor[Float64]) -> Tensor[Int]: return or_t (to_float64(self), rhs)                        # a(int→f64) | b(f64)
    fn __or__( self: Tensor[Int], rhs: Tensor[Float32]) -> Tensor[Int]: return or_t (to_float32(self), rhs)                        # a(int→f32) | b(f32)
    fn __or__( self: Tensor[Int], rhs: Tensor[Int])     -> Tensor[Int]: return or_t (self, rhs)                                    # a | b (int,int)
    fn __or__( self: Tensor[Int], rhs: Float64)         -> Tensor[Int]: return or_t (to_float64(self), scalar64(rhs))             # a(int→f64) | s(f64)
    fn __or__( self: Tensor[Int], rhs: Float32)         -> Tensor[Int]: return or_t (to_float32(self), scalar32(rhs))             # a(int→f32) | s(f32)
    fn __or__( self: Tensor[Int], rhs: Int)             -> Tensor[Int]: return or_t (self, scalar_int(rhs))                       # a | s(int)

    fn __xor__(self: Tensor[Int], rhs: Tensor[Float64]) -> Tensor[Int]: return xor_t(to_float64(self), rhs)                        # a(int→f64) ^ b(f64)
    fn __xor__(self: Tensor[Int], rhs: Tensor[Float32]) -> Tensor[Int]: return xor_t(to_float32(self), rhs)                        # a(int→f32) ^ b(f32)
    fn __xor__(self: Tensor[Int], rhs: Tensor[Int])     -> Tensor[Int]: return xor_t(self, rhs)                                    # a ^ b (int,int)
    fn __xor__(self: Tensor[Int], rhs: Float64)         -> Tensor[Int]: return xor_t(to_float64(self), scalar64(rhs))             # a(int→f64) ^ s(f64)
    fn __xor__(self: Tensor[Int], rhs: Float32)         -> Tensor[Int]: return xor_t(to_float32(self), scalar32(rhs))             # a(int→f32) ^ s(f32)
    fn __xor__(self: Tensor[Int], rhs: Int)             -> Tensor[Int]: return xor_t(self, scalar_int(rhs))                       # a ^ s(int)
 
    # =========================
    # Float64 reflected logical (result mask: Tensor[Int])
    # =========================
    fn __rand__(self: Tensor[Float64], lhs: Float64) -> Tensor[Int]: return and_t(scalar64(lhs), self)                         # s(f64) & a(f64)
    fn __rand__(self: Tensor[Float64], lhs: Float32) -> Tensor[Int]: return and_t(to_float64(scalar32(lhs)), self)            # s(f32→f64) & a(f64)
    fn __rand__(self: Tensor[Float64], lhs: Int)     -> Tensor[Int]: return and_t(to_float64(scalar_int(lhs)), self)          # s(int→f64) & a(f64)

    fn __ror__( self: Tensor[Float64], lhs: Float64) -> Tensor[Int]: return or_t (scalar64(lhs), self)                        # s(f64) | a(f64)
    fn __ror__( self: Tensor[Float64], lhs: Float32) -> Tensor[Int]: return or_t (to_float64(scalar32(lhs)), self)            # s(f32→f64) | a(f64)
    fn __ror__( self: Tensor[Float64], lhs: Int)     -> Tensor[Int]: return or_t (to_float64(scalar_int(lhs)), self)          # s(int→f64) | a(f64)

    fn __rxor__(self: Tensor[Float64], lhs: Float64) -> Tensor[Int]: return xor_t(scalar64(lhs), self)                        # s(f64) ^ a(f64)
    fn __rxor__(self: Tensor[Float64], lhs: Float32) -> Tensor[Int]: return xor_t(to_float64(scalar32(lhs)), self)            # s(f32→f64) ^ a(f64)
    fn __rxor__(self: Tensor[Float64], lhs: Int)     -> Tensor[Int]: return xor_t(to_float64(scalar_int(lhs)), self)          # s(int→f64) ^ a(f64)


    # =========================
    # Float32 reflected logical (promotion: if lhs is Float64 ⇒ promote self to Float64)
    # =========================
    fn __rand__(self: Tensor[Float32], lhs: Float64) -> Tensor[Int]: return and_t(to_float64(scalar64(lhs)), to_float64(self))# s(f64) & a(f32→f64)
    fn __rand__(self: Tensor[Float32], lhs: Float32) -> Tensor[Int]: return and_t(scalar32(lhs), self)                        # s(f32) & a(f32)
    fn __rand__(self: Tensor[Float32], lhs: Int)     -> Tensor[Int]: return and_t(to_float32(scalar_int(lhs)), self)          # s(int→f32) & a(f32)

    fn __ror__( self: Tensor[Float32], lhs: Float64) -> Tensor[Int]: return or_t (to_float64(scalar64(lhs)), to_float64(self))# s(f64) | a(f32→f64)
    fn __ror__( self: Tensor[Float32], lhs: Float32) -> Tensor[Int]: return or_t (scalar32(lhs), self)                        # s(f32) | a(f32)
    fn __ror__( self: Tensor[Float32], lhs: Int)     -> Tensor[Int]: return or_t (to_float32(scalar_int(lhs)), self)          # s(int→f32) | a(f32)

    fn __rxor__(self: Tensor[Float32], lhs: Float64) -> Tensor[Int]: return xor_t(to_float64(scalar64(lhs)), to_float64(self))# s(f64) ^ a(f32→f64)
    fn __rxor__(self: Tensor[Float32], lhs: Float32) -> Tensor[Int]: return xor_t(scalar32(lhs), self)                        # s(f32) ^ a(f32)
    fn __rxor__(self: Tensor[Float32], lhs: Int)     -> Tensor[Int]: return xor_t(to_float32(scalar_int(lhs)), self)          # s(int→f32) ^ a(f32)


    # =========================
    # Int reflected logical (promotion: lhs Float64 ⇒ f64; lhs Float32 ⇒ f32; else Int)
    # =========================
    fn __rand__(self: Tensor[Int], lhs: Float64) -> Tensor[Int]: return and_t(to_float64(scalar64(lhs)), to_float64(self))    # s(f64) & a(int→f64)
    fn __rand__(self: Tensor[Int], lhs: Float32) -> Tensor[Int]: return and_t(to_float32(scalar32(lhs)), to_float32(self))    # s(f32) & a(int→f32)
    fn __rand__(self: Tensor[Int], lhs: Int)     -> Tensor[Int]: return and_t(scalar_int(lhs), self)                          # s(int) & a(int)

    fn __ror__( self: Tensor[Int], lhs: Float64) -> Tensor[Int]: return or_t (to_float64(scalar64(lhs)), to_float64(self))    # s(f64) | a(int→f64)
    fn __ror__( self: Tensor[Int], lhs: Float32) -> Tensor[Int]: return or_t (to_float32(scalar32(lhs)), to_float32(self))    # s(f32) | a(int→f32)
    fn __ror__( self: Tensor[Int], lhs: Int)     -> Tensor[Int]: return or_t (scalar_int(lhs), self)                          # s(int) | a(int)

    fn __rxor__(self: Tensor[Int], lhs: Float64) -> Tensor[Int]: return xor_t(to_float64(scalar64(lhs)), to_float64(self))    # s(f64) ^ a(int→f64)
    fn __rxor__(self: Tensor[Int], lhs: Float32) -> Tensor[Int]: return xor_t(to_float32(scalar32(lhs)), to_float32(self))    # s(f32) ^ a(int→f32)
    fn __rxor__(self: Tensor[Int], lhs: Int)     -> Tensor[Int]: return xor_t(scalar_int(lhs), self)                          # s(int) ^ a(int)

    # =========================
    # Float64 comparisons → mask(Int)
    # Promotion: Int < Float32 < Float64
    # =========================
    fn __lt__(self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Int]: return lt_t(self, rhs)                               # a < b
    fn __lt__(self: Tensor[Float64], rhs: Float64)        -> Tensor[Int]: return lt_t(self, scalar64(rhs))                    # a < s(f64)
    fn __lt__(self: Tensor[Float64], rhs: Float32)        -> Tensor[Int]: return lt_t(self, to_float64(scalar32(rhs)))        # a < s(f32→f64)
    fn __lt__(self: Tensor[Float64], rhs: Int)            -> Tensor[Int]: return lt_t(self, to_float64(scalar_int(rhs)))      # a < s(int→f64)

    fn __le__(self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Int]: return le_t(self, rhs)                               # a <= b
    fn __le__(self: Tensor[Float64], rhs: Float64)        -> Tensor[Int]: return le_t(self, scalar64(rhs))                    # a <= s(f64)
    fn __le__(self: Tensor[Float64], rhs: Float32)        -> Tensor[Int]: return le_t(self, to_float64(scalar32(rhs)))        # a <= s(f32→f64)
    fn __le__(self: Tensor[Float64], rhs: Int)            -> Tensor[Int]: return le_t(self, to_float64(scalar_int(rhs)))      # a <= s(int→f64)

    fn __gt__(self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Int]: return gt_t(self, rhs)                               # a > b
    fn __gt__(self: Tensor[Float64], rhs: Float64)        -> Tensor[Int]: return gt_t(self, scalar64(rhs))                    # a > s(f64)
    fn __gt__(self: Tensor[Float64], rhs: Float32)        -> Tensor[Int]: return gt_t(self, to_float64(scalar32(rhs)))        # a > s(f32→f64)
    fn __gt__(self: Tensor[Float64], rhs: Int)            -> Tensor[Int]: return gt_t(self, to_float64(scalar_int(rhs)))      # a > s(int→f64)

    fn __ge__(self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Int]: return ge_t(self, rhs)                               # a >= b
    fn __ge__(self: Tensor[Float64], rhs: Float64)        -> Tensor[Int]: return ge_t(self, scalar64(rhs))                    # a >= s(f64)
    fn __ge__(self: Tensor[Float64], rhs: Float32)        -> Tensor[Int]: return ge_t(self, to_float64(scalar32(rhs)))        # a >= s(f32→f64)
    fn __ge__(self: Tensor[Float64], rhs: Int)            -> Tensor[Int]: return ge_t(self, to_float64(scalar_int(rhs)))      # a >= s(int→f64)

    fn __eq__(self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Int]: return eq_t(self, rhs)                               # a == b
    fn __eq__(self: Tensor[Float64], rhs: Float64)        -> Tensor[Int]: return eq_t(self, scalar64(rhs))                    # a == s(f64)
    fn __eq__(self: Tensor[Float64], rhs: Float32)        -> Tensor[Int]: return eq_t(self, to_float64(scalar32(rhs)))        # a == s(f32→f64)
    fn __eq__(self: Tensor[Float64], rhs: Int)            -> Tensor[Int]: return eq_t(self, to_float64(scalar_int(rhs)))      # a == s(int→f64)

    fn __ne__(self: Tensor[Float64], rhs: Tensor[Float64]) -> Tensor[Int]: return ne_t(self, rhs)                               # a != b
    fn __ne__(self: Tensor[Float64], rhs: Float64)        -> Tensor[Int]: return ne_t(self, scalar64(rhs))                    # a != s(f64)
    fn __ne__(self: Tensor[Float64], rhs: Float32)        -> Tensor[Int]: return ne_t(self, to_float64(scalar32(rhs)))        # a != s(f32→f64)
    fn __ne__(self: Tensor[Float64], rhs: Int)            -> Tensor[Int]: return ne_t(self, to_float64(scalar_int(rhs)))      # a != s(int→f64)
    # =========================
    # Float32 comparisons → mask(Int)
    # Promotion: with Float64 ⇒ compare in Float64, with Int ⇒ compare in Float32
    # =========================
    fn __lt__(self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Int]: return lt_t(self, rhs)                               # a < b
    fn __lt__(self: Tensor[Float32], rhs: Float32)        -> Tensor[Int]: return lt_t(self, scalar32(rhs))                    # a < s(f32)
    fn __lt__(self: Tensor[Float32], rhs: Float64)        -> Tensor[Int]: return lt_t(to_float64(self), scalar64(rhs))        # a(f32→f64) < s(f64)
    fn __lt__(self: Tensor[Float32], rhs: Int)            -> Tensor[Int]: return lt_t(self, to_float32(scalar_int(rhs)))      # a < s(int→f32)

    fn __le__(self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Int]: return le_t(self, rhs)                               # a <= b
    fn __le__(self: Tensor[Float32], rhs: Float32)        -> Tensor[Int]: return le_t(self, scalar32(rhs))                    # a <= s(f32)
    fn __le__(self: Tensor[Float32], rhs: Float64)        -> Tensor[Int]: return le_t(to_float64(self), scalar64(rhs))        # a(f32→f64) <= s(f64)
    fn __le__(self: Tensor[Float32], rhs: Int)            -> Tensor[Int]: return le_t(self, to_float32(scalar_int(rhs)))      # a <= s(int→f32)

    fn __gt__(self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Int]: return gt_t(self, rhs)                               # a > b
    fn __gt__(self: Tensor[Float32], rhs: Float32)        -> Tensor[Int]: return gt_t(self, scalar32(rhs))                    # a > s(f32)
    fn __gt__(self: Tensor[Float32], rhs: Float64)        -> Tensor[Int]: return gt_t(to_float64(self), scalar64(rhs))        # a(f32→f64) > s(f64)
    fn __gt__(self: Tensor[Float32], rhs: Int)            -> Tensor[Int]: return gt_t(self, to_float32(scalar_int(rhs)))      # a > s(int→f32)

    fn __ge__(self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Int]: return ge_t(self, rhs)                               # a >= b
    fn __ge__(self: Tensor[Float32], rhs: Float32)        -> Tensor[Int]: return ge_t(self, scalar32(rhs))                    # a >= s(f32)
    fn __ge__(self: Tensor[Float32], rhs: Float64)        -> Tensor[Int]: return ge_t(to_float64(self), scalar64(rhs))        # a(f32→f64) >= s(f64)
    fn __ge__(self: Tensor[Float32], rhs: Int)            -> Tensor[Int]: return ge_t(self, to_float32(scalar_int(rhs)))      # a >= s(int→f32)

    fn __eq__(self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Int]: return eq_t(self, rhs)                               # a == b
    fn __eq__(self: Tensor[Float32], rhs: Float32)        -> Tensor[Int]: return eq_t(self, scalar32(rhs))                    # a == s(f32)
    fn __eq__(self: Tensor[Float32], rhs: Float64)        -> Tensor[Int]: return eq_t(to_float64(self), scalar64(rhs))        # a(f32→f64) == s(f64)
    fn __eq__(self: Tensor[Float32], rhs: Int)            -> Tensor[Int]: return eq_t(self, to_float32(scalar_int(rhs)))      # a == s(int→f32)

    fn __ne__(self: Tensor[Float32], rhs: Tensor[Float32]) -> Tensor[Int]: return ne_t(self, rhs)                               # a != b
    fn __ne__(self: Tensor[Float32], rhs: Float32)        -> Tensor[Int]: return ne_t(self, scalar32(rhs))                    # a != s(f32)
    fn __ne__(self: Tensor[Float32], rhs: Float64)        -> Tensor[Int]: return ne_t(to_float64(self), scalar64(rhs))        # a(f32→f64) != s(f64)
    fn __ne__(self: Tensor[Float32], rhs: Int)            -> Tensor[Int]: return ne_t(self, to_float32(scalar_int(rhs)))      # a != s(int→f32)
    # =========================
    # Int comparisons → mask(Int)
    # Promotion: with Float64 ⇒ compare in Float64; with Float32 ⇒ compare in Float32; else Int
    # =========================
    fn __lt__(self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]: return lt_t(self, rhs)                                        # a < b
    fn __lt__(self: Tensor[Int], rhs: Int)        -> Tensor[Int]: return lt_t(self, scalar_int(rhs))                            # a < s(int)
    fn __lt__(self: Tensor[Int], rhs: Float32)    -> Tensor[Int]: return lt_t(to_float32(self), scalar32(rhs))                  # a(int→f32) < s(f32)
    fn __lt__(self: Tensor[Int], rhs: Float64)    -> Tensor[Int]: return lt_t(to_float64(self), scalar64(rhs))                  # a(int→f64) < s(f64)

    fn __le__(self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]: return le_t(self, rhs)                                        # a <= b
    fn __le__(self: Tensor[Int], rhs: Int)        -> Tensor[Int]: return le_t(self, scalar_int(rhs))                            # a <= s(int)
    fn __le__(self: Tensor[Int], rhs: Float32)    -> Tensor[Int]: return le_t(to_float32(self), scalar32(rhs))                  # a(int→f32) <= s(f32)
    fn __le__(self: Tensor[Int], rhs: Float64)    -> Tensor[Int]: return le_t(to_float64(self), scalar64(rhs))                  # a(int→f64) <= s(f64)

    fn __gt__(self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]: return gt_t(self, rhs)                                        # a > b
    fn __gt__(self: Tensor[Int], rhs: Int)        -> Tensor[Int]: return gt_t(self, scalar_int(rhs))                            # a > s(int)
    fn __gt__(self: Tensor[Int], rhs: Float32)    -> Tensor[Int]: return gt_t(to_float32(self), scalar32(rhs))                  # a(int→f32) > s(f32)
    fn __gt__(self: Tensor[Int], rhs: Float64)    -> Tensor[Int]: return gt_t(to_float64(self), scalar64(rhs))                  # a(int→f64) > s(f64)

    fn __ge__(self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]: return ge_t(self, rhs)                                        # a >= b
    fn __ge__(self: Tensor[Int], rhs: Int)        -> Tensor[Int]: return ge_t(self, scalar_int(rhs))                            # a >= s(int)
    fn __ge__(self: Tensor[Int], rhs: Float32)    -> Tensor[Int]: return ge_t(to_float32(self), scalar32(rhs))                  # a(int→f32) >= s(f32)
    fn __ge__(self: Tensor[Int], rhs: Float64)    -> Tensor[Int]: return ge_t(to_float64(self), scalar64(rhs))                  # a(int→f64) >= s(f64)

    fn __eq__(self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]: return eq_t(self, rhs)                                        # a == b
    fn __eq__(self: Tensor[Int], rhs: Int)        -> Tensor[Int]: return eq_t(self, scalar_int(rhs))                            # a == s(int)
    fn __eq__(self: Tensor[Int], rhs: Float32)    -> Tensor[Int]: return eq_t(to_float32(self), scalar32(rhs))                  # a(int→f32) == s(f32)
    fn __eq__(self: Tensor[Int], rhs: Float64)    -> Tensor[Int]: return eq_t(to_float64(self), scalar64(rhs))                  # a(int→f64) == s(f64)

    fn __ne__(self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]: return ne_t(self, rhs)                                        # a != b
    fn __ne__(self: Tensor[Int], rhs: Int)        -> Tensor[Int]: return ne_t(self, scalar_int(rhs))                            # a != s(int)
    fn __ne__(self: Tensor[Int], rhs: Float32)    -> Tensor[Int]: return ne_t(to_float32(self), scalar32(rhs))                  # a(int→f32) != s(f32)
    fn __ne__(self: Tensor[Int], rhs: Float64)    -> Tensor[Int]: return ne_t(to_float64(self), scalar64(rhs))                  # a(int→f64) != s(f64)

    # =========================
    # Scalar arithmetic with full promotion (Int < Float32 < Float64)
    # Uses only: scalar_int/scalar32/scalar64, to_int/to_float32/to_float64, add_t/sub_t/mul_t/div_t/mod_t/pow_t
    # =========================

    # -------------------------
    # Tensor[Float64] overloads
    # -------------------------
    fn add_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Float64]: return add_t(self, scalar64(s))                    # a + s(f64)
    fn add_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Float64]: return add_t(self, to_float64(scalar32(s)))       # a + s(f32→f64)
    fn add_scalar(self: Tensor[Float64], s: Int)     -> Tensor[Float64]: return add_t(self, to_float64(scalar_int(s)))     # a + s(int→f64)

    fn sub_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Float64]: return sub_t(self, scalar64(s))                    # a - s(f64)
    fn sub_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Float64]: return sub_t(self, to_float64(scalar32(s)))       # a - s(f32→f64)
    fn sub_scalar(self: Tensor[Float64], s: Int)     -> Tensor[Float64]: return sub_t(self, to_float64(scalar_int(s)))     # a - s(int→f64)

    fn mul_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Float64]: return mul_t(self, scalar64(s))                    # a * s(f64)
    fn mul_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Float64]: return mul_t(self, to_float64(scalar32(s)))       # a * s(f32→f64)
    fn mul_scalar(self: Tensor[Float64], s: Int)     -> Tensor[Float64]: return mul_t(self, to_float64(scalar_int(s)))     # a * s(int→f64)

    fn div_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Float64]: return div_t(self, scalar64(s))                    # a / s(f64)
    fn div_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Float64]: return div_t(self, to_float64(scalar32(s)))       # a / s(f32→f64)
    fn div_scalar(self: Tensor[Float64], s: Int)     -> Tensor[Float64]: return div_t(self, to_float64(scalar_int(s)))     # a / s(int→f64)

    fn mod_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Float64]: return mod_t(self, scalar64(s))                    # a % s(f64)
    fn mod_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Float64]: return mod_t(self, to_float64(scalar32(s)))       # a % s(f32→f64)
    fn mod_scalar(self: Tensor[Float64], s: Int)     -> Tensor[Float64]: return mod_t(self, to_float64(scalar_int(s)))     # a % s(int→f64)

    fn pow_scalar(self: Tensor[Float64], p: Float64) -> Tensor[Float64]: return pow_t(self, scalar64(p))                    # a ** p(f64)
    fn pow_scalar(self: Tensor[Float64], p: Float32) -> Tensor[Float64]: return pow_t(self, to_float64(scalar32(p)))       # a ** p(f32→f64)
    fn pow_scalar(self: Tensor[Float64], p: Int)     -> Tensor[Float64]: return pow_t(self, to_float64(scalar_int(p)))     # a ** p(int→f64)

    # -------------------------
    # Tensor[Float32] overloads
    # -------------------------
    fn add_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Float64]: return add_t(to_float64(self), scalar64(s))        # (a→f64) + s(f64)
    fn add_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Float32]: return add_t(self, scalar32(s))                     # a + s(f32)
    fn add_scalar(self: Tensor[Float32], s: Int)     -> Tensor[Float32]: return add_t(self, to_float32(scalar_int(s)))      # a + s(int→f32)

    fn sub_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Float64]: return sub_t(to_float64(self), scalar64(s))        # (a→f64) - s(f64)
    fn sub_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Float32]: return sub_t(self, scalar32(s))                     # a - s(f32)
    fn sub_scalar(self: Tensor[Float32], s: Int)     -> Tensor[Float32]: return sub_t(self, to_float32(scalar_int(s)))      # a - s(int→f32)

    fn mul_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Float64]: return mul_t(to_float64(self), scalar64(s))        # (a→f64) * s(f64)
    fn mul_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Float32]: return mul_t(self, scalar32(s))                     # a * s(f32)
    fn mul_scalar(self: Tensor[Float32], s: Int)     -> Tensor[Float32]: return mul_t(self, to_float32(scalar_int(s)))      # a * s(int→f32)

    fn div_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Float64]: return div_t(to_float64(self), scalar64(s))        # (a→f64) / s(f64)
    fn div_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Float64]: return div_t(self, scalar32(s))                     # a / s(f32)
    fn div_scalar(self: Tensor[Float32], s: Int)     -> Tensor[Float64]: return div_t(self, to_float32(scalar_int(s)))      # a / s(int→f32)

    fn mod_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Float64]: return mod_t(to_float64(self), scalar64(s))        # (a→f64) % s(f64)
    fn mod_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Float64]: return mod_t(self, scalar32(s))                     # a % s(f32)
    fn mod_scalar(self: Tensor[Float32], s: Int)     -> Tensor[Float64]: return mod_t(self, to_float32(scalar_int(s)))      # a % s(int→f32)

    fn pow_scalar(self: Tensor[Float32], p: Float64) -> Tensor[Float64]: return pow_t(to_float64(self), scalar64(p))        # (a→f64) ** p(f64)
    fn pow_scalar(self: Tensor[Float32], p: Float32) -> Tensor[Float32]: return pow_t(self, scalar32(p))                     # a ** p(f32)
    fn pow_scalar(self: Tensor[Float32], p: Int)     -> Tensor[Float32]: return pow_t(self, to_float32(scalar_int(p)))      # a ** p(int→f32)

    # -------------------------
    # Tensor[Int] overloads
    # -------------------------
    fn add_scalar(self: Tensor[Int], s: Float64) -> Tensor[Float64]: return add_t(to_float64(self), scalar64(s))            # (a→f64) + s(f64)
    fn add_scalar(self: Tensor[Int], s: Float32) -> Tensor[Float32]: return add_t(to_float32(self), scalar32(s))            # (a→f32) + s(f32)
    fn add_scalar(self: Tensor[Int], s: Int)     -> Tensor[Int]:     return add_t(self, scalar_int(s))                      # a + s(int)

    fn sub_scalar(self: Tensor[Int], s: Float64) -> Tensor[Float64]: return sub_t(to_float64(self), scalar64(s))            # (a→f64) - s(f64)
    fn sub_scalar(self: Tensor[Int], s: Float32) -> Tensor[Float32]: return sub_t(to_float32(self), scalar32(s))            # (a→f32) - s(f32)
    fn sub_scalar(self: Tensor[Int], s: Int)     -> Tensor[Int]:     return sub_t(self, scalar_int(s))                      # a - s(int)

    fn mul_scalar(self: Tensor[Int], s: Float64) -> Tensor[Float64]: return mul_t(to_float64(self), scalar64(s))            # (a→f64) * s(f64)
    fn mul_scalar(self: Tensor[Int], s: Float32) -> Tensor[Float32]: return mul_t(to_float32(self), scalar32(s))            # (a→f32) * s(f32)
    fn mul_scalar(self: Tensor[Int], s: Int)     -> Tensor[Int]:     return mul_t(self, scalar_int(s))                      # a * s(int)

    fn div_scalar(self: Tensor[Int], s: Float64) -> Tensor[Float64]: return div_t(to_float64(self), scalar64(s))            # (a→f64) / s(f64)
    fn div_scalar(self: Tensor[Int], s: Float32) -> Tensor[Float64]: return div_t(to_float32(self), scalar32(s))            # (a→f32) / s(f32)
    fn div_scalar(self: Tensor[Int], s: Int)     -> Tensor[Float64]:     return div_t(self, scalar_int(s))                      # a / s(int)

    fn mod_scalar(self: Tensor[Int], s: Float64) -> Tensor[Float64]: return mod_t(to_float64(self), scalar64(s))            # (a→f64) % s(f64)
    fn mod_scalar(self: Tensor[Int], s: Float32) -> Tensor[Float64]: return mod_t(to_float32(self), scalar32(s))            # (a→f32) % s(f32)
    fn mod_scalar(self: Tensor[Int], s: Int)     -> Tensor[Float64]:     return mod_t(self, scalar_int(s))                      # a % s(int)

    fn pow_scalar(self: Tensor[Int], p: Float64) -> Tensor[Float64]: return pow_t(to_float64(self), scalar64(p))            # (a→f64) ** p(f64)
    fn pow_scalar(self: Tensor[Int], p: Float32) -> Tensor[Float32]: return pow_t(to_float32(self), scalar32(p))            # (a→f32) ** p(f32)
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
    fn mod   (self: Tensor[Float32], other: Tensor[Float32]) -> Tensor[Float32]: return mod_t(self, other)                 # a % b (f32,f32→f32)
    fn mod   (self: Tensor[Float32], other: Tensor[Int])     -> Tensor[Float32]: return mod_t(self, to_float32(other))     # a % b (f32,int→f32)

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
    fn divide(self: Tensor[Int], other: Tensor[Float32]) -> Tensor[Float32]: return div_t(to_float32(self), other)         # a / b (int,f32→f32)
    fn divide(self: Tensor[Int], other: Tensor[Int])     -> Tensor[Int]:     return div_t(self, other)                     # a / b (int,int→int)

    fn mod   (self: Tensor[Int], other: Tensor[Float64]) -> Tensor[Float64]: return mod_t(to_float64(self), other)         # a % b (int,f64→f64)
    fn mod   (self: Tensor[Int], other: Tensor[Float32]) -> Tensor[Float32]: return mod_t(to_float32(self), other)         # a % b (int,f32→f32)
    fn mod   (self: Tensor[Int], other: Tensor[Int])     -> Tensor[Int]:     return mod_t(self, other)                     # a % b (int,int→int)

    fn pow   (self: Tensor[Int], other: Tensor[Float64]) -> Tensor[Float64]: return pow_t(to_float64(self), other)         # a ** b (int,f64→f64)
    fn pow   (self: Tensor[Int], other: Tensor[Float32]) -> Tensor[Float32]: return pow_t(to_float32(self), other)         # a ** b (int,f32→f32)
    fn pow   (self: Tensor[Int], other: Tensor[Int])     -> Tensor[Int]:     return pow_t(self, other)                     # a ** b (int,int→int)

    # =========================
    # Float64 overloads (full combos)
    # =========================

    # Logical (bitwise-style over numeric domain; result promoted by rule)
    fn and_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Int]: return to_int(and_t(self, scalar64(s)))                                         # a(f64) & s(f64) ⇒ f64
    fn and_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Int]: return to_int(and_t(self, to_float64(scalar32(s))))                             # a(f64) & s(f32→f64) ⇒ f64
    fn and_scalar(self: Tensor[Float64], s: Int)     -> Tensor[Int]: return to_int(and_t(self, to_float64(scalar_int(s))))                           # a(f64) & s(int→f64) ⇒ f64

    fn or_scalar (self: Tensor[Float64], s: Float64) -> Tensor[Int]: return to_int(or_t (self, scalar64(s)))                                         # a(f64) | s(f64) ⇒ f64
    fn or_scalar (self: Tensor[Float64], s: Float32) -> Tensor[Int]: return to_int(or_t (self, to_float64(scalar32(s))))                             # a(f64) | s(f32→f64) ⇒ f64
    fn or_scalar (self: Tensor[Float64], s: Int)     -> Tensor[Int]: return to_int(or_t (self, to_float64(scalar_int(s))))                           # a(f64) | s(int→f64) ⇒ f64

    fn xor_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Int]: return to_int(xor_t(self, scalar64(s)))                                         # a(f64) ^ s(f64) ⇒ f64
    fn xor_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Int]: return to_int(xor_t(self, to_float64(scalar32(s))))                             # a(f64) ^ s(f32→f64) ⇒ f64
    fn xor_scalar(self: Tensor[Float64], s: Int)     -> Tensor[Int]: return to_int(xor_t(self, to_float64(scalar_int(s))))                           # a(f64) ^ s(int→f64) ⇒ f64

    # Logical boolean ops (mask-style; returns Bool)
    fn logical_not(self: Tensor[Float64]) -> Tensor[Bool]: return lnot_t(self)                                                                    # ~a ⇒ bool mask
    fn logical_and(self: Tensor[Float64], other: Tensor[Float64]) -> Tensor[Bool]: return land_t(self, other)                                     # a(f64) & b(f64) ⇒ bool
    fn logical_and(self: Tensor[Float64], other: Tensor[Float32]) -> Tensor[Bool]: return land_t(self, to_float64(other))                         # a(f64) & b(f32→f64) ⇒ bool
    fn logical_and(self: Tensor[Float64], other: Tensor[Int])     -> Tensor[Bool]: return land_t(self, to_float64(other))                         # a(f64) & b(int→f64) ⇒ bool

    fn logical_or (self: Tensor[Float64], other: Tensor[Float64]) -> Tensor[Bool]: return lor_t (self, other)                                     # a(f64) | b(f64) ⇒ bool
    fn logical_or (self: Tensor[Float64], other: Tensor[Float32]) -> Tensor[Bool]: return lor_t (self, to_float64(other))                         # a(f64) | b(f32→f64) ⇒ bool
    fn logical_or (self: Tensor[Float64], other: Tensor[Int])     -> Tensor[Bool]: return lor_t (self, to_float64(other))                         # a(f64) | b(int→f64) ⇒ bool

    fn logical_xor(self: Tensor[Float64], other: Tensor[Float64]) -> Tensor[Bool]: return lxor_t(self, other)                                     # a(f64) ^ b(f64) ⇒ bool
    fn logical_xor(self: Tensor[Float64], other: Tensor[Float32]) -> Tensor[Bool]: return lxor_t(self, to_float64(other))                         # a(f64) ^ b(f32→f64) ⇒ bool
    fn logical_xor(self: Tensor[Float64], other: Tensor[Int])     -> Tensor[Bool]: return lxor_t(self, to_float64(other))                         # a(f64) ^ b(int→f64) ⇒ bool


    # =========================
    # Float32 overloads (full combos)
    # =========================

    # Numeric bitwise with scalar (result promoted by rule)
    fn and_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Int]: return to_int(and_t(to_float64(self), scalar64(s)))                              # a(f32→f64) & s(f64) ⇒ f64
    fn and_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Int]: return to_int(and_t(self, scalar32(s)))                                         # a(f32) & s(f32) ⇒ f32
    fn and_scalar(self: Tensor[Float32], s: Int)     -> Tensor[Int]: return to_int(and_t(self, to_float32(scalar_int(s))))                           # a(f32) & s(int→f32) ⇒ f32

    fn or_scalar (self: Tensor[Float32], s: Float64) -> Tensor[Int]: return to_int(or_t (to_float64(self), scalar64(s)))                             # a(f32→f64) | s(f64) ⇒ f64
    fn or_scalar (self: Tensor[Float32], s: Float32) -> Tensor[Int]: return to_int(or_t (self, scalar32(s)))                                         # a(f32) | s(f32) ⇒ f32
    fn or_scalar (self: Tensor[Float32], s: Int)     -> Tensor[Int]: return to_int(or_t (self, to_float32(scalar_int(s))))                           # a(f32) | s(int→f32) ⇒ f32

    fn xor_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Int]: return to_int(xor_t(to_float64(self), scalar64(s)))                             # a(f32→f64) ^ s(f64) ⇒ f64
    fn xor_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Int]: return to_int(xor_t(self, scalar32(s)))                                         # a(f32) ^ s(f32) ⇒ f32
    fn xor_scalar(self: Tensor[Float32], s: Int)     -> Tensor[Int]: return to_int(xor_t(self, to_float32(scalar_int(s))))                           # a(f32) ^ s(int→f32) ⇒ f32

    # Boolean logical with tensor (return Bool; promote both args)
    fn logical_not(self: Tensor[Float32]) -> Tensor[Bool]: return lnot_t(self)                                                                     # ~a ⇒ bool mask
    fn logical_and(self: Tensor[Float32], other: Tensor[Float64]) -> Tensor[Bool]: return land_t(to_float64(self), other)                         # a(f32→f64) & b(f64) ⇒ bool
    fn logical_and(self: Tensor[Float32], other: Tensor[Float32]) -> Tensor[Bool]: return land_t(self, other)                                     # a(f32) & b(f32) ⇒ bool
    fn logical_and(self: Tensor[Float32], other: Tensor[Int])     -> Tensor[Bool]: return land_t(self, to_float32(other))                         # a(f32) & b(int→f32) ⇒ bool

    fn logical_or (self: Tensor[Float32], other: Tensor[Float64]) -> Tensor[Bool]: return lor_t (to_float64(self), other)                         # a(f32→f64) | b(f64) ⇒ bool
    fn logical_or (self: Tensor[Float32], other: Tensor[Float32]) -> Tensor[Bool]: return lor_t (self, other)                                     # a(f32) | b(f32) ⇒ bool
    fn logical_or (self: Tensor[Float32], other: Tensor[Int])     -> Tensor[Bool]: return lor_t (self, to_float32(other))                         # a(f32) | b(int→f32) ⇒ bool

    fn logical_xor(self: Tensor[Float32], other: Tensor[Float64]) -> Tensor[Bool]: return lxor_t(to_float64(self), other)                         # a(f32→f64) ^ b(f64) ⇒ bool
    fn logical_xor(self: Tensor[Float32], other: Tensor[Float32]) -> Tensor[Bool]: return lxor_t(self, other)                                     # a(f32) ^ b(f32) ⇒ bool
    fn logical_xor(self: Tensor[Float32], other: Tensor[Int])     -> Tensor[Bool]: return lxor_t(self, to_float32(other))                         # a(f32) ^ b(int→f32) ⇒ bool


    # =========================
    # Int overloads (full combos)
    # =========================

    # Numeric bitwise with scalar (result promoted by rule)
    fn and_scalar(self: Tensor[Int], s: Float64) -> Tensor[Int]: return to_int(and_t(to_float64(self), scalar64(s)))                                   # a(int→f64) & s(f64) ⇒ f64
    fn and_scalar(self: Tensor[Int], s: Float32) -> Tensor[Int]: return to_int(and_t(to_float32(self), scalar32(s)))                                  # a(int→f32) & s(f32) ⇒ f32
    fn and_scalar(self: Tensor[Int], s: Int)     -> Tensor[Int]:     return and_t(self, scalar_int(s))                                           # a(int) & s(int) ⇒ int

    fn or_scalar (self: Tensor[Int], s: Float64) -> Tensor[Int]: return to_int(or_t (to_float64(self), scalar64(s)))                                  # a(int→f64) | s(f64) ⇒ f64
    fn or_scalar (self: Tensor[Int], s: Float32) -> Tensor[Int]: return to_int(or_t (to_float32(self), scalar32(s)))                                  # a(int→f32) | s(f32) ⇒ f32
    fn or_scalar (self: Tensor[Int], s: Int)     -> Tensor[Int]:     return or_t (self, scalar_int(s))                                           # a(int) | s(int) ⇒ int

    fn xor_scalar(self: Tensor[Int], s: Float64) -> Tensor[Int]: return to_int(xor_t(to_float64(self), scalar64(s)))                                   # a(int→f64) ^ s(f64) ⇒ f64
    fn xor_scalar(self: Tensor[Int], s: Float32) -> Tensor[Int]: return to_int(xor_t(to_float32(self), scalar32(s)))                                   # a(int→f32) ^ s(f32) ⇒ f32
    fn xor_scalar(self: Tensor[Int], s: Int)     -> Tensor[Int]:     return xor_t(self, scalar_int(s))                                           # a(int) ^ s(int) ⇒ int

    # Boolean logical with tensor (return Bool; promote both args)
    fn logical_not(self: Tensor[Int]) -> Tensor[Bool]: return lnot_t(self)                                                                         # ~a ⇒ bool mask
    fn logical_and(self: Tensor[Int], other: Tensor[Float64]) -> Tensor[Bool]: return land_t(to_float64(self), other)                              # a(int→f64) & b(f64) ⇒ bool
    fn logical_and(self: Tensor[Int], other: Tensor[Float32]) -> Tensor[Bool]: return land_t(to_float32(self), other)                              # a(int→f32) & b(f32) ⇒ bool
    fn logical_and(self: Tensor[Int], other: Tensor[Int])     -> Tensor[Bool]: return land_t(self, other)                                          # a(int) & b(int) ⇒ bool

    fn logical_or (self: Tensor[Int], other: Tensor[Float64]) -> Tensor[Bool]: return lor_t (to_float64(self), other)                              # a(int→f64) | b(f64) ⇒ bool
    fn logical_or (self: Tensor[Int], other: Tensor[Float32]) -> Tensor[Bool]: return lor_t (to_float32(self), other)                              # a(int→f32) | b(f32) ⇒ bool
    fn logical_or (self: Tensor[Int], other: Tensor[Int])     -> Tensor[Bool]: return lor_t (self, other)                                          # a(int) | b(int) ⇒ bool

    fn logical_xor(self: Tensor[Int], other: Tensor[Float64]) -> Tensor[Bool]: return lxor_t(to_float64(self), other)                              # a(int→f64) ^ b(f64) ⇒ bool
    fn logical_xor(self: Tensor[Int], other: Tensor[Float32]) -> Tensor[Bool]: return lxor_t(to_float32(self), other)                              # a(int→f32) ^ b(f32) ⇒ bool
    fn logical_xor(self: Tensor[Int], other: Tensor[Int])     -> Tensor[Bool]: return lxor_t(self, other)                                          # a(int) ^ b(int) ⇒ bool
    # =========================
    # Float64 overloads — Shifts (full combos, one-liners)
    # Promotion: Int < Float32 < Float64
    # =========================

    # Scalar
    fn lshift_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Float64]: return shl_t(self, scalar64(s))                                # a<<s (f64)
    fn lshift_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Float64]: return shl_t(self, to_float64(scalar32(s)))                   # a<<s (f32→f64)
    fn lshift_scalar(self: Tensor[Float64], s: Int)     -> Tensor[Float64]: return shl_t(self, to_float64(scalar_int(s)))                 # a<<s (int→f64)
    fn rshift_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Float64]: return shr_t(self, scalar64(s))                                # a>>s (f64)
    fn rshift_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Float64]: return shr_t(self, to_float64(scalar32(s)))                   # a>>s (f32→f64)
    fn rshift_scalar(self: Tensor[Float64], s: Int)     -> Tensor[Float64]: return shr_t(self, to_float64(scalar_int(s)))                 # a>>s (int→f64)

    # Tensor
    fn lshift(self: Tensor[Float64], other: Tensor[Float64]) -> Tensor[Float64]: return shl_t(self, other)                                  # a<<b (f64,f64)
    fn lshift(self: Tensor[Float64], other: Tensor[Float32]) -> Tensor[Float64]: return shl_t(self, to_float64(other))                     # a<<b (f64,f32→f64)
    fn lshift(self: Tensor[Float64], other: Tensor[Int])     -> Tensor[Float64]: return shl_t(self, to_float64(other))                     # a<<b (f64,int→f64)
    fn rshift(self: Tensor[Float64], other: Tensor[Float64]) -> Tensor[Float64]: return shr_t(self, other)                                  # a>>b (f64,f64)
    fn rshift(self: Tensor[Float64], other: Tensor[Float32]) -> Tensor[Float64]: return shr_t(self, to_float64(other))                     # a>>b (f64,f32→f64)
    fn rshift(self: Tensor[Float64], other: Tensor[Int])     -> Tensor[Float64]: return shr_t(self, to_float64(other))                     # a>>b (f64,int→f64)


    # =========================
    # Float32 overloads — Shifts (full combos, one-liners)
    # Promotion: Int < Float32 < Float64
    # =========================

    # Scalar (note: any f64 ⇒ promote to f64)
    fn lshift_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Float64]: return shl_t(to_float64(self), scalar64(s))                   # a<<s (self→f64, s f64)
    fn lshift_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Float32]: return shl_t(self, scalar32(s))                               # a<<s (f32)
    fn lshift_scalar(self: Tensor[Float32], s: Int)     -> Tensor[Float32]: return shl_t(self, to_float32(scalar_int(s)))                # a<<s (int→f32)
    fn rshift_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Float64]: return shr_t(to_float64(self), scalar64(s))                   # a>>s (self→f64, s f64)
    fn rshift_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Float32]: return shr_t(self, scalar32(s))                               # a>>s (f32)
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
    fn lshift_scalar(self: Tensor[Int], s: Float64) -> Tensor[Float64]: return shl_t(to_float64(self), scalar64(s))                       # a<<s (→f64)
    fn lshift_scalar(self: Tensor[Int], s: Float32) -> Tensor[Float32]: return shl_t(to_float32(self), scalar32(s))                      # a<<s (→f32)
    fn lshift_scalar(self: Tensor[Int], s: Int)     -> Tensor[Int]:     return shl_t(self, scalar_int(s))                                # a<<s (int)
    fn rshift_scalar(self: Tensor[Int], s: Float64) -> Tensor[Float64]: return shr_t(to_float64(self), scalar64(s))                       # a>>s (→f64)
    fn rshift_scalar(self: Tensor[Int], s: Float32) -> Tensor[Float32]: return shr_t(to_float32(self), scalar32(s))                      # a>>s (→f32)
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
    fn lt_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Int]: return lt_t(self, scalar64(s))                                # a < s(f64)
    fn lt_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Int]: return lt_t(self, to_float64(scalar32(s)))                   # a < s(f32→f64)
    fn lt_scalar(self: Tensor[Float64], s: Int)     -> Tensor[Int]: return lt_t(self, to_float64(scalar_int(s)))                 # a < s(int→f64)

    fn le_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Int]: return le_t(self, scalar64(s))                                # a <= s(f64)
    fn le_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Int]: return le_t(self, to_float64(scalar32(s)))                   # a <= s(f32→f64)
    fn le_scalar(self: Tensor[Float64], s: Int)     -> Tensor[Int]: return le_t(self, to_float64(scalar_int(s)))                 # a <= s(int→f64)

    fn gt_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Int]: return gt_t(self, scalar64(s))                                # a > s(f64)
    fn gt_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Int]: return gt_t(self, to_float64(scalar32(s)))                   # a > s(f32→f64)
    fn gt_scalar(self: Tensor[Float64], s: Int)     -> Tensor[Int]: return gt_t(self, to_float64(scalar_int(s)))                 # a > s(int→f64)

    fn ge_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Int]: return ge_t(self, scalar64(s))                                # a >= s(f64)
    fn ge_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Int]: return ge_t(self, to_float64(scalar32(s)))                   # a >= s(f32→f64)
    fn ge_scalar(self: Tensor[Float64], s: Int)     -> Tensor[Int]: return ge_t(self, to_float64(scalar_int(s)))                 # a >= s(int→f64)

    fn eq_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Int]: return eq_t(self, scalar64(s))                                # a == s(f64)
    fn eq_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Int]: return eq_t(self, to_float64(scalar32(s)))                   # a == s(f32→f64)
    fn eq_scalar(self: Tensor[Float64], s: Int)     -> Tensor[Int]: return eq_t(self, to_float64(scalar_int(s)))                 # a == s(int→f64)

    fn ne_scalar(self: Tensor[Float64], s: Float64) -> Tensor[Int]: return ne_t(self, scalar64(s))                                # a != s(f64)
    fn ne_scalar(self: Tensor[Float64], s: Float32) -> Tensor[Int]: return ne_t(self, to_float64(scalar32(s)))                   # a != s(f32→f64)
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
    fn lt_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Int]: return lt_t(to_float64(self), scalar64(s))                   # a(f32→f64) < s(f64)
    fn lt_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Int]: return lt_t(self, scalar32(s))                               # a(f32) < s(f32)
    fn lt_scalar(self: Tensor[Float32], s: Int)     -> Tensor[Int]: return lt_t(self, to_float32(scalar_int(s)))                 # a(f32) < s(int→f32)

    fn le_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Int]: return le_t(to_float64(self), scalar64(s))                   # a(f32→f64) <= s(f64)
    fn le_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Int]: return le_t(self, scalar32(s))                               # a(f32) <= s(f32)
    fn le_scalar(self: Tensor[Float32], s: Int)     -> Tensor[Int]: return le_t(self, to_float32(scalar_int(s)))                 # a(f32) <= s(int→f32)

    fn gt_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Int]: return gt_t(to_float64(self), scalar64(s))                   # a(f32→f64) > s(f64)
    fn gt_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Int]: return gt_t(self, scalar32(s))                               # a(f32) > s(f32)
    fn gt_scalar(self: Tensor[Float32], s: Int)     -> Tensor[Int]: return gt_t(self, to_float32(scalar_int(s)))                 # a(f32) > s(int→f32)

    fn ge_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Int]: return ge_t(to_float64(self), scalar64(s))                   # a(f32→f64) >= s(f64)
    fn ge_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Int]: return ge_t(self, scalar32(s))                               # a(f32) >= s(f32)
    fn ge_scalar(self: Tensor[Float32], s: Int)     -> Tensor[Int]: return ge_t(self, to_float32(scalar_int(s)))                 # a(f32) >= s(int→f32)

    fn eq_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Int]: return eq_t(to_float64(self), scalar64(s))                   # a(f32→f64) == s(f64)
    fn eq_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Int]: return eq_t(self, scalar32(s))                               # a(f32) == s(f32)
    fn eq_scalar(self: Tensor[Float32], s: Int)     -> Tensor[Int]: return eq_t(self, to_float32(scalar_int(s)))                 # a(f32) == s(int→f32)

    fn ne_scalar(self: Tensor[Float32], s: Float64) -> Tensor[Int]: return ne_t(to_float64(self), scalar64(s))                   # a(f32→f64) != s(f64)
    fn ne_scalar(self: Tensor[Float32], s: Float32) -> Tensor[Int]: return ne_t(self, scalar32(s))                               # a(f32) != s(f32)
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
    fn lt_scalar(self: Tensor[Int], s: Float64) -> Tensor[Int]: return lt_t(to_float64(self), scalar64(s))                        # a(int→f64) < s(f64)
    fn lt_scalar(self: Tensor[Int], s: Float32) -> Tensor[Int]: return lt_t(to_float32(self), scalar32(s))                        # a(int→f32) < s(f32)
    fn lt_scalar(self: Tensor[Int], s: Int)     -> Tensor[Int]: return lt_t(self, scalar_int(s))                                  # a(int) < s(int)

    fn le_scalar(self: Tensor[Int], s: Float64) -> Tensor[Int]: return le_t(to_float64(self), scalar64(s))                        # a(int→f64) <= s(f64)
    fn le_scalar(self: Tensor[Int], s: Float32) -> Tensor[Int]: return le_t(to_float32(self), scalar32(s))                        # a(int→f32) <= s(f32)
    fn le_scalar(self: Tensor[Int], s: Int)     -> Tensor[Int]: return le_t(self, scalar_int(s))                                  # a(int) <= s(int)

    fn gt_scalar(self: Tensor[Int], s: Float64) -> Tensor[Int]: return gt_t(to_float64(self), scalar64(s))                        # a(int→f64) > s(f64)
    fn gt_scalar(self: Tensor[Int], s: Float32) -> Tensor[Int]: return gt_t(to_float32(self), scalar32(s))                        # a(int→f32) > s(f32)
    fn gt_scalar(self: Tensor[Int], s: Int)     -> Tensor[Int]: return gt_t(self, scalar_int(s))                                  # a(int) > s(int)

    fn ge_scalar(self: Tensor[Int], s: Float64) -> Tensor[Int]: return ge_t(to_float64(self), scalar64(s))                        # a(int→f64) >= s(f64)
    fn ge_scalar(self: Tensor[Int], s: Float32) -> Tensor[Int]: return ge_t(to_float32(self), scalar32(s))                        # a(int→f32) >= s(f32)
    fn ge_scalar(self: Tensor[Int], s: Int)     -> Tensor[Int]: return ge_t(self, scalar_int(s))                                  # a(int) >= s(int)

    fn eq_scalar(self: Tensor[Int], s: Float64) -> Tensor[Int]: return eq_t(to_float64(self), scalar64(s))                        # a(int→f64) == s(f64)
    fn eq_scalar(self: Tensor[Int], s: Float32) -> Tensor[Int]: return eq_t(to_float32(self), scalar32(s))                        # a(int→f32) == s(f32)
    fn eq_scalar(self: Tensor[Int], s: Int)     -> Tensor[Int]: return eq_t(self, scalar_int(s))                                  # a(int) == s(int)

    fn ne_scalar(self: Tensor[Int], s: Float64) -> Tensor[Int]: return ne_t(to_float64(self), scalar64(s))                        # a(int→f64) != s(f64)
    fn ne_scalar(self: Tensor[Int], s: Float32) -> Tensor[Int]: return ne_t(to_float32(self), scalar32(s))                        # a(int→f32) != s(f32)
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

    fn flatten(x: Tensor[Float64]) -> Tensor[Float64]:        return flatten(x)
    fn flatten(x: Tensor[Float32]) -> Tensor[Float32]:        return flatten(x)
    fn flatten(x: Tensor[Int]) -> Tensor[Int]:        return flatten(x)
    
    fn view(x: Tensor[Float64],shape: List[Int]) -> Tensor[Float64]:        return view(x,shape)
    fn view(x: Tensor[Float32],shape: List[Int]) -> Tensor[Float32]:        return view(x,shape)
    fn view(x: Tensor[Int],shape: List[Int]) -> Tensor[Int]:        return view(x,shape)
    

    fn dtype_name(x: Tensor[Bool])    -> String: return "Bool"
    fn dtype_name(x: Tensor[Int])     -> String: return "Int"
    fn dtype_name(x: Tensor[UInt])    -> String: return "UInt"
    fn dtype_name(x: Tensor[Int8])    -> String: return "Int8"
    fn dtype_name(x: Tensor[Int16])   -> String: return "Int16"
    fn dtype_name(x: Tensor[Int32])   -> String: return "Int32"
    fn dtype_name(x: Tensor[Int64])   -> String: return "Int64"
    fn dtype_name(x: Tensor[UInt8])   -> String: return "UInt8"
    fn dtype_name(x: Tensor[UInt16])  -> String: return "UInt16"
    fn dtype_name(x: Tensor[UInt32])  -> String: return "UInt32"
    fn dtype_name(x: Tensor[UInt64])  -> String: return "UInt64"
    fn dtype_name(x: Tensor[Float32]) -> String: return "Float32"
    fn dtype_name(x: Tensor[Float64]) -> String: return "Float64"
 
    fn is_contiguous(self: Tensor[Float64]) -> Bool:    return is_row_major_contiguous(self._shape, self._strides)
    fn is_contiguous(self: Tensor[Float32]) -> Bool:    return is_row_major_contiguous(self._shape, self._strides)
    fn is_contiguous(self: Tensor[Int]) -> Bool:        return is_row_major_contiguous(self._shape, self._strides)


    fn sigmoid(self: Tensor[Float64]) -> Tensor[Float64]: return sigmoid_t(self) 
    fn sigmoid(self: Tensor[Float32]) -> Tensor[Float64]: return sigmoid_t(self) 
    fn sigmoid(self: Tensor[Int])     -> Tensor[Float64]: return sigmoid_t(self) 


    fn contiguous(self: Tensor[Float64]) -> Tensor[Float64]:
        if is_row_major_contiguous(self._shape, self._strides):
            return self.copy()
        var flat = self.flatten()
        var shp = copy_list_int(self._shape)
        var strides = compute_row_major_strides(shp)
        return Tensor[Float64](flat._data, shp, strides) 

    fn contiguous(self: Tensor[Float32]) -> Tensor[Float32]:
        if is_row_major_contiguous(self._shape, self._strides):
            return self.copy()
        var flat = self.flatten()
        var shp = copy_list_int(self._shape)
        var strides = compute_row_major_strides(shp)
        return Tensor[Float32](flat._data, shp, strides) 

    fn contiguous(self: Tensor[Int]) -> Tensor[Int]:
        if is_row_major_contiguous(self._shape, self._strides):
            return self.copy()
        var flat = self.flatten()
        var shp = copy_list_int(self._shape)
        var strides = compute_row_major_strides(shp)
        return Tensor[Int](flat._data, shp, strides)



        # ---------------- High-performance appends on Tensor._data ---------------- #
    @always_inline
    fn _reserve_extra(mut self, extra: Int) -> None:
        # Ensure capacity for 'extra' more elements.
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


#     # MIT License
# # SPDX-License-Identifier: MIT
# # Project:      Momijo
# # Module:       tensor.ops.generic
# # File:         src/momijo/tensor/ops_generic.mojo
# #
# # Description:
# #   Generic dunder/method surface for Tensor[T]:
# #     - Arithmetic / power / unary / comparisons → generic for T
# #     - Logical AND/OR/XOR (project-style) → generic for T
# #     - True bitwise + shifts → Int-only overrides
# #   Each function is multi-line and documented with a short usage hint. 
 

#     # ---------- Arithmetic ----------

#     @always_inline
#     fn __add__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], rhs: Tensor[T]
#     ) -> Tensor[T]:
#         # Usage: c = a + b  # elementwise add, broadcasting supported
#         return add_t(self, rhs)

#     @always_inline
#     fn __add__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], rhs: T
#     ) -> Tensor[T]:
#         # Usage: c = a + 3  # add scalar to tensor
#         return add_t(self, _scalar_of(rhs))

#     @always_inline
#     fn __sub__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], rhs: Tensor[T]
#     ) -> Tensor[T]:
#         # Usage: c = a - b  # elementwise subtract
#         return sub_t(self, rhs)

#     @always_inline
#     fn __sub__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], rhs: T
#     ) -> Tensor[T]:
#         # Usage: c = a - 3  # subtract scalar
#         return sub_t(self, _scalar_of(rhs))

#     @always_inline
#     fn __mul__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], rhs: Tensor[T]
#     ) -> Tensor[T]:
#         # Usage: c = a * b  # elementwise multiply
#         return mul_t(self, rhs)

#     @always_inline
#     fn __mul__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], rhs: T
#     ) -> Tensor[T]:
#         # Usage: c = a * 3  # multiply by scalar
#         return mul_t(self, _scalar_of(rhs))

#     @always_inline
#     fn __truediv__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], rhs: Tensor[T]
#     ) -> Tensor[T]:
#         # Usage: c = a / b  # elementwise division
#         return div_t(self, rhs)

#     @always_inline
#     fn __truediv__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], rhs: T
#     ) -> Tensor[T]:
#         # Usage: c = a / 3  # divide by scalar
#         return div_t(self, _scalar_of(rhs))

#     @always_inline
#     fn __mod__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], rhs: Tensor[T]
#     ) -> Tensor[T]:
#         # Usage: c = a % b  # elementwise modulo
#         return mod_t(self, rhs)

#     @always_inline
#     fn __mod__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], rhs: T
#     ) -> Tensor[T]:
#         # Usage: c = a % 3  # modulo with scalar
#         return mod_t(self, _scalar_of(rhs))


#     # ---------- Reflected arithmetic (scalar op on left) ----------

#     @always_inline
#     fn __radd__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], lhs: T
#     ) -> Tensor[T]:
#         # Usage: c = 3 + a  # scalar on the left
#         return add_t(_scalar_of(lhs), self)

#     @always_inline
#     fn __rsub__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], lhs: T
#     ) -> Tensor[T]:
#         # Usage: c = 3 - a  # scalar minus tensor
#         return sub_t(_scalar_of(lhs), self)

#     @always_inline
#     fn __rmul__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], lhs: T
#     ) -> Tensor[T]:
#         # Usage: c = 3 * a  # scalar times tensor
#         return mul_t(_scalar_of(lhs), self)

#     @always_inline
#     fn __rtruediv__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], lhs: T
#     ) -> Tensor[T]:
#         # Usage: c = 3 / a  # scalar divided by tensor
#         return div_t(_scalar_of(lhs), self)

#     @always_inline
#     fn __rmod__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], lhs: T
#     ) -> Tensor[T]:
#         # Usage: c = 3 % a  # scalar modulo tensor
#         return mod_t(_scalar_of(lhs), self)


#     # ---------- In-place arithmetic ----------

#     @always_inline
#     fn __iadd__[T: ImplicitlyCopyable & Copyable & Movable](
#         mut self: Tensor[T], rhs: Tensor[T]
#     ) -> Tensor[T]:
#         # Usage: a += b
#         self = add_t(self, rhs)
#         return self

#     @always_inline
#     fn __iadd__[T: ImplicitlyCopyable & Copyable & Movable](
#         mut self: Tensor[T], rhs: T
#     ) -> Tensor[T]:
#         # Usage: a += 3
#         self = add_t(self, _scalar_of(rhs))
#         return self

#     @always_inline
#     fn __isub__[T: ImplicitlyCopyable & Copyable & Movable](
#         mut self: Tensor[T], rhs: Tensor[T]
#     ) -> Tensor[T]:
#         # Usage: a -= b
#         self = sub_t(self, rhs)
#         return self

#     @always_inline
#     fn __isub__[T: ImplicitlyCopyable & Copyable & Movable](
#         mut self: Tensor[T], rhs: T
#     ) -> Tensor[T]:
#         # Usage: a -= 3
#         self = sub_t(self, _scalar_of(rhs))
#         return self

#     @always_inline
#     fn __imul__[T: ImplicitlyCopyable & Copyable & Movable](
#         mut self: Tensor[T], rhs: Tensor[T]
#     ) -> Tensor[T]:
#         # Usage: a *= b
#         self = mul_t(self, rhs)
#         return self

#     @always_inline
#     fn __imul__[T: ImplicitlyCopyable & Copyable & Movable](
#         mut self: Tensor[T], rhs: T
#     ) -> Tensor[T]:
#         # Usage: a *= 3
#         self = mul_t(self, _scalar_of(rhs))
#         return self

#     @always_inline
#     fn __itruediv__[T: ImplicitlyCopyable & Copyable & Movable](
#         mut self: Tensor[T], rhs: Tensor[T]
#     ) -> Tensor[T]:
#         # Usage: a /= b
#         self = div_t(self, rhs)
#         return self

#     @always_inline
#     fn __itruediv__[T: ImplicitlyCopyable & Copyable & Movable](
#         mut self: Tensor[T], rhs: T
#     ) -> Tensor[T]:
#         # Usage: a /= 3
#         self = div_t(self, _scalar_of(rhs))
#         return self

#     @always_inline
#     fn __imod__[T: ImplicitlyCopyable & Copyable & Movable](
#         mut self: Tensor[T], rhs: Tensor[T]
#     ) -> Tensor[T]:
#         # Usage: a %= b
#         self = mod_t(self, rhs)
#         return self

#     @always_inline
#     fn __imod__[T: ImplicitlyCopyable & Copyable & Movable](
#         mut self: Tensor[T], rhs: T
#     ) -> Tensor[T]:
#         # Usage: a %= 3
#         self = mod_t(self, _scalar_of(rhs))
#         return self


#     # ---------- Power ----------

#     @always_inline
#     fn __pow__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], rhs: Tensor[T]
#     ) -> Tensor[T]:
#         # Usage: c = a ** b  # elementwise power
#         return pow_t(self, rhs)

#     @always_inline
#     fn __pow__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], rhs: T
#     ) -> Tensor[T]:
#         # Usage: c = a ** 2  # power with scalar exponent
#         return pow_t(self, _scalar_of(rhs))

#     @always_inline
#     fn __rpow__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], lhs: T
#     ) -> Tensor[T]:
#         # Usage: c = 2 ** a  # scalar base, tensor exponent
#         return pow_t(_scalar_of(lhs), self)

#     @always_inline
#     fn __ipow__[T: ImplicitlyCopyable & Copyable & Movable](
#         mut self: Tensor[T], rhs: Tensor[T]
#     ) -> Tensor[T]:
#         # Usage: a **= b
#         self = pow_t(self, rhs)
#         return self

#     @always_inline
#     fn __ipow__[T: ImplicitlyCopyable & Copyable & Movable](
#         mut self: Tensor[T], rhs: T
#     ) -> Tensor[T]:
#         # Usage: a **= 2
#         self = pow_t(self, _scalar_of(rhs))
#         return self


#     # ---------- Unary ----------

#     @always_inline
#     fn __neg__[T: ImplicitlyCopyable & Copyable & Movable](self: Tensor[T]) -> Tensor[T]:
#         # Usage: b = -a
#         return neg_t(self)

#     @always_inline
#     fn __pos__[T: ImplicitlyCopyable & Copyable & Movable](self: Tensor[T]) -> Tensor[T]:
#         # Usage: b = +a  # no-op sign
#         return pos_t(self)

#     @always_inline
#     fn __abs__[T: ImplicitlyCopyable & Copyable & Movable](self: Tensor[T]) -> Tensor[T]:
#         # Usage: b = abs(a)
#         return abs_t(self)


#     # ---------- Logical (project-style masks for all T) ----------

#     @always_inline
#     fn __and__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], rhs: Tensor[T]
#     ) -> Tensor[T]:
#         # Usage: m = a & b  # logical-and via project convention
#         return and_t(self, rhs)

#     @always_inline
#     fn __and__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], rhs: T
#     ) -> Tensor[T]:
#         # Usage: m = a & 1  # logical-and with scalar
#         return and_t(self, _scalar_of(rhs))

#     @always_inline
#     fn __or__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], rhs: Tensor[T]
#     ) -> Tensor[T]:
#         # Usage: m = a | b  # logical-or via project convention
#         return or_t(self, rhs)

#     @always_inline
#     fn __or__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], rhs: T
#     ) -> Tensor[T]:
#         # Usage: m = a | 1  # logical-or with scalar
#         return or_t(self, _scalar_of(rhs))

#     @always_inline
#     fn __xor__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], rhs: Tensor[T]
#     ) -> Tensor[T]:
#         # Usage: m = a ^ b  # logical-xor via project convention
#         return xor_t(self, rhs)

#     @always_inline
#     fn __xor__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], rhs: T
#     ) -> Tensor[T]:
#         # Usage: m = a ^ 1  # logical-xor with scalar
#         return xor_t(self, _scalar_of(rhs))

#     @always_inline
#     fn __invert__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T]
#     ) -> Tensor[Int]:
#         # Usage: m = ~a  # logical-not → Int mask {0,1}
#         return not_t(self)


#     # ---------- Reflected logical ----------

#     @always_inline
#     fn __rand__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], lhs: T
#     ) -> Tensor[T]:
#         # Usage: m = 1 & a  # scalar on the left
#         return and_t(_scalar_of(lhs), self)

#     @always_inline
#     fn __ror__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], lhs: T
#     ) -> Tensor[T]:
#         # Usage: m = 1 | a
#         return or_t(_scalar_of(lhs), self)

#     @always_inline
#     fn __rxor__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], lhs: T
#     ) -> Tensor[T]:
#         # Usage: m = 1 ^ a
#         return xor_t(_scalar_of(lhs), self)


#     # ---------- Comparisons → mask(Int) ----------

#     @always_inline
#     fn __lt__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], rhs: Tensor[T]
#     ) -> Tensor[Int]:
#         # Usage: m = a < b  # returns Int mask {0,1}
#         return lt_t(self, rhs)

#     @always_inline
#     fn __lt__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], rhs: T
#     ) -> Tensor[Int]:
#         # Usage: m = a < 3
#         return lt_t(self, _scalar_of(rhs))

#     @always_inline
#     fn __le__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], rhs: Tensor[T]
#     ) -> Tensor[Int]:
#         # Usage: m = a <= b
#         return le_t(self, rhs)

#     @always_inline
#     fn __le__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], rhs: T
#     ) -> Tensor[Int]:
#         # Usage: m = a <= 3
#         return le_t(self, _scalar_of(rhs))

#     @always_inline
#     fn __gt__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], rhs: Tensor[T]
#     ) -> Tensor[Int]:
#         # Usage: m = a > b
#         return gt_t(self, rhs)

#     @always_inline
#     fn __gt__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], rhs: T
#     ) -> Tensor[Int]:
#         # Usage: m = a > 3
#         return gt_t(self, _scalar_of(rhs))

#     @always_inline
#     fn __ge__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], rhs: Tensor[T]
#     ) -> Tensor[Int]:
#         # Usage: m = a >= b
#         return ge_t(self, rhs)

#     @always_inline
#     fn __ge__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], rhs: T
#     ) -> Tensor[Int]:
#         # Usage: m = a >= 3
#         return ge_t(self, _scalar_of(rhs))

#     @always_inline
#     fn __eq__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], rhs: Tensor[T]
#     ) -> Tensor[Int]:
#         # Usage: m = a == b
#         return eq_t(self, rhs)

#     @always_inline
#     fn __eq__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], rhs: T
#     ) -> Tensor[Int]:
#         # Usage: m = a == 3
#         return eq_t(self, _scalar_of(rhs))

#     @always_inline
#     fn __ne__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], rhs: Tensor[T]
#     ) -> Tensor[Int]:
#         # Usage: m = a != b
#         return ne_t(self, rhs)

#     @always_inline
#     fn __ne__[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], rhs: T
#     ) -> Tensor[Int]:
#         # Usage: m = a != 3
#         return ne_t(self, _scalar_of(rhs))


#     # =========================
#     # Generic methods (T)
#     # =========================

#     # ---------- Arithmetic (scalar) ----------

#     @always_inline
#     fn add_scalar[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], s: T
#     ) -> Tensor[T]:
#         # Usage: y = x.add_scalar(3)
#         return add_t(self, _scalar_of(s))

#     @always_inline
#     fn sub_scalar[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], s: T
#     ) -> Tensor[T]:
#         # Usage: y = x.sub_scalar(3)
#         return sub_t(self, _scalar_of(s))

#     @always_inline
#     fn mul_scalar[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], s: T
#     ) -> Tensor[T]:
#         # Usage: y = x.mul_scalar(3)
#         return mul_t(self, _scalar_of(s))

#     @always_inline
#     fn div_scalar[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], s: T
#     ) -> Tensor[T]:
#         # Usage: y = x.div_scalar(3)
#         return div_t(self, _scalar_of(s))

#     @always_inline
#     fn mod_scalar[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], s: T
#     ) -> Tensor[T]:
#         # Usage: y = x.mod_scalar(3)
#         return mod_t(self, _scalar_of(s))

#     @always_inline
#     fn pow_scalar[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], p: T
#     ) -> Tensor[T]:
#         # Usage: y = x.pow_scalar(2)
#         return pow_t(self, _scalar_of(p))


#     # ---------- Arithmetic (tensor) ----------

#     @always_inline
#     fn add[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], other: Tensor[T]
#     ) -> Tensor[T]:
#         # Usage: y = x.add(other)
#         return add_t(self, other)

#     @always_inline
#     fn sub[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], other: Tensor[T]
#     ) -> Tensor[T]:
#         # Usage: y = x.sub(other)
#         return sub_t(self, other)

#     @always_inline
#     fn mul[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], other: Tensor[T]
#     ) -> Tensor[T]:
#         # Usage: y = x.mul(other)
#         return mul_t(self, other)

#     @always_inline
#     fn divide[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], other: Tensor[T]
#     ) -> Tensor[T]:
#         # Usage: y = x.divide(other)
#         return div_t(self, other)

#     @always_inline
#     fn mod[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], other: Tensor[T]
#     ) -> Tensor[T]:
#         # Usage: y = x.mod(other)
#         return mod_t(self, other)

#     @always_inline
#     fn pow[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], other: Tensor[T]
#     ) -> Tensor[T]:
#         # Usage: y = x.pow(other)
#         return pow_t(self, other)


#     # ---------- Logical (project-style) ----------

#     @always_inline
#     fn and_scalar[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], s: T
#     ) -> Tensor[T]:
#         # Usage: m = x.and_scalar(1)
#         return and_t(self, _scalar_of(s))

#     @always_inline
#     fn or_scalar[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], s: T
#     ) -> Tensor[T]:
#         # Usage: m = x.or_scalar(1)
#         return or_t(self, _scalar_of(s))

#     @always_inline
#     fn xor_scalar[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], s: T
#     ) -> Tensor[T]:
#         # Usage: m = x.xor_scalar(1)
#         return xor_t(self, _scalar_of(s))

#     @always_inline
#     fn logical_not[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T]
#     ) -> Tensor[Int]:
#         # Usage: m = x.logical_not()  # Int mask
#         return not_t(self)

#     @always_inline
#     fn logical_and[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], other: Tensor[T]
#     ) -> Tensor[T]:
#         # Usage: m = x.logical_and(other)
#         return and_t(self, other)

#     @always_inline
#     fn logical_or[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], other: Tensor[T]
#     ) -> Tensor[T]:
#         # Usage: m = x.logical_or(other)
#         return or_t(self, other)

#     @always_inline
#     fn logical_xor[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], other: Tensor[T]
#     ) -> Tensor[T]:
#         # Usage: m = x.logical_xor(other)
#         return xor_t(self, other)


#     # ---------- Comparisons (→ Int mask) ----------

#     @always_inline
#     fn lt_scalar[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], s: T
#     ) -> Tensor[Int]:
#         # Usage: m = x.lt_scalar(3)
#         return lt_t(self, _scalar_of(s))

#     @always_inline
#     fn le_scalar[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], s: T
#     ) -> Tensor[Int]:
#         # Usage: m = x.le_scalar(3)
#         return le_t(self, _scalar_of(s))

#     @always_inline
#     fn gt_scalar[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], s: T
#     ) -> Tensor[Int]:
#         # Usage: m = x.gt_scalar(3)
#         return gt_t(self, _scalar_of(s))

#     @always_inline
#     fn ge_scalar[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], s: T
#     ) -> Tensor[Int]:
#         # Usage: m = x.ge_scalar(3)
#         return ge_t(self, _scalar_of(s))

#     @always_inline
#     fn eq_scalar[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], s: T
#     ) -> Tensor[Int]:
#         # Usage: m = x.eq_scalar(3)
#         return eq_t(self, _scalar_of(s))

#     @always_inline
#     fn ne_scalar[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], s: T
#     ) -> Tensor[Int]:
#         # Usage: m = x.ne_scalar(3)
#         return ne_t(self, _scalar_of(s))

#     @always_inline
#     fn lt[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], other: Tensor[T]
#     ) -> Tensor[Int]:
#         # Usage: m = x.lt(other)
#         return lt_t(self, other)

#     @always_inline
#     fn le[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], other: Tensor[T]
#     ) -> Tensor[Int]:
#         # Usage: m = x.le(other)
#         return le_t(self, other)

#     @always_inline
#     fn gt[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], other: Tensor[T]
#     ) -> Tensor[Int]:
#         # Usage: m = x.gt(other)
#         return gt_t(self, other)

#     @always_inline
#     fn ge[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], other: Tensor[T]
#     ) -> Tensor[Int]:
#         # Usage: m = x.ge(other)
#         return ge_t(self, other)

#     @always_inline
#     fn eq[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], other: Tensor[T]
#     ) -> Tensor[Int]:
#         # Usage: m = x.eq(other)
#         return eq_t(self, other)

#     @always_inline
#     fn ne[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], other: Tensor[T]
#     ) -> Tensor[Int]:
#         # Usage: m = x.ne(other)
#         return ne_t(self, other)

#     @always_inline
#     fn where_mask[T: ImplicitlyCopyable & Copyable & Movable](
#         self: Tensor[T], mask: Tensor[Int], other: Tensor[T]
#     ) -> Tensor[T]:
#         # Usage: y = x.where_mask(mask, other)  # select(mask, x, other)
#         return where(mask, self, other)


#     # =========================
#     # Int-only true bitwise + shifts (override generic logical)
#     # =========================

#     @always_inline
#     fn __and__(self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]:
#         # Usage: m = ai & bi  # true bitwise AND
#         return band_t(self, rhs)

#     @always_inline
#     fn __and__(self: Tensor[Int], rhs: Int) -> Tensor[Int]:
#         # Usage: m = ai & 1  # bitwise AND with scalar
#         return band_t(self, scalar_int(rhs))

#     @always_inline
#     fn __or__(self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]:
#         # Usage: m = ai | bi  # true bitwise OR
#         return bor_t(self, rhs)

#     @always_inline
#     fn __or__(self: Tensor[Int], rhs: Int) -> Tensor[Int]:
#         # Usage: m = ai | 1  # bitwise OR with scalar
#         return bor_t(self, scalar_int(rhs))

#     @always_inline
#     fn __xor__(self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]:
#         # Usage: m = ai ^ bi  # true bitwise XOR
#         return bxor_t(self, rhs)

#     @always_inline
#     fn __xor__(self: Tensor[Int], rhs: Int) -> Tensor[Int]:
#         # Usage: m = ai ^ 1  # bitwise XOR with scalar
#         return bxor_t(self, scalar_int(rhs))

#     @always_inline
#     fn __invert__(self: Tensor[Int]) -> Tensor[Int]:
#         # Usage: m = ~ai  # bitwise NOT
#         return bnot_t(self)

#     @always_inline
#     fn __iand__(mut self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]:
#         # Usage: ai &= bi
#         self = band_t(self, rhs)
#         return self

#     @always_inline
#     fn __ior__(mut self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]:
#         # Usage: ai |= bi
#         self = bor_t(self, rhs)
#         return self

#     @always_inline
#     fn __ixor__(mut self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]:
#         # Usage: ai ^= bi
#         self = bxor_t(self, rhs)
#         return self

#     @always_inline
#     fn __lshift__(self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]:
#         # Usage: y = ai << bi
#         return shl_t(self, rhs)

#     @always_inline
#     fn __lshift__(self: Tensor[Int], rhs: Int) -> Tensor[Int]:
#         # Usage: y = ai << 3
#         return shl_t(self, scalar_int(rhs))

#     @always_inline
#     fn __rshift__(self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]:
#         # Usage: y = ai >> bi
#         return shr_t(self, rhs)

#     @always_inline
#     fn __rshift__(self: Tensor[Int], rhs: Int) -> Tensor[Int]:
#         # Usage: y = ai >> 3
#         return shr_t(self, scalar_int(rhs))

#     @always_inline
#     fn __ilshift__(mut self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]:
#         # Usage: ai <<= bi
#         self = shl_t(self, rhs)
#         return self

#     @always_inline
#     fn __ilshift__(mut self: Tensor[Int], rhs: Int) -> Tensor[Int]:
#         # Usage: ai <<= 3
#         self = shl_t(self, scalar_int(rhs))
#         return self

#     @always_inline
#     fn __irshift__(mut self: Tensor[Int], rhs: Tensor[Int]) -> Tensor[Int]:
#         # Usage: ai >>= bi
#         self = shr_t(self, rhs)
#         return self

#     @always_inline
#     fn __irshift__(mut self: Tensor[Int], rhs: Int) -> Tensor[Int]:
#         # Usage: ai >>= 3
#         self = shr_t(self, scalar_int(rhs))
#         return self


    
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
