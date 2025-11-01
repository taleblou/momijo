# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.learn
# Module:       learn.utils.summary
# File:         src/momijo/learn/utils/summary.mojo
#
# Description:
#   Lightweight, backend-agnostic model summarizer using a builder pattern.
#   Models implement a `summarize(self, s: Pointer[Summarizer])` method and
#   record their layers/blocks into the summarizer. This avoids reflection and
#   keeps compile-time friendliness in Mojo.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from collections.list import List
from momijo.tensor import tensor

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

@always_inline
fn _numel(shape: List[Int]) -> Int:
    # Returns the product of shape dimensions.
    var p = 1
    var i = 0
    var n = len(shape)
    while i < n:
        p = p * shape[i]
        i = i + 1
    return p

@always_inline
fn _shape_str(shape: List[Int]) -> String:
    # Pretty string for a shape list, e.g., [B, C, H, W]
    var s = "["
    var i = 0
    var n = len(shape)
    while i < n:
        s = s + String(shape[i])
        if i + 1 < n:
            s = s + ", "
        i = i + 1
    s = s + "]"
    return s

@always_inline
fn _pad_right(s: String, width: Int) -> String:
    # Pads string s on the right with spaces to the given width.
    var out = s
    var k = width - Int(len(out))
    var i = 0
    while i < k:
        out = out + " "
        i = i + 1
    return out

@always_inline
fn _fmt_int(x: Int) -> String:
    # Converts an integer to string (kept separate for future locale formatting).
    return String(x)

# Generic numel for tensors (dtype-agnostic)
@always_inline
fn _tensor_numel[T: Copyable & Movable & ImplicitlyCopyable](t: tensor.Tensor[T]) -> Int:
    return _numel(t.shape())

# -----------------------------------------------------------------------------
# Records
# -----------------------------------------------------------------------------

struct LayerRecord:
    # One row in the summary table.
    var name: String
    var in_shape: List[Int]
    var out_shape: List[Int]
    var params_count: Int
    var buffers_count: Int
    var trainable_params: Int

    fn __init__(out self):
        self.name = ""
        self.in_shape = List[Int]()
        self.out_shape = List[Int]()
        self.params_count = 0
        self.buffers_count = 0
        self.trainable_params = 0

# -----------------------------------------------------------------------------
# Summarizer (builder)
# -----------------------------------------------------------------------------

struct Summarizer:
    # Collects layer records and renders a formatted summary.
    var title: String
    var rows: List[LayerRecord]
    var total_params: Int
    var total_buffers: Int
    var total_trainable: Int

    fn __init__(out self, title: String = "Model Summary"):
        self.title = title
        self.rows = List[LayerRecord]()
        self.total_params = 0
        self.total_buffers = 0
        self.total_trainable = 0

    fn add_counts(
        mut self,
        name: String,
        in_shape: List[Int],
        out_shape: List[Int],
        params_count: Int,
        buffers_count: Int,
        trainable_params: Int
    ) -> None:
        # Adds a record using raw parameter/buffer counts.
        var r = LayerRecord()
        r.name = name
        r.in_shape = in_shape
        r.out_shape = out_shape
        r.params_count = params_count
        r.buffers_count = buffers_count
        r.trainable_params = trainable_params
        self.rows.append(r)

        self.total_params = self.total_params + r.params_count
        self.total_buffers = self.total_buffers + r.buffers_count
        self.total_trainable = self.total_trainable + r.trainable_params

    fn add_leaf(mut self, name: String, in_shape: List[Int], out_shape: List[Int]) -> None:
        # Adds a parameterless operation (e.g., activation, reshape).
        self.add_counts(name, in_shape, out_shape, 0, 0, 0)

    # Generic overload: add using parameter tensors of any dtype.
    fn add_params_generic[T: Copyable & Movable & ImplicitlyCopyable](
        mut self,
        name: String,
        in_shape: List[Int],
        out_shape: List[Int],
        params: List[tensor.Tensor[T]],
        buffers: List[tensor.Tensor[T]],
        trainable: Bool
    ) -> None:
        var pc = 0
        var i = 0
        var np = len(params)
        while i < np:
            pc = pc + _tensor_numel(params[i])
            i = i + 1

        var bc = 0
        var j = 0
        var nb = len(buffers)
        while j < nb:
            bc = bc + _tensor_numel(buffers[j])
            j = j + 1

        var tc = pc
        if not trainable:
            tc = 0

        self.add_counts(name, in_shape, out_shape, pc, bc, tc)

    # Convenience: Float64 parameter tensors with explicit buffers list.
    fn add_params_f64(
        mut self,
        name: String,
        in_shape: List[Int],
        out_shape: List[Int],
        params: List[tensor.Tensor[Float64]],
        buffers: List[tensor.Tensor[Float64]],
        trainable: Bool
    ) -> None:
        self.add_params_generic[Float64](name, in_shape, out_shape, params, buffers, trainable)

    # Convenience: UInt8 parameter/buffer tensors (e.g., quantized or byte buffers).
    fn add_params_u8(
        mut self,
        name: String,
        in_shape: List[Int],
        out_shape: List[Int],
        params: List[tensor.Tensor[UInt8]],
        buffers: List[tensor.Tensor[UInt8]],
        trainable: Bool
    ) -> None:
        self.add_params_generic[UInt8](name, in_shape, out_shape, params, buffers, trainable)

    fn render(self) -> String:
        # Renders a pretty table with fixed-width columns.
        var w_idx  = 5
        var w_name = 26
        var w_io   = 28
        var w_p    = 12
        var w_b    = 10

        var sep = String("-")
        var i = 0
        while i < (w_idx + 1 + w_name + 1 + w_io + 1 + w_p + 1 + w_b):
            sep = sep + "-"
            i = i + 1

        var s = self.title + "\n" + sep + "\n"
        s = s + _pad_right("#", w_idx) + " "
        s = s + _pad_right("Name", w_name) + " "
        s = s + _pad_right("In -> Out", w_io) + " "
        s = s + _pad_right("Params", w_p) + " "
        s = s + _pad_right("Buffers", w_b) + "\n"
        s = s + sep + "\n"

        var row_idx = 0
        var n = len(self.rows)
        while row_idx < n:
            var r = self.rows[row_idx]
            var io = _shape_str(r.in_shape) + " -> " + _shape_str(r.out_shape)
            s = s + _pad_right(_fmt_int(row_idx), w_idx) + " "
            s = s + _pad_right(r.name, w_name) + " "
            s = s + _pad_right(io, w_io) + " "
            s = s + _pad_right(_fmt_int(r.params_count), w_p) + " "
            s = s + _pad_right(_fmt_int(r.buffers_count), w_b) + "\n"
            row_idx = row_idx + 1

        s = s + sep + "\n"
        s = s + "Total params: " + String(self.total_params)
        s = s + " | Trainable: " + String(self.total_trainable)
        s = s + " | Buffers: " + String(self.total_buffers) + "\n"
        return s

# -----------------------------------------------------------------------------
# Public API (duck-typed)
# -----------------------------------------------------------------------------

fn model_summary(model) -> String:
    # Creates a summarizer and asks the model to populate it via
    # `summarize(self, s: Pointer[Summarizer])`. The model is responsible for
    # calling the relevant add_* methods to describe its structure.
    var s = Summarizer("Model Summary")
    model.summarize(&s)
    return s.render()



# ----------------------------- Min/Max Observer -------------------------------
struct MinMaxObserver(Copyable, Movable):
    var min_val: Float64
    var max_val: Float64
    var initialized: Bool

    fn __init__(out self):
        self.min_val = 0.0
        self.max_val = 0.0
        self.initialized = False

    fn observe(mut self, x: tensor.Tensor[Float64]):
        var n = x._n
        if n <= 0: return
        var i = 0
        if not self.initialized:
            self.min_val = x._data[0]
            self.max_val = x._data[0]
            self.initialized = True
            i = 1
        while i < n:
            var v = x._data[i]
            if v < self.min_val: self.min_val = v
            if v > self.max_val: self.max_val = v
            i = i + 1

    fn scale_symmetric(self, qmax: Float64 = 127.0) -> Float64:
        var a = self.min_val
        var b = self.max_val
        var m = a
        if b > m: m = b
        if -a > m: m = -a
        if m <= 0.0: return 1.0
        return m / qmax