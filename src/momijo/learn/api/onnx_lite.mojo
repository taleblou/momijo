# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.api.onnx_lite
# File:         src/momijo/learn/api/onnx_lite.mojo
#
# Description:
#   Minimal, dependency-free "ONNX-lite" JSON exporter (no file IO).
#   - Returns JSON as String.
#   - Extensible via ad-hoc polymorphism: define
#       emit_onnx_lite(your_type, in_name) -> (out_name: String, node_json: String).
#   - Built-in emitters: Linear, ReLU.
#   - Unknown types fall back to no-op (Identity only).

from collections.list import List
from momijo.tensor import tensor

# keep imports minimal to avoid missing modules
from momijo.learn.nn.layers import Linear
from momijo.learn.nn.layers import ReLU

# --------------------------- helpers ------------------------------------------

fn _json_int_list(xs: List[Int]) -> String:
    var s = String("[")
    var n = len(xs)
    var i = 0
    while i < n:
        s = s + String(xs[i])
        if i + 1 < n:
            s = s + ","
        i = i + 1
    s = s + "]"
    return s

fn _json_tensor_flat(x: tensor.Tensor[Float32]) -> String:
    # Emits a flat JSON array from tensor's underlying storage order.
    var s = String("[")
    var n = x.numel()
    var i = 0
    while i < n:
        s = s + String(x._data[i])
        if i + 1 < n:
            s = s + ","
        i = i + 1
    s = s + "]"
    return s

# --------------------------- Public API ---------------------------------------

# Overload 1: no path (returns JSON)
fn export_onnx_lite[M: Copyable & Movable](
    model: M,
    input_shape: List[Int],
    input_name: String = "x",
    output_name: String = "y",
    opset: Int = 17
) -> String:
    var json = String("{\n")
    json = json + "  \"opset\": " + String(opset) + ",\n"
    json = json + "  \"inputs\": [{\"name\":\"" + input_name + "\", \"shape\": " + _json_int_list(input_shape) + "}],\n"
    json = json + "  \"outputs\": [{\"name\":\"" + output_name + "\"}],\n"
    json = json + "  \"nodes\": [\n"

    var nodes = String("")

    # Delegate to type-specific emitter (if available); else fallback no-op
    var out_name: String
    var frag: String
    (out_name, frag) = emit_onnx_lite(model, input_name)
    if frag != "":
        nodes = nodes + frag

    # If last internal name differs from desired output name, add an Identity hop.
    if out_name != output_name:
        if nodes != "":
            nodes = nodes + ",\n"
        nodes = nodes + "    {\"op\":\"Identity\",\"name\":\"identity_nop\",\"in\":\"" + out_name + "\",\"out\":\"" + output_name + "\"}"

    json = json + nodes + "\n"
    json = json + "  ]\n"
    json = json + "}\n"
    return json

# Overload 2: with path (kept only to match a 6-arg call-site; path is ignored)
fn export_onnx_lite[M: Copyable & Movable](
    model: M,
    input_shape: List[Int],
    path: String,                      # ignored on purpose (no-IO version)
    input_name: String = "x",
    output_name: String = "y",
    opset: Int = 17
) -> String:
    return export_onnx_lite(model, input_shape, input_name, output_name, opset)

# --------------------------- Emitters -----------------------------------------
# Each returns (out_name, node_json_fragment).

# Default fallback for unknown types:
fn emit_onnx_lite[T: Copyable & Movable](model: T, in_name: String) -> (String, String):
    # No nodes emitted; pass-through. An Identity will be added
    return (in_name, String(""))

# Linear
fn emit_onnx_lite(layer: Linear, in_name: String) -> (String, String):
    var out_name = in_name + "_linear"
    var frag = String("")
    frag = frag + "    {\"op\":\"Linear\",\"name\":\"linear\",\"in\":\"" + in_name + "\",\"out\":\"" + out_name + "\",\n"
    frag = frag + "     \"W\": " + _json_tensor_flat(layer.weight) + ",\n"
    frag = frag + "     \"b\": " + _json_tensor_flat(layer.bias_t) + "}"
    return (out_name, frag)

# ReLU
fn emit_onnx_lite(layer: ReLU, in_name: String) -> (String, String):
    var out_name = in_name + "_relu"
    var frag = "    {\"op\":\"ReLU\",\"name\":\"relu\",\"in\":\"" + in_name + "\",\"out\":\"" + out_name + "\"}"
    return (out_name, frag)
