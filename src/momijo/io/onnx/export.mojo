# Project:      Momijo
# Module:       src.momijo.io.onnx.export
# File:         export.mojo
# Path:         src/momijo/io/onnx/export.mojo
#
# Description:  Filesystem/IO helpers with Path-centric APIs and safe resource
#               management (binary/text modes and encoding clarity).
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
#   - Key functions: _tensor_to_onnx, _node_to_onnx, _make_graph, export_onnx
#   - Uses generic functions/types with explicit trait bounds.


from momijo.tensor.tensor import Tensor
from onnx import TensorProto, helper
import onnx

fn _tensor_to_onnx(t: Tensor, name: String) -> onnx.TensorProto:
    var arr = t.to_numpy()
    var proto = onnx.helper.make_tensor(
        name=name,
        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[arr.dtype],
        dims=arr.shape,
        vals=arr.flatten().tolist()
    )
    return proto
fn _node_to_onnx(op_type: String, inputs: List[String], outputs: List[String], attrs: Dict[String, Any] = {}) -> onnx.NodeProto:
    var attr_list = List[onnx.AttributeProto]()
    for (k,v) in attrs.items():
        attr_list.append(onnx.helper.make_attribute(k,v))
    return onnx.helper.make_node(op_type, inputs, outputs, **attrs)
fn _make_graph(name: String, nodes: List[onnx.NodeProto], inputs: List[onnx.ValueInfoProto],
               outputs: List[onnx.ValueInfoProto], initializers: List[onnx.TensorProto]) -> onnx.GraphProto:
    return onnx.helper.make_graph(nodes, name, inputs, outputs, initializer=initializers)

# -----------------------------------------------------------------------------
# Export function
# -----------------------------------------------------------------------------
fn export_onnx(model: Any, path: String) -> None:
    # Expecting model with fields: inputs, outputs, nodes, params
    var nodes = List[onnx.NodeProto]()
    var inputs = List[onnx.ValueInfoProto]()
    var outputs = List[onnx.ValueInfoProto]()
    var initializers = List[onnx.TensorProto]()

    # Convert inputs
    for inp in model.inputs:
        inputs.append(helper.make_tensor_value_info(inp.name, TensorProto.FLOAT, inp.shape))

    # Convert outputs
    for out in model.outputs:
        outputs.append(helper.make_tensor_value_info(out.name, TensorProto.FLOAT, out.shape))

    # Convert params
    for (name,tensor) in model.params.items():
        initializers.append(_tensor_to_onnx(tensor, name))

    # Convert nodes
    for n in model.nodes:
        nodes.append(_node_to_onnx(n.op, n.inputs, n.outputs, n.attrs))

    var graph = _make_graph("MomijoGraph", nodes, inputs, outputs, initializers)
    var model_proto = helper.make_model(graph, producer_name="momijo")
    onnx.save(model_proto, path)