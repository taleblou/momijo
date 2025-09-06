# ============================================================================
# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.io.checkpoints
# File: src/momijo/io/checkpoints/weights.mojo
# ============================================================================

from momijo.tensor.tensor import Tensor

# -----------------------------------------------------------------------------
# ParamRef: simple wrapper for a named tensor
# -----------------------------------------------------------------------------
struct ParamRef:
    var name: String
    var tensor: Tensor

    fn __init__(out self, name: String, tensor: Tensor):
        self.name = name
        self.tensor = tensor


# -----------------------------------------------------------------------------
# StateDict: dictionary of parameter name -> tensor
# -----------------------------------------------------------------------------
struct StateDict:
    var params: Dict[String, Tensor]

    fn __init__(out self):
        self.params = Dict[String, Tensor]()

    # Set parameter
    fn __setitem__(mut self, key: String, value: Tensor):
        self.params[key] = value

    # Get parameter
    fn __getitem__(self, key: String) -> Tensor:
        return self.params[key]

    # Check if parameter exists
    fn contains(self, key: String) -> Bool:
        return self.params.contains(key)

    # Length
    fn __len__(self) -> Int:
        return len(self.params)

    # Return keys
    fn keys(self) -> List[String]:
        var out_keys = List[String]()
        for k in self.params.keys():
            out_keys.append(k)
        return out_keys

    # Return items (list of (String, Tensor))
    fn items(self) -> List[(String, Tensor)]:
        var out_items = List[(String, Tensor)]()
        for (k, v) in self.params.items():
            out_items.append((k, v))
        return out_items

    # Convert to JSON-serializable metadata (for saving)
    fn to_meta(self) -> List[Dict[String, Any]]:
        var result = List[Dict[String, Any]]()
        for (k, v) in self.params.items():
            var entry = {
                "name": k,
                "dtype": v.dtype_name(),
                "shape": v.shape_as_list(),
                "nbytes": v.nbytes()
            }
            result.append(entry)
        return result

    # Construct from another dict (helper)
    @staticmethod
    fn from_dict(d: Dict[String, Tensor]) -> StateDict:
        var st = StateDict()
        for (k, v) in d.items():
            st[k] = v
        return st


# -----------------------------------------------------------------------------
# Minimal self-test
# -----------------------------------------------------------------------------

fn _self_test() -> Bool:
    var ok = True
    var sd = StateDict()
    var t1 = Tensor.ones([2,2], dtype="f32")
    var t2 = Tensor.zeros([3], dtype="f32")

    sd["w1"] = t1
    sd["b1"] = t2

    ok = ok and sd.contains("w1")
    ok = ok and sd.contains("b1")
    ok = ok and len(sd) == 2
    ok = ok and sd["w1"].shape_as_list() == [2,2]
    ok = ok and sd["b1"].shape_as_list() == [3]

    var meta = sd.to_meta()
    ok = ok and len(meta) == 2

    return ok


fn main():
    if _self_test():
        print("Weights module self-test: OK")
    else:
        print("Weights module self-test: FAIL")
