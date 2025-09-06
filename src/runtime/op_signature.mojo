# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.runtime
# File: src/momijo/runtime/op_signature.mojo

# OpSignature: runtime representation of an operator's signature (name, inputs, outputs).

@fieldwise_init
struct OpSignature:
    var name: String
    var input_count: Int
    var output_count: Int

    fn __init__(out self, name: String, input_count: Int, output_count: Int):
        self.name = name
        self.input_count = input_count
        self.output_count = output_count

    fn summary(self) -> String:
        var s = String("OpSignature(") + self.name
        s = s + String(", in=") + String(self.input_count)
        s = s + String(", out=") + String(self.output_count) + String(")")
        return s


fn make_op_signature(name: String, inputs: Int, outputs: Int) -> OpSignature:
    return OpSignature(name, inputs, outputs)


fn _self_test() -> Bool:
    var sig = make_op_signature(String("matmul"), 2, 1)
    var ok = True
    if sig.input_count != 2 or sig.output_count != 1:
        ok = False
    if len(sig.summary()) == 0:
        ok = False
    return ok
