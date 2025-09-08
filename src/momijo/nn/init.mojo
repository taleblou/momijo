# Project:      Momijo
# Module:       src.momijo.nn.init
# File:         init.mojo
# Path:         src/momijo/nn/init.mojo
#
# Description:  Neural-network utilities for Momijo integrating with tensors,
#               optimizers, and training/evaluation loops.
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
#   - Key functions: nn_version, healthcheck, _self_test, main


from momijo.core.error import module
from momijo.tensor.errors import OK
from pathlib import Path
from pathlib.path import Path

fn nn_version() -> String:
    return String("0.1.0")

# Lightweight healthcheck for the nn package.
fn healthcheck() -> Bool:
    # Extend with deeper checks when submodules wire up.
    return True

# Smoke test for this module
fn _self_test() -> Bool:
    var ok = True
    ok = ok and (nn_version().bytes().len() > 0)
    ok = ok and healthcheck()
    return ok
fn main() -> None:
    if _self_test():
        print(String("momijo.nn.init: OK"))
    else:
        print(String("momijo.nn.init: FAILED"))