# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.nn
# File: src/momijo/nn/init.mojo

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