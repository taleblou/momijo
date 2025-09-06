# MIT License
# Copyright (c) 2025 Morteza Talebou
# SPDX-License-Identifier: MIT
#
# Module: momijo.nn.init
# Path:   src/momijo/nn/init.mojo
#
# This file provides small, non-invasive package utilities (no re-exports).
# Keep it lightweight to avoid dependency cycles.

# Return a short package version string (manually maintained).
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

fn main():
    if _self_test():
        print(String("momijo.nn.init: OK"))
    else:
        print(String("momijo.nn.init: FAILED"))
