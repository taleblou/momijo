# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.distributed.env
# File:         src/momijo/learn/distributed/env.mojo
#
# Description:
#   Environment helpers for distributed setups (reading RANK/WORLD_SIZE).
#   Uses a compatibility shim because `sys.env.getenv` may be unavailable
#   in some Mojo toolchains.

# -----------------------------------------------------------------------------
# Compat shim: getenv_compat(key) -> Optional[String]

#   from sys.env import getenv
#   @always_inline
#   fn getenv_compat(key: String) -> Optional[String]:
#       return getenv(key)
# -----------------------------------------------------------------------------
@always_inline
fn getenv_compat(_key: String) -> Optional[String]:
    var _ = _key
    return None

# -----------------------------------------------------------------------------
# Parse helpers (بدون هیچ cast/raises)
# -----------------------------------------------------------------------------
@always_inline
fn _parse_int_opt(s: String) -> Optional[Int]:
    if len(s) == 0:
        return None

    var neg = False
    var i = 0
    if s[0] == "-":
        neg = True
        i = 1
        if i >= len(s):
            return None  # فقط "-"

    var acc: Int = 0
    while i < len(s):

        var ch = s[i]
        var d: Optional[Int] = None

        if ch == "0": d = 0
        elif ch == "1": d = 1
        elif ch == "2": d = 2
        elif ch == "3": d = 3
        elif ch == "4": d = 4
        elif ch == "5": d = 5
        elif ch == "6": d = 6
        elif ch == "7": d = 7
        elif ch == "8": d = 8
        elif ch == "9": d = 9
        else:
            return None


        acc = acc * 10 + d.value()
        i = i + 1

    return -acc if neg else acc

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
@always_inline
fn get_rank_world() -> (Int, Int, Bool):
    # Returns (rank, world_size, ok)
    var r_opt = getenv_compat("RANK")
    var w_opt = getenv_compat("WORLD_SIZE")
    if r_opt is None or w_opt is None:
        return (0, 1, False)

    var r_i = _parse_int_opt(r_opt.value())
    var w_i = _parse_int_opt(w_opt.value())
    if r_i is None or w_i is None:
        return (0, 1, False)

    return (r_i.value(), w_i.value(), True)
