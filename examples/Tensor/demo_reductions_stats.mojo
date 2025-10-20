# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tensor
# Module:       tests.tensor
# File:         tests/tensor/demo_reductions_stats.mojo
#
# Description:
#   Demo for reductions and basic statistics:
#   - sum / mean / std / min / max
#   - dim-wise reductions with keepdim
#   - NaN-safe reducers (nanmean, nansum)
#   - uniques (values/counts) and bincount
#   - topk, sort, argsort
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from momijo.tensor import tensor

# -----------------------------------------------------------------------------
# Small banner printer
# -----------------------------------------------------------------------------
fn banner(title: String) -> None:
    print("\n=== " + title + " ===")

# -----------------------------------------------------------------------------
# 7) Reductions & Stats
# -----------------------------------------------------------------------------
fn demo_reductions_stats() -> None:
    banner("7) REDUCTIONS & STATS")

    var x = tensor.arange(1, 7, 1).to_float64().reshape([2, 3])
    print("x:\n" + x.__str__())

    # Basic scalar reductions
    var sum_str  = x.sum().__str__()
    var mean_str = x.mean().__str__()
    var std_str  = x.std().__str__()
    print("sum / mean / std:\n" + sum_str + " | " + mean_str + " | " + std_str)

    # Min/max and a dim-wise sum with keepdim=True
    print("min / max: " + x.min().__str__() + " " + x.max().__str__())
    print("sum(dim=0, keepdim=True):\n" + x.sum(0, True).__str__())

    # NaN-safe reducers (works even if x contains NaNs)
    var xn = x.clone()
    print("nanmean over dim=1: " + xn.nanmean(1).__str__())
    print("nansum (all elements): " + xn.nansum().__str__())

    # Discrete stats: uniques (values/counts) and bincount
    var q = tensor.from_list_int([1, 2, 2, 3, 3, 3, 5])
    var req = q.unique()
    print("unique values:\n" + req.values.__str__())
    print("counts per unique (aligned):\n" + req.counts.__str__())
    print("bincount (0..max): " + q.bincount().__str__())

    # Order statistics: topk, sort, argsort
    var r = tensor.from_list_float64([0.2, 1.5, -3.2, 4.1, 0.0])
    var topk = r.topk(3)  # returns [values, indices] pair
    print("topk = 3:")
    print(topk[0].__str__())  # top values
    print(topk[1].__str__())  # their indices

    print("sorted:\n" + r.sort().__str__())
    print("argsort:\n" + r.argsort().__str__())

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
fn main() -> None:
    demo_reductions_stats()
