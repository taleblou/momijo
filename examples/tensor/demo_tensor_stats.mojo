# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tests
# Module:       tests.test_tensor_stats
# File:         examples/Tensor/demo_tensor_stats.mojo
#
# Description:
#   sort / unique / bincount / histogram / digitize tests for momijo.tensor.
 
from momijo.tensor import tensor

# -----------------------------------------------------------------------------
# sort / unique / hist tests
# -----------------------------------------------------------------------------
fn np_sort_unique_hist() -> None:
    print("\n=== np_sort_unique_hist ===")

    # Sample int tensor
    var x = tensor.Tensor([3, 1, 2, 3, 2, 1, 1, 5, 4, 4])

    # 1) sort
    var sorted_x = x.sort()
    print("sorted: " + sorted_x.__str__())

    # 2) unique with counts (tuple-like result: p1=values, p2=counts)
    var res = x.unique()
    var uniques = res.values.copy()
    var counts  = res.counts.copy()
    print("unique: " + uniques.__str__() + " | counts: " + counts.__str__())

    # 3) bincount (index = value, value = frequency)
    var binc = x.bincount()
    print("bincount: " + binc.__str__())

    # 4) histogram with explicit bin edges [0,2), [2,4), [4,6)
    res = x.histogram(bins=[0, 2, 4, 6])
    var hist  = res.values.copy()
    var edges = res.counts.copy()
    print("hist: " + hist.__str__() + " | edges: " + edges.__str__())

    # 5) digitize values into bin indices using edges [2, 4]
    #    Boundary handling follows your library's convention.
    var bins = x.digitize([2, 4])
    print("digitize: " + bins.__str__())

    # ---- Lightweight sanity checks (non-failing prints) ----
    # Ensure lengths match for unique & counts
    print("len(unique) = " + String(uniques.__len__()) + " | len(counts) = " + String(counts.__len__()))

    # Sum of counts should equal original length (if sum() is supported)
    var total: Int = counts.sum_all()
    print("sum(counts) (should equal len(x)) = " + String(total))
    print("len(x) = " + String(x.__len__()))

    # Histogram sum equals len(x) if all values fall inside [0,6)
    print("hist sum = " + String(hist.sum_all()))

# -----------------------------------------------------------------------------
# entrypoint
# -----------------------------------------------------------------------------
fn main() -> None:
    np_sort_unique_hist()
