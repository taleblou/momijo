
# Momijo Arrow Core
# This file is part of the Momijo project. See the LICENSE file at the repository root.

from momijo.arrow_core.bitmap import Bitmap, bitmap_set_valid

fn bitmap_set_range(mut b: Bitmap, start: Int, count: Int, v: Bool):
    var i = 0
    while i < count:
        bitmap_set_valid(b, start + i, v)
        i += 1

fn bitmap_from_bools(bools: List[Bool]) -> Bitmap:
    var n = len(bools)
    var bm = Bitmap(n, False)
    var i = 0
    while i < n:
        if bools[i]:
            bitmap_set_valid(bm, i, True)
        i += 1
    return bm
