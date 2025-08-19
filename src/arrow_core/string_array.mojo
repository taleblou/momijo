# MIT License
# Copyright (c) 2025 Morteza Taleblou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of arrow_core. See LICENSE at repository root.

from momijo.arrow_core.offsets import Offsets
from momijo.arrow_core.bitmap import Bitmap, bitmap_set_valid

# Minimal StringArray: track only element count via Offsets + validity via Bitmap.
struct StringArray:
    var offsets: Offsets
    var validity: Bitmap

# Constructor: __init__(out self)
    fn __init__(out self):
        self.offsets = Offsets()
        self.validity = Bitmap(0, True)

# Function push(mut self, s: String, valid: Bool = True)
    fn push(mut self, s: String, valid: Bool = True):
        # Treat each pushed string as one logical element
        self.offsets.add_length(1)
        var new_count = self.len()
        self.validity = Bitmap(new_count, True)
        if not valid:
            bitmap_set_valid(self.validity, new_count - 1, False)

# Function len(self) -> Int
    fn len(self) -> Int:
        # offsets always starts with an initial 0, so number of elements is len-1
        return self.offsets.len() - 1