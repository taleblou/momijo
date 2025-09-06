# ============================================================================
#  Momijo Visualization - scene/facet.mojo
#  Copyright (c) 2025  Morteza Talebou
#  MIT License - https://taleblou.ir/
#  This file follows the user's Mojo checklist: no global/export; __init__(out self,...); var only.
# ============================================================================

struct FacetSpec:
    var by: String   # categorical field
    var cols: Int
    fn __init__(out self):
        self.by = String("")
        self.cols = 2
