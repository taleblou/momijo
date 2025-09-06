# ============================================================================
#  Momijo Visualization - runtime/backend_select.mojo
#  Copyright (c) 2025  Morteza Talebou  (https://taleblou.ir/)
#  Licensed under the MIT License. See LICENSE in the project root.
# ============================================================================

struct BackendKind:
    var value: Int
    fn __init__(out self, value: Int):
        self.value = value

struct BackendKinds:
    @staticmethod
    fn svg() -> BackendKind: return BackendKind(0)
    @staticmethod
    fn png() -> BackendKind: return BackendKind(1)

fn _ext(path: String) -> String:
    var n = len(path)
    var i = n - 1
    var dot = -1
    while i >= 0:
        if path[i] == 46:  # '.'
            dot = i
            break
        i -= 1
    if dot >= 0 and dot+1 < n:
        return String(path[dot+1:n])
    return String("")

fn select_backend_from_ext(path: String) -> BackendKind:
    var e = _ext(path)
    if e == String("svg"): return BackendKinds.svg()
    if e == String("png"): return BackendKinds.png()
    # default
    return BackendKinds.svg()
