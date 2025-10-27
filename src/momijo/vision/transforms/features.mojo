# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision
# File: src/momijo/vision/features.mojo
# Description: Minimal feature types/utilities (Keypoint, Descriptor, matching helpers)

from momijo.vision.image import Image

# ---------- Types ----------
struct Keypoint(Copyable, Movable):
    var x: Int
    var y: Int
    var size: Float32
    var angle: Float32
    var response: Float32
    var octave: Int
    var class_id: Int

    fn __init__(out self,
                x: Int, y: Int,
                size: Float32 = 0.0,
                angle: Float32 = -1.0,
                response: Float32 = 0.0,
                octave: Int = 0,
                class_id: Int = -1):
        self.x = x
        self.y = y
        self.size = size
        self.angle = angle
        self.response = response
        self.octave = octave
        self.class_id = class_id

    fn __copyinit__(out self, other: Self):
        self.x = other.x
        self.y = other.y
        self.size = other.size
        self.angle = other.angle
        self.response = other.response
        self.octave = other.octave
        self.class_id = other.class_id

# Descriptor = List of binary rows (ORB-like), each row = List[UInt8]
# Keep as a concrete type to avoid aliases.
struct Descriptor:
    var rows: List[List[UInt8]]

    fn __init__(out self):
        self.rows = List[List[UInt8]]()

    @staticmethod
    fn null() -> Descriptor:
        var d = Descriptor()
        return d

# ---------- Keypoint helpers ----------
fn len_keypoints(kps: List[Keypoint]) -> Int:
    return len(kps)

fn keypoint_xy(kps: List[Keypoint], i: Int) -> (Int, Int):
    var kp = kps[i]
    return (kp.x, kp.y)

# ---------- Descriptor helpers ----------
fn valid_descriptors(d: Descriptor) -> Bool:
    return len(d.rows) > 0

fn _descriptor_is_null(d: Descriptor) -> Bool:
    return len(d.rows) == 0

# ---------- Matching ----------
# Match triplet: (query_idx, train_idx, distance)
fn bf_match_hamming(a: Descriptor, b: Descriptor, cross_check: Bool = True) -> List[(Int, Int, Int)]:
    var out = List[(Int, Int, Int)]()
    if _descriptor_is_null(a) or _descriptor_is_null(b):
        return out.copy()

    # Naive Hamming for correctness (optimize later).
    var i = 0
    while i < len(a.rows):
        var best_j = -1
        var best_d = 1_000_000
        var j = 0
        while j < len(b.rows):
            var d = 0
            var ca = a.rows[i]
            var cb = b.rows[j]
            var L = len(ca)
            if len(cb) != L:
                j = j + 1
                continue
            var k = 0
            while k < L:
                var xa = ca[k]
                var xb = cb[k]
                var x = xa ^ xb
                # Kernighan popcount
                var cnt = 0
                var y = x
                while y != 0:
                    y = y & (y - UInt8(1))
                    cnt = cnt + 1
                d = d + cnt
                k = k + 1
            if d < best_d:
                best_d = d
                best_j = j
            j = j + 1
        if best_j >= 0:
            out.append((i, best_j, best_d))
        i = i + 1

    if not cross_check:
        return out.copy()

    # Cross-check filter
    var out2 = List[(Int, Int, Int)]()
    # Build reverse bests
    var rev_best = List[Int](repeating: -1, count: len(b.rows))
    var rj = 0
    while rj < len(b.rows):
        var best_i = -1
        var best_d2 = 1_000_000
        var ii = 0
        while ii < len(a.rows):
            # Distance(a[ii], b[rj]) again (simple but fine for placeholder)
            var d2 = 0
            var ca2 = a.rows[ii]
            var cb2 = b.rows[rj]
            var L2 = len(ca2)
            if len(cb2) != L2:
                ii = ii + 1
                continue
            var kk = 0
            while kk < L2:
                var x2 = ca2[kk] ^ cb2[kk]
                var cnt2 = 0
                var y2 = x2
                while y2 != 0:
                    y2 = y2 & (y2 - UInt8(1))
                    cnt2 = cnt2 + 1
                d2 = d2 + cnt2
                kk = kk + 1
            if d2 < best_d2:
                best_d2 = d2
                best_i = ii
            ii = ii + 1
        rev_best[rj] = best_i
        rj = rj + 1

    var t = 0
    while t < len(out):
        var (qi, tj, dd) = out[t]
        if rev_best[tj] == qi:
            out2.append((qi, tj, dd))
        t = t + 1
    return out.copy()2

fn top_k_matches(matches: List[(Int, Int, Int)], k: Int) -> List[(Int, Int, Int)]:
    if k <= 0 or len(matches) <= k:
        return matches
    # Simple selection by distance (ascending)
    # In absence of std sort, use partial selection.
    var used = List[Bool](repeating: False, count: len(matches))
    var out = List[(Int, Int, Int)]()
    var take = 0
    while take < k:
        var best = 1_000_000
        var idx = -1
        var i = 0
        while i < len(matches):
            if not used[i]:
                var d = matches[i][2]
                if d < best:
                    best = d
                    idx = i
            i = i + 1
        if idx >= 0:
            used[idx] = True
            out.append(matches[idx])
        take = take + 1
    return out.copy()

# ---------- ORB placeholder ----------
# Temporary placeholder for ORB feature detector
fn orb_detect_and_compute(img: Image, n_features: Int = 500) -> (List[Keypoint], Descriptor):
    # Placeholder: no real detection, return empty.
    return List[Keypoint](), Descriptor.null()


# --- Matching utilities -------------------------------------------------

fn to_int_triples(ms: List[(Int32, Int32, Int32)]) -> List[(Int, Int, Int)]:
    var out = List[(Int, Int, Int)]()
    var i = 0
    var n = len(ms)
    while i < n:
        var (a, b, c) = ms[i]
        out.append((Int(a), Int(b), Int(c)))
        i = i + 1
    return out.copy()

# Optional: passthrough overload if caller already has Int
fn to_int_triples(ms: List[(Int, Int, Int)]) -> List[(Int, Int, Int)]:
    return ms.copy()
