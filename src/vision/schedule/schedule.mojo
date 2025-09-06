# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.vision
# File: src/momijo/vision/schedule.mojo
#
# Minimal scheduling utilities for Momijo Vision (standalone, dependency-light).
# Focus: range-splitting, tile ordering, and static work assignment for workers.
# Style:
#   - No 'export', no 'let', no 'inout'.
#   - Constructors use `fn __init__(out self, ...)`.
#
# This module is CPU-agnostic and does not spawn threads. It just computes
# static schedules (lists) that a higher-level executor can consume.
#
# Implemented:
#   - Work1D(start, count)
#   - Tile(x,y,w,h)  (duplicated locally to avoid imports)
#   - split_1d(n, parts) -> List[Work1D]                       # contiguous range splits
#   - split_equal(n, chunk) -> List[Work1D]                    # fixed-size chunking
#   - tile_grid(h,w, tile_h,tile_w, overlap_h,overlap_w) -> List[Tile]
#   - tile_order(tiles, mode) -> List[Int]                     # permutation of indices
#   - assign_round_robin(m, k) -> List[Int]                    # worker id for each item
#   - assign_by_load(items, weights, k) -> List[Int]           # greedy load balance by weights
#   - shard_tiles(tiles, workers, mode) -> List[List[Int]]     # per-worker lists (by index)
#   - batch_indices(idx, batch_size) -> List[List[Int]]        # mini-batches of indices
#   - __self_test__() -> Bool
#
# Notes:
# - Overlaps in tile_grid are handled via stride = tile_dim - overlap (>=1).
# - tile_order supports RASTER (row-major), SERPENTINE (zig-zag), and MORTON (Z-order, interleaved bits).
# - assign_by_load uses a simple greedy algorithm (assign next heaviest to lightest worker).

# -------------------------
# Basic structs
# -------------------------
struct Work1D(Copyable, Movable):
    var start: Int
    var count: Int
    fn __init__(out self, start: Int, count: Int):
        self.start = start
        self.count = count
    fn end(self) -> Int: return self.start + self.count
    fn to_string(self) -> String:
        return String("Work1D(") + String(self.start) + String(", ") + String(self.count) + String(")")

struct Tile(Copyable, Movable):
    var x: Int
    var y: Int
    var w: Int
    var h: Int
    fn __init__(out self, x: Int, y: Int, w: Int, h: Int):
        self.x = x; self.y = y; self.w = w; self.h = h
    fn to_string(self) -> String:
        return String("Tile(") + String(self.x) + String(",") + String(self.y) + String(",") + String(self.w) + String("x") + String(self.h) + String(")")

struct TileOrderMode(Copyable, Movable):
    var id: Int
    fn __init__(out self, id: Int): self.id = id
    @staticmethod fn RASTER()    -> TileOrderMode: return TileOrderMode(0)
    @staticmethod fn SERPENTINE()-> TileOrderMode: return TileOrderMode(1)
    @staticmethod fn MORTON()    -> TileOrderMode: return TileOrderMode(2)
    fn __eq__(self, other: TileOrderMode) -> Bool: return self.id == other.id

# -------------------------
# Helpers
# -------------------------
@staticmethod
fn _min(a: Int, b: Int) -> Int:
    if a < b: return a
    return b

@staticmethod
fn _max(a: Int, b: Int) -> Int:
    if a > b: return a
    return b

@staticmethod
fn _alloc_i32(n: Int) -> List[Int]:
    var out: List[Int] = List[Int]()
    var i = 0
    while i < n:
        out.append(0); i += 1
    return out

@staticmethod
fn _alloc_list_int(n: Int) -> List[Int]:
    var out: List[Int] = List[Int]()
    var i = 0
    while i < n: out.append(0); i += 1
    return out

@staticmethod
fn _append_int(mut v: List[Int], x: Int) -> List[Int]:
    v.append(x); return v

# -------------------------
# 1D splits
# -------------------------
@staticmethod
fn split_1d(n: Int, parts: Int) -> List[Work1D]:
    var out: List[Work1D] = List[Work1D]()
    if n <= 0 or parts <= 0:
        return out
    var base = n // parts
    var rem = n % parts
    var start = 0
    var i = 0
    while i < parts:
        var cnt = base
        if i < rem: cnt = cnt + 1
        if cnt > 0:
            out.append(Work1D(start, cnt))
            start = start + cnt
        i += 1
    return out

@staticmethod
fn split_equal(n: Int, chunk: Int) -> List[Work1D]:
    var out: List[Work1D] = List[Work1D]()
    if n <= 0 or chunk <= 0:
        return out
    var start = 0
    while start < n:
        var cnt = chunk
        if start + cnt > n: cnt = n - start
        out.append(Work1D(start, cnt))
        start = start + cnt
    return out

# -------------------------
# Tiling
# -------------------------
@staticmethod
fn tile_grid(h: Int, w: Int, tile_h: Int, tile_w: Int, overlap_h: Int, overlap_w: Int) -> List[Tile]:
    var tiles: List[Tile] = List[Tile]()
    if h <= 0 or w <= 0 or tile_h <= 0 or tile_w <= 0:
        return tiles

    var ov_h = overlap_h; if ov_h < 0: ov_h = 0; if ov_h >= tile_h: ov_h = tile_h - 1
    var ov_w = overlap_w; if ov_w < 0: ov_w = 0; if ov_w >= tile_w: ov_w = tile_w - 1
    var stride_h = tile_h - ov_h; if stride_h < 1: stride_h = 1
    var stride_w = tile_w - ov_w; if stride_w < 1: stride_w = 1

    var y = 0
    while True:
        var th = tile_h
        if y + th > h: th = h - y
        var x = 0
        while True:
            var tw = tile_w
            if x + tw > w: tw = w - x
            tiles.append(Tile(x, y, tw, th))
            if x + tile_w >= w:
                break
            x = x + stride_w
            if x + tile_w > w:
                x = _max(0, w - tile_w)
        if y + tile_h >= h:
            break
        y = y + stride_h
        if y + tile_h > h:
            y = _max(0, h - tile_h)
    return tiles

# -------------------------
# Tile ordering
# -------------------------
@staticmethod
fn _morton2D(x: Int, y: Int) -> Int:
    # Interleave bits of x and y: morton = x0 y0 x1 y1 ...
    var v = 0
    var bit = 0
    var i = 0
    while i < 16:  # enough for typical tile grid counts
        var xb = (x >> i) & 1
        var yb = (y >> i) & 1
        v = v | (xb << bit)
        bit = bit + 1
        v = v | (yb << bit)
        bit = bit + 1
        i += 1
    return v

@staticmethod
fn tile_order(tiles: List[Tile], mode: TileOrderMode) -> List[Int]:
    var idx: List[Int] = List[Int]()
    var n = len(tiles)
    var i = 0
    while i < n: idx.append(i); i += 1
    if n <= 1: return idx

    # Simple bubble-like sort (since n of tiles usually modest). Could be replaced by better sort when available.
    var swapped = True
    while swapped:
        swapped = False
        var j = 1
        while j < n:
            var a = idx[j-1]; var b = idx[j]
            var swap = False
            if mode == TileOrderMode.RASTER():
                # row-major: y,x
                if tiles[a].y > tiles[b].y or (tiles[a].y == tiles[b].y and tiles[a].x > tiles[b].x):
                    swap = True
            elif mode == TileOrderMode.SERPENTINE():
                var ya = tiles[a].y; var yb = tiles[b].y
                if ya != yb:
                    if ya > yb: swap = True
                else:
                    # if even row: x ascending; odd row: x descending
                    var row = ya
                    if (row % 2) == 0:
                        if tiles[a].x > tiles[b].x: swap = True
                    else:
                        if tiles[a].x < tiles[b].x: swap = True
            else:
                # MORTON
                var ma = _morton2D(tiles[a].x, tiles[a].y)
                var mb = _morton2D(tiles[b].x, tiles[b].y)
                if ma > mb: swap = True

            if swap:
                var tmp = idx[j-1]
                idx[j-1] = idx[j]
                idx[j] = tmp
                swapped = True
            j += 1
    return idx

# -------------------------
# Work assignment
# -------------------------
@staticmethod
fn assign_round_robin(m: Int, k: Int) -> List[Int]:
    # returns worker id [0..k-1] for each of m items
    var out: List[Int] = List[Int]()
    if m <= 0 or k <= 0: return out
    var i = 0
    while i < m:
        out.append(i % k)
        i += 1
    return out

@staticmethod
fn assign_by_load(items: List[Int], weights: List[Int], k: Int) -> List[Int]:
    # Greedy: sort items by weight desc, assign to current lightest worker
    var m = len(items)
    var out: List[Int] = List[Int]()
    var i = 0
    while i < m: out.append(0); i += 1
    if k <= 0 or m == 0: return out

    # indices 0..m-1
    var order: List[Int] = List[Int]()
    i = 0
    while i < m: order.append(i); i += 1

    # simple bubble sort by weights desc on 'order'
    var swapped = True
    while swapped:
        swapped = False
        var j = 1
        while j < m:
            var a = order[j-1]; var b = order[j]
            if weights[a] < weights[b]:
                var tmp = order[j-1]; order[j-1] = order[j]; order[j] = tmp; swapped = True
            j += 1

    var loads = _alloc_i32(k)
    i = 0
    while i < m:
        # find worker with minimum load
        var wi = 0; var best = loads[0]; var t = 1
        while t < k:
            if loads[t] < best:
                best = loads[t]; wi = t
            t += 1
        var it = order[i]
        out[it] = wi
        loads[wi] = loads[wi] + weights[it]
        i += 1
    return out

@staticmethod
fn shard_tiles(tiles: List[Tile], workers: Int, mode: TileOrderMode) -> List[List[Int]]:
    # Returns per-worker lists of tile indices according to 'mode' ordering, assigned round-robin.
    var order = tile_order(tiles, mode)
    var m = len(order)
    var w = workers; if w < 1: w = 1
    # init groups
    var groups: List[List[Int]] = List[List[Int]]()
    var i = 0
    while i < w:
        var g: List[Int] = List[Int]()
        groups.append(g)
        i += 1
    # assign
    i = 0
    while i < m:
        var wid = i % w
        var g2 = groups[wid]
        g2.append(order[i])
        groups[wid] = g2
        i += 1
    return groups

# -------------------------
# Batching helper
# -------------------------
@staticmethod
fn batch_indices(idx: List[Int], batch_size: Int) -> List[List[Int]]:
    var out: List[List[Int]] = List[List[Int]]()
    if batch_size <= 0:
        return out
    var i = 0
    while i < len(idx):
        var g: List[Int] = List[Int]()
        var j = 0
        while j < batch_size and (i + j) < len(idx):
            g.append(idx[i + j])
            j += 1
        out.append(g)
        i = i + batch_size
    return out

# -------------------------
# Minimal smoke test
# -------------------------
@staticmethod
fn __self_test__() -> Bool:
    # 1D split
    var s = split_1d(10, 3)
    if len(s) != 3: return False
    if not (s[0].start == 0 and s[0].count == 4): return False
    if not (s[1].start == 4 and s[1].count == 3): return False
    if not (s[2].start == 7 and s[2].count == 3): return False

    var chunks = split_equal(9, 4)
    if not (len(chunks) == 3 and chunks[0].count == 4 and chunks[1].count == 4 and chunks[2].count == 1): return False

    # Tiles
    var tiles = tile_grid(5, 7, 3, 4, 1, 1)
    if len(tiles) < 3: return False

    var ord_r = tile_order(tiles, TileOrderMode.RASTER())
    var ord_s = tile_order(tiles, TileOrderMode.SERPENTINE())
    var ord_m = tile_order(tiles, TileOrderMode.MORTON())
    if not (len(ord_r) == len(tiles) and len(ord_s) == len(tiles) and len(ord_m) == len(tiles)): return False

    # Assignment
    var rr = assign_round_robin(7, 3)
    if not (len(rr) == 7 and rr[0] == 0 and rr[1] == 1 and rr[2] == 2 and rr[3] == 0): return False

    var items: List[Int] = List[Int](); items.append(0); items.append(1); items.append(2); items.append(3)
    var weights: List[Int] = List[Int](); weights.append(5); weights.append(1); weights.append(4); weights.append(2)
    var ass = assign_by_load(items, weights, 2)
    if len(ass) != 4: return False

    var groups = shard_tiles(tiles, 3, TileOrderMode.SERPENTINE())
    if len(groups) != 3: return False

    # Batching
    var idx: List[Int] = List[Int](); var i = 0
    while i < 10: idx.append(i); i += 1
    var batches = batch_indices(idx, 4)
    if not (len(batches) == 3 and len(batches[0]) == 4 and len(batches[2]) == 2): return False

    return True
