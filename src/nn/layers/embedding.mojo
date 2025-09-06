# MIT License
# Copyright (c) 2025
# SPDX-License-Identifier: MIT
#
# Module: momijo.nn.embedding
# Path:   src/momijo/nn/embedding.mojo
#
# Minimal Embedding layer for pedagogy/smoke tests.
# Works on indices (Int) and returns List-based Float64 vectors/matrices.
#
# Momijo style:
# - No global vars, no `export`. Use `var` (not `let`).
# - Constructors: fn __init__(out self, ...)
# - Prefer `mut/out` over `inout`.

# --- Helpers ---
fn zeros1d(n: Int) -> List[Float64]:
    var y = List[Float64]()
    for i in range(n): y.push(0.0)
    return y

fn zeros2d(r: Int, c: Int) -> List[List[Float64]]:
    var y = List[List[Float64]]()
    for i in range(r):
        var row = List[Float64]()
        for j in range(c): row.push(0.0)
        y.push(row)
    return y

fn add1d(mut a: List[Float64], b: List[Float64]) -> List[Float64]:
    var n = len(a)
    for i in range(n): a[i] += b[i]
    return a

fn scale1d(x: List[Float64], s: Float64) -> List[Float64]:
    var n = len(x)
    var y = zeros1d(n)
    for i in range(n): y[i] = x[i] * s
    return y

# --- Embedding ---
struct Embedding:
    var vocab_size: Int
    var embedding_dim: Int
    var padding_idx: Int
    var weight: List[List[Float64]]  # [vocab_size][embedding_dim]

    fn __init__(out self, num_embeddings: Int, embedding_dim: Int, padding_idx: Int = -1, w_init: Float64 = 0.01):
        self.vocab_size = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        # deterministic tiny init
        self.weight = zeros2d(num_embeddings, embedding_dim)
        for i in range(num_embeddings):
            for j in range(embedding_dim):
                self.weight[i][j] = w_init  # constant init for pedagogy

    # Get embedding vector for a single index (clamped to [0, vocab_size-1]).
    # If index equals padding_idx (>=0), returns zeros.
    fn forward_index(self, index: Int) -> List[Float64]:
        var idx = index
        if idx < 0: idx = 0
        if idx >= self.vocab_size: idx = self.vocab_size - 1
        if self.padding_idx >= 0 and idx == self.padding_idx:
            return zeros1d(self.embedding_dim)
        return self.weight[idx]

    # Get embeddings for a list of indices -> matrix [len(indices), embedding_dim]
    fn forward_indices(self, indices: List[Int]) -> List[List[Float64]]:
        var n = len(indices)
        var Y = zeros2d(n, self.embedding_dim)
        for t in range(n):
            var row = self.forward_index(indices[t])
            for j in range(self.embedding_dim):
                Y[t][j] = row[j]
        return Y

    # EmbeddingBag (sum): sum rows for given indices (skips padding_idx rows)
    fn embed_bag_sum(self, indices: List[Int]) -> List[Float64]:
        var acc = zeros1d(self.embedding_dim)
        var n = len(indices)
        for t in range(n):
            var idx = indices[t]
            if idx < 0: idx = 0
            if idx >= self.vocab_size: idx = self.vocab_size - 1
            if self.padding_idx >= 0 and idx == self.padding_idx:
                continue
            acc = add1d(acc, self.weight[idx])
        return acc

    # EmbeddingBag (mean): average of selected rows (ignoring paddings). If none, returns zeros.
    fn embed_bag_mean(self, indices: List[Int]) -> List[Float64]:
        var acc = zeros1d(self.embedding_dim)
        var count: Int = 0
        var n = len(indices)
        for t in range(n):
            var idx = indices[t]
            if idx < 0: idx = 0
            if idx >= self.vocab_size: idx = self.vocab_size - 1
            if self.padding_idx >= 0 and idx == self.padding_idx:
                continue
            acc = add1d(acc, self.weight[idx])
            count += 1
        if count == 0: 
            return acc
        return scale1d(acc, 1.0 / Float64(count))

    # Mutator: set embedding row (clamps to vocab)
    fn set_row(mut self, index: Int, vec: List[Float64]):
        var idx = index
        if idx < 0: idx = 0
        if idx >= self.vocab_size: idx = self.vocab_size - 1
        var m = self.embedding_dim
        for j in range(m):
            self.weight[idx][j] = vec[j]

# --- Smoke tests ---
fn _self_test() -> Bool:
    var ok = True

    var emb = Embedding(5, 3, padding_idx=0, w_init=0.2)

    # Single index
    var v2 = emb.forward_index(2)
    ok = ok and (len(v2) == 3)

    # Padding index -> zeros
    var v0 = emb.forward_index(0)
    ok = ok and (v0[0] == 0.0 and v0[1] == 0.0 and v0[2] == 0.0)

    # Indices matrix
    var idxs = List[Int](); idxs.push(0); idxs.push(1); idxs.push(4); idxs.push(9)  # 9 will clamp to 4
    var M = emb.forward_indices(idxs)
    ok = ok and (len(M) == 4) and (len(M[0]) == 3)

    # Bag-sum (skip padding=0)
    var sumv = emb.embed_bag_sum(idxs)
    ok = ok and (len(sumv) == 3)

    # Bag-mean
    var meanv = emb.embed_bag_mean(idxs)
    ok = ok and (len(meanv) == 3)

    # set_row then read back
    var nv = zeros1d(3); nv[0] = 1.0; nv[1] = 2.0; nv[2] = 3.0
    emb.set_row(3, nv)
    var v3 = emb.forward_index(3)
    ok = ok and (v3[0] == 1.0 and v3[1] == 2.0 and v3[2] == 3.0)

    return ok
 
