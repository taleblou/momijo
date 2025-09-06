# MIT License
# Copyright (c) 2025
# SPDX-License-Identifier: MIT
#
# Module: momijo.nn.loop_classification
# Path:   src/momijo/nn/loop_classification.mojo
#
# A tiny, dependency-light classification training loop (softmax regression).
# Self-contained math (List[Float64]) with:
#   - SoftmaxClassifier (linear logits + softmax)
#   - Cross-entropy loss (stable via log-sum-exp)
#   - Mini-batch SGD optimizer
#   - StepLR scheduler (optional)
#   - Metrics: accuracy, loss
#
# This file is designed to be stand-alone for smoke tests, without importing
# other modules. It is NOT an autograd system; gradients are implemented
# analytically for the linear softmax model.
#
# API (core):
#   struct SoftmaxClassifier(in_features, num_classes)
#     - forward_logits(X[N,D]) -> [N,C]
#     - predict(X[N,D]) -> [N] (argmax)
#   fn train_epoch(...)
#   fn evaluate_accuracy(...)
#
# A self-test trains on a toy, linearly-separable dataset and checks accuracy.

# --------- Helpers ---------
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

fn copy2d(x: List[List[Float64]]) -> List[List[Float64]]:
    var r = len(x)
    var y = List[List[Float64]]()
    for i in range(r):
        var c = 0
        if len(x[i]) > 0: c = len(x[i])
        var row = List[Float64]()
        for j in range(c): row.push(x[i][j])
        y.push(row)
    return y

fn argmax1d(x: List[Float64]) -> Int:
    var n = len(x)
    if n == 0: return 0
    var m = x[0]
    var idx = 0
    for i in range(1, n):
        if x[i] > m:
            m = x[i]
            idx = i
    return idx

# exp via truncated taylor (ok for smoke tests)
fn _exp(x: Float64) -> Float64:
    var term = 1.0
    var sum = 1.0
    var n = 1
    var k = 1.0
    while n <= 20:
        term *= x / k
        sum += term
        n += 1
        k += 1.0
    return sum

# log via solving e^y = x with iterations; clamp for x<=0
fn _log(x: Float64) -> Float64:
    if x <= 0.0: return -745.0
    var y = 0.0
    for it in range(8):
        var ey = _exp(y)
        y = y + 2.0 * (x - ey) / (x + ey)
    return y

fn logsumexp_row(z: List[Float64]) -> Float64:
    var m = -1.7976931348623157e308
    for i in range(len(z)):
        if z[i] > m: m = z[i]
    var s = 0.0
    for i in range(len(z)):
        s += _exp(z[i] - m)
    return m + _log(s)

fn log_softmax_batch(Z: List[List[Float64]]) -> List[List[Float64]]:
    var N = len(Z)
    var C = 0
    if N > 0: C = len(Z[0])
    var Y = zeros2d(N, C)
    for n in range(N):
        var lse = logsumexp_row(Z[n])
        for c in range(C):
            Y[n][c] = Z[n][c] - lse
    return Y

fn cross_entropy_mean(Z: List[List[Float64]], targets: List[Int]) -> Float64:
    # Z: logits [N,C], targets: indices [N]
    var N = len(Z)
    if N == 0: return 0.0
    var C = len(Z[0])
    var L = 0.0
    for n in range(N):
        var t = targets[n]
        if t < 0 or t >= C: continue
        var lse = logsumexp_row(Z[n])
        L += (lse - Z[n][t])
    return L / Float64(N)

# --------- Model: SoftmaxClassifier ---------
struct SoftmaxClassifier:
    var in_features: Int
    var num_classes: Int
    var W: List[List[Float64]]   # [C,D]
    var b: List[Float64]         # [C]

    fn __init__(out self, in_features: Int, num_classes: Int, w_init: Float64 = 0.01):
        self.in_features = in_features
        self.num_classes = num_classes
        self.W = zeros2d(num_classes, in_features)
        self.b = zeros1d(num_classes)
        for c in range(num_classes):
            for d in range(in_features):
                self.W[c][d] = w_init

    fn forward_logits(self, X: List[List[Float64]]) -> List[List[Float64]]:
        var N = len(X)
        var D = self.in_features
        var C = self.num_classes
        var Z = zeros2d(N, C)
        for n in range(N):
            for c in range(C):
                var s = 0.0
                for d in range(D):
                    s += self.W[c][d] * X[n][d]
                Z[n][c] = s + self.b[c]
        return Z

    fn predict(self, X: List[List[Float64]]) -> List[Int]:
        var Z = self.forward_logits(X)
        var N = len(Z)
        var out = List[Int]()
        for n in range(N):
            out.push(argmax1d(Z[n]))
        return out

# --------- Optimizer: mini SGD ---------
struct SGD:
    var lr: Float64
    fn __init__(out self, lr: Float64 = 1e-2): self.lr = lr
    fn set_lr(mut self, lr: Float64): self.lr = lr

# Compute grads and take one SGD step on (W,b) given batch X,y
fn sgd_step(mut model: SoftmaxClassifier, opt: SGD, X: List[List[Float64]], y: List[Int]) -> Float64:
    var N = len(X)
    if N == 0: return 0.0
    var D = model.in_features
    var C = model.num_classes
    var Z = model.forward_logits(X)          # [N,C]
    var ls = log_softmax_batch(Z)            # [N,C]
    # probs and dL/dz = softmax - onehot ; average over N
    var G = zeros2d(N, C)
    for n in range(N):
        var denom = 0.0
        # softmax from log_softmax
        for c in range(C):
            G[n][c] = _exp(ls[n][c])  # temp store P[n][c]
            denom += G[n][c]
        if denom == 0.0: denom = 1.0
        for c in range(C):
            G[n][c] = G[n][c] / denom
        var t = y[n]
        if t >= 0 and t < C:
            G[n][t] -= 1.0
        # mean over batch
        for c in range(C):
            G[n][c] = G[n][c] / Float64(N)
    # grads
    var dW = zeros2d(C, D)
    var db = zeros1d(C)
    for c in range(C):
        var s = 0.0
        for n in range(N):
            db[c] += G[n][c]
        for d in range(D):
            var acc = 0.0
            for n in range(N):
                acc += G[n][c] * X[n][d]
            dW[c][d] = acc
    # update
    var lr = opt.lr
    for c in range(C):
        model.b[c] = model.b[c] - lr * db[c]
        for d in range(D):
            model.W[c][d] = model.W[c][d] - lr * dW[c][d]
    # loss
    return cross_entropy_mean(Z, y)

# --------- StepLR (optional) ---------
struct StepLR:
    var base_lr: Float64
    var step_size: Int
    var gamma: Float64
    var last_epoch: Int
    var current_lr: Float64

    fn __init__(out self, base_lr: Float64, step_size: Int, gamma: Float64 = 0.5):
        if step_size <= 0: step_size = 1
        self.base_lr = base_lr
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = 0
        self.current_lr = base_lr

    fn step(mut self):
        self.last_epoch += 1
        var k = self.last_epoch / self.step_size
        var pow = 1.0
        for i in range(k): pow *= self.gamma
        self.current_lr = self.base_lr * pow

    fn get_lr(self) -> Float64: return self.current_lr

# --------- Data helpers ---------
fn batch_indices(n: Int, batch_size: Int) -> List[List[Int]]:
    var idxs = List[Int]()
    for i in range(n): idxs.push(i)
    # simple deterministic shuffle: reverse chunks (no RNG needed)
    var res = List[List[Int]]()
    var i = 0
    while i < n:
        var end = i + batch_size
        if end > n: end = n
        var chunk = List[Int]()
        for j in range(i, end): chunk.push(idxs[j])
        # reverse the chunk for a touch of mixing
        var m = len(chunk)
        for a in range(m / 2):
            var t = chunk[a]
            chunk[a] = chunk[m - 1 - a]
            chunk[m - 1 - a] = t
        res.push(chunk)
        i = end
    return res

fn slice_rows(X: List[List[Float64]], idxs: List[Int]) -> List[List[Float64]]:
    var out = List[List[Float64]]()
    for k in range(len(idxs)):
        out.push(X[idxs[k]])
    return out

fn slice_vec(y: List[Int], idxs: List[Int]) -> List[Int]:
    var out = List[Int]()
    for k in range(len(idxs)):
        out.push(y[idxs[k]])
    return out

# --------- Training loop ---------
struct TrainLog:
    var epoch_losses: List[Float64]
    var train_acc: List[Float64]

    fn __init__(out self):
        self.epoch_losses = List[Float64]()
        self.train_acc = List[Float64]()

fn accuracy(model: SoftmaxClassifier, X: List[List[Float64]], y: List[Int]) -> Float64:
    var preds = model.predict(X)
    var N = len(y)
    if N == 0: return 0.0
    var ok = 0
    for i in range(N):
        if preds[i] == y[i]: ok += 1
    return Float64(ok) / Float64(N)

fn train_softmax_classifier(mut model: SoftmaxClassifier, X: List[List[Float64]], y: List[Int], epochs: Int = 20, batch_size: Int = 16, lr: Float64 = 0.1, step_lr_size: Int = 10, step_lr_gamma: Float64 = 0.5) -> TrainLog:
    var log = TrainLog()
    var opt = SGD(lr)
    var sched = StepLR(lr, step_lr_size, step_lr_gamma)
    var N = len(X)
    for ep in range(epochs):
        # update optimizer LR from scheduler
        opt.set_lr(sched.get_lr())
        var epoch_loss = 0.0
        var nbatches = 0
        var batches = batch_indices(N, batch_size)
        for b in range(len(batches)):
            var idxs = batches[b]
            var Xb = slice_rows(X, idxs)
            var yb = slice_vec(y, idxs)
            epoch_loss += sgd_step(model, opt, Xb, yb)
            nbatches += 1
        if nbatches == 0: nbatches = 1
        epoch_loss = epoch_loss / Float64(nbatches)
        log.epoch_losses.push(epoch_loss)
        log.train_acc.push(accuracy(model, X, y))
        sched.step()
    return log

# --------- Synthetic dataset for tests ---------
# 3-class, 2D clusters, then expand to D features by adding simple transforms
fn make_toy_dataset(n_per_class: Int = 20, D: Int = 4) -> (List[List[Float64]], List[Int], Int, Int):
    var C = 3
    var N = n_per_class * C
    var X2 = zeros2d(N, 2)
    var y = List[Int]()
    # class 0 around (-2, -2)
    for i in range(n_per_class):
        var n = i
        X2[n][0] = -2.0 + 0.05 * Float64(i)
        X2[n][1] = -2.0 + 0.03 * Float64(i)
        y.push(0)
    # class 1 around (+2, -2)
    for i in range(n_per_class):
        var n = n_per_class + i
        X2[n][0] =  2.0 + 0.04 * Float64(i)
        X2[n][1] = -2.0 + 0.02 * Float64(i)
        y.push(1)
    # class 2 around (0, +2)
    for i in range(n_per_class):
        var n = 2 * n_per_class + i
        X2[n][0] =  0.0 + 0.03 * Float64(i)
        X2[n][1] =  2.0 - 0.02 * Float64(i)
        y.push(2)
    # expand to D by adding polynomial features
    var X = zeros2d(N, D)
    for n in range(N):
        var a = X2[n][0]
        var b = X2[n][1]
        X[n][0] = a
        if D > 1: X[n][1] = b
        if D > 2: X[n][2] = a * b
        if D > 3: X[n][3] = a * a + b * b
        # if D larger, remaining stay zero
    return (X, y, N, C)

# --------- Smoke test ---------
fn _self_test() -> Bool:
    var ok = True
    var (X, y, N, C) = make_toy_dataset(20, 4)
    var model = SoftmaxClassifier(4, C, 0.01)
    var log = train_softmax_classifier(model, X, y, 25, 16, 0.2, 10, 0.5)
    var acc = accuracy(model, X, y)
    ok = ok and (acc >= 0.85)
    ok = ok and (len(log.epoch_losses) == 25)
    return ok

 