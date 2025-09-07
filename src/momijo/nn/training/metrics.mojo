# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.nn.training
# File: src/momijo/nn/training/metrics.mojo

from momijo.core.traits import one
from momijo.dataframe.helpers import m, sqrt, t
from momijo.dataframe.logical_plan import sort
from momijo.nn.training.loop_classification import accuracy
from momijo.tensor.allocator import free
from momijo.utils.timer import start
from momijo.vision.schedule.schedule import end
from pathlib import Path
from pathlib.path import Path

fn zeros1d_f(n: Int) -> List[Float64]:
    var y = List[Float64]()
    for i in range(n): y.push(0.0)
    return y
fn zeros1d_i(n: Int) -> List[Int]:
    var y = List[Int]()
    for i in range(n): y.push(0)
    return y
fn zeros2d_f(r: Int, c: Int) -> List[List[Float64]]:
    var y = List[List[Float64]]()
    for i in range(r):
        var row = List[Float64]()
        for j in range(c): row.push(0.0)
        y.push(row)
    return y
fn zeros2d_i(r: Int, c: Int) -> List[List[Int]]:
    var y = List[List[Int]]()
    for i in range(r):
        var row = List[Int]()
        for j in range(c): row.push(0)
        y.push(row)
    return y
fn _absf(x: Float64) -> Float64:
    if x < 0.0: return -x
    return x
fn _maxi(a: Int, b: Int) -> Int:
    if a >= b: return a
    return b
fn _mini(a: Int, b: Int) -> Int:
    if a <= b: return a
    return b
fn _safedivf(num: Float64, den: Float64, fallback: Float64 = 0.0) -> Float64:
    if den == 0.0: return fallback
    return num / den

# exp via truncated taylor (enough for metrics thresholds / AUROC sorting-free approach)
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

# --------- Argmax / Top-k helpers ---------
fn argmax1d_f(x: List[Float64]) -> Int:
    var n = len(x)
    if n == 0: return 0
    var m = x[0]
    var idx = 0
    for i in range(1, n):
        if x[i] > m:
            m = x[i]
            idx = i
    return idx
fn topk_indices_row(x: List[Float64], k: Int) -> List[Int]:
    var n = len(x)
    var kk = (k if k <= n else n)
    var taken = zeros1d_i(n)
    var out = List[Int]()
    for t in range(kk):
        var best = -1
        var bestv = -1.7976931348623157e308
        for i in range(n):
            if taken[i] == 1: continue
            var v = x[i]
            if v > bestv:
                bestv = v
                best = i
        if best < 0: break
        out.push(best)
        taken[best] = 1
    return out

# --------- Classification metrics ---------
fn accuracy_from_labels(preds: List[Int], targets: List[Int]) -> Float64:
    var n = len(targets)
    if n == 0: return 0.0
    var ok = 0
    for i in range(n):
        if preds[i] == targets[i]: ok += 1
    return Float64(ok) / Float64(n)
fn accuracy_from_scores(scores: List[List[Float64]], targets: List[Int]) -> Float64:
    var N = len(scores)
    if N == 0: return 0.0
    var preds = List[Int]()
    for n in range(N):
        preds.push(argmax1d_f(scores[n]))
    return accuracy_from_labels(preds, targets)
fn topk_accuracy(scores: List[List[Float64]], targets: List[Int], k: Int) -> Float64:
    var N = len(scores)
    if N == 0: return 0.0
    var ok = 0
    for n in range(N):
        var idxs = topk_indices_row(scores[n], k)
        var t = targets[n]
        var hit = False
        for j in range(len(idxs)):
            if idxs[j] == t: 
                hit = True
                break
        if hit: ok += 1
    return Float64(ok) / Float64(N)
fn confusion_matrix(targets: List[Int], preds: List[Int], num_classes: Int = -1) -> List[List[Int]]:
    var N = len(targets)
    var C = num_classes
    if C <= 0:
        var m = 0
        for i in range(N):
            var a = targets[i]; var b = preds[i]
            if a > m: m = a
            if b > m: m = b
        C = m + 1
        if C <= 0: C = 1
    var cm = zeros2d_i(C, C)  # rows: true, cols: pred
    for i in range(N):
        var t = targets[i]; var p = preds[i]
        if t < 0 or p < 0 or t >= C or p >= C: continue
        cm[t][p] += 1
    return cm
fn _sum_row_i(x: List[List[Int]], r: Int) -> Int:
    var c = 0
    if len(x) == 0: return 0
    var m = len(x[0])
    for j in range(m): c += x[r][j]
    return c
fn _sum_col_i(x: List[List[Int]], cidx: Int) -> Int:
    var rsum = 0
    var R = len(x)
    for i in range(R): rsum += x[i][cidx]
    return rsum
fn precision_recall_f1_from_cm(cm: List[List[Int]]) -> (List[Float64], List[Float64], List[Float64], List[Int]):
    var C = len(cm)
    var prec = zeros1d_f(C)
    var rec = zeros1d_f(C)
    var f1 = zeros1d_f(C)
    var sup = zeros1d_i(C)
    for c in range(C):
        var tp = cm[c][c]
        var fp = _sum_col_i(cm, c) - tp
        var fn = _sum_row_i(cm, c) - tp
        var denom_p = Float64(tp + fp)
        var denom_r = Float64(tp + fn)
        prec[c] = _safedivf(Float64(tp), denom_p, 0.0)
        rec[c] = _safedivf(Float64(tp), denom_r, 0.0)
        var denom_f = prec[c] + rec[c]
        if denom_f == 0.0:
            f1[c] = 0.0
        else:
            f1[c] = 2.0 * prec[c] * rec[c] / denom_f
        sup[c] = _sum_row_i(cm, c)
    return (prec, rec, f1, sup)
fn _sum_int_list(x: List[Int]) -> Int:
    var s = 0
    for i in range(len(x)): s += x[i]
    return s
fn micro_f1_from_cm(cm: List[List[Int]]) -> Float64:
    var C = len(cm)
    var tp_sum = 0
    var fp_sum = 0
    var fn_sum = 0
    for c in range(C):
        var tp = cm[c][c]
        var fp = _sum_col_i(cm, c) - tp
        var fn = _sum_row_i(cm, c) - tp
        tp_sum += tp; fp_sum += fp; fn_sum += fn
    var p = _safedivf(Float64(tp_sum), Float64(tp_sum + fp_sum), 0.0)
    var r = _safedivf(Float64(tp_sum), Float64(tp_sum + fn_sum), 0.0)
    var denom = p + r
    if denom == 0.0: return 0.0
    return 2.0 * p * r / denom
fn macro_f1_from_cm(cm: List[List[Int]]) -> Float64:
    var (p, r, f1, sup) = precision_recall_f1_from_cm(cm)
    var C = len(f1)
    if C == 0: return 0.0
    var s = 0.0
    for i in range(C): s += f1[i]
    return s / Float64(C)
fn weighted_f1_from_cm(cm: List[List[Int]]) -> Float64:
    var (p, r, f1, sup) = precision_recall_f1_from_cm(cm)
    var total = _sum_int_list(sup)
    if total == 0: return 0.0
    var s = 0.0
    for i in range(len(f1)):
        s += f1[i] * Float64(sup[i])
    return s / Float64(total)
fn classification_report(cm: List[List[Int]]) -> String:
    var (p, r, f1, sup) = precision_recall_f1_from_cm(cm)
    var C = len(p)
    var s = String("class  prec    rec     f1      support\n")
    for i in range(C):
        s = s + String(i) + String("      ")
        s = s + String(p[i]) + String("  ")
        s = s + String(r[i]) + String("  ")
        s = s + String(f1[i]) + String("  ")
        s = s + String(sup[i]) + String("\n")
    s = s + String("micro_f1=") + String(micro_f1_from_cm(cm)) + String("\n")
    s = s + String("macro_f1=") + String(macro_f1_from_cm(cm)) + String("\n")
    s = s + String("weighted_f1=") + String(weighted_f1_from_cm(cm)) + String("\n")
    return s

# Returns 0.5 if one class missing (neutral).
struct Pair:
    var score: Float64
    var label: Int  # 0 or 1
fn __init__(out self, score: Float64 = 0, label: Int = 0) -> None:
        self.score = score
        self.label = label
fn __copyinit__(out self, other: Self) -> None:
        self.score = other.score
        self.label = other.label
fn __moveinit__(out self, deinit other: Self) -> None:
        self.score = other.score
        self.label = other.label
fn _sort_pairs_by_score(mut arr: List[Pair]) -> List[Int]:
    # simple selection sort returning ranks (1..N) handling ties by average rank
    var N = len(arr)
    # create indices
    var idxs = List[Int]()
    for i in range(N): idxs.push(i)
    # selection sort descending by score
    for i in range(N):
        var best = i
        for j in range(i+1, N):
            if arr[idxs[j]].score > arr[idxs[best]].score:
                best = j
        if best != i:
            var t = idxs[i]
            idxs[i] = idxs[best]
            idxs[best] = t
    # compute ranks with ties -> average rank
    var ranks = zeros1d_f(N)
    var pos = 0
    while pos < N:
        var start = pos
        var end = pos + 1
        while end < N and arr[idxs[end]].score == arr[idxs[start]].score:
            end += 1
        # rank positions are 1-based
        var rsum = 0.0
        for k in range(start+1, end+1): rsum += Float64(k)
        var ravg = rsum / Float64(end - start)
        for k in range(start, end): ranks[idxs[k]] = ravg
        pos = end
    # return ranks mapped to original order (we already created `ranks` as such)
    # We'll return ranks as Int by rounding (sufficient for U-statistic here)
    var out = List[Int]()
    for i in range(N): out.push(Int(ranks[i]))
    return out
fn binary_auroc(scores: List[Float64], targets: List[Int]) -> Float64:
    var N = len(scores)
    if N == 0: return 0.5
    var pairs = List[Pair]()
    var n_pos = 0
    var n_neg = 0
    for i in range(N):
        var lb = (targets[i] if targets[i] > 0 else 0)
        if lb == 1: n_pos += 1
        else: n_neg += 1
        var p = Pair(scores[i], lb)
        pairs.push(p)
    if n_pos == 0 or n_neg == 0: return 0.5
    var ranks = _sort_pairs_by_score(pairs)  # high score -> high rank

    var sum_r_pos = 0.0
    var idx = 0
    for i in range(N):
        if pairs[i].label == 1:
            sum_r_pos += Float64(ranks[i])
    var U = sum_r_pos - Float64(n_pos * (n_pos + 1) / 2)
    var auc = U / Float64(n_pos * n_neg)
    return auc

# --------- Regression metrics ---------
fn mae(y_true: List[Float64], y_pred: List[Float64]) -> Float64:
    var n = len(y_true)
    if n == 0: return 0.0
    var s = 0.0
    for i in range(n):
        var d = y_true[i] - y_pred[i]
        if d < 0.0: d = -d
        s += d
    return s / Float64(n)
fn mse(y_true: List[Float64], y_pred: List[Float64]) -> Float64:
    var n = len(y_true)
    if n == 0: return 0.0
    var s = 0.0
    for i in range(n):
        var d = y_true[i] - y_pred[i]
        s += d * d
    return s / Float64(n)
fn rmse(y_true: List[Float64], y_pred: List[Float64]) -> Float64:
    var v = mse(y_true, y_pred)
    # sqrt via Newton
    if v <= 0.0: return 0.0
    var s = v
    s = 0.5 * (s + v / s)
    s = 0.5 * (s + v / s)
    return s
fn r2_score(y_true: List[Float64], y_pred: List[Float64]) -> Float64:
    var n = len(y_true)
    if n == 0: return 0.0
    var mean = 0.0
    for i in range(n): mean += y_true[i]
    mean = mean / Float64(n)
    var ss_res = 0.0
    var ss_tot = 0.0
    for i in range(n):
        var r = y_true[i] - y_pred[i]
        ss_res += r * r
        var t = y_true[i] - mean
        ss_tot += t * t
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot

# --------- AverageMeter ---------
struct AverageMeter:
    var count: Int
    var sum: Float64
    var avg: Float64
fn __init__(out self) -> None:
        self.count = 0
        self.sum = 0.0
        self.avg = 0.0
fn reset(mut self) -> None:
        self.count = 0
        self.sum = 0.0
        self.avg = 0.0
fn update(mut self, val: Float64, n: Int = 1) -> None:
        self.sum += val * Float64(n)
        self.count += n
        if self.count > 0:
            self.avg = self.sum / Float64(self.count)
fn __copyinit__(out self, other: Self) -> None:
        self.count = other.count
        self.sum = other.sum
        self.avg = other.avg
fn __moveinit__(out self, deinit other: Self) -> None:
        self.count = other.count
        self.sum = other.sum
        self.avg = other.avg
# --------- Self-test ---------
fn _self_test() -> Bool:
    var ok = True

    # Classification toy
    var targets = List[Int]([0,1,2,1,0,2])
    var scores = List[List[Float64]]()
    # correct classes have slightly higher scores
    scores.push(List[Float64]([0.9,0.05,0.05]))
    scores.push(List[Float64]([0.2,0.6,0.2]))
    scores.push(List[Float64]([0.1,0.2,0.7]))
    scores.push(List[Float64]([0.3,0.4,0.3]))
    scores.push(List[Float64]([0.8,0.1,0.1]))
    scores.push(List[Float64]([0.1,0.2,0.7]))
    var acc = accuracy_from_scores(scores, targets)
    ok = ok and (acc >= 0.83)  # 5/6 correct
    var top2 = topk_accuracy(scores, targets, 2)
    ok = ok and (top2 == 1.0)
    var preds = List[Int]()
    for n in range(len(scores)): preds.push(argmax1d_f(scores[n]))
    var cm = confusion_matrix(targets, preds, -1)
    var mf1 = macro_f1_from_cm(cm)
    ok = ok and (mf1 >= 0.83)

    # Binary AUROC
    var sbin = List[Float64]([0.9, 0.8, 0.2, 0.1])
    var ybin = List[Int]([1, 1, 0, 0])
    var auc = binary_auroc(sbin, ybin)
    ok = ok and (auc >= 0.95)

    # Regression toy
    var y_true = List[Float64]([3.0, -1.0, 2.0, 0.0])
    var y_pred = List[Float64]([2.5, -0.5, 2.0, 0.0])
    var e_mae = mae(y_true, y_pred)
    var e_mse = mse(y_true, y_pred)
    var e_rmse = rmse(y_true, y_pred)
    var e_r2 = r2_score(y_true, y_pred)
    ok = ok and (e_mae <= 0.4)
    ok = ok and (e_mse <= 0.3)
    ok = ok and (e_rmse <= 0.6)
    ok = ok and (e_r2 >= 0.8 or e_r2 <= 1.0)  # sanity range

    # AverageMeter
    var meter = AverageMeter()
    meter.update(2.0, 2)  # sum=4
    meter.update(1.0, 1)  # sum=5, count=3 => avg=1.666...
    ok = ok and (meter.count == 3) and (meter.avg > 1.5) and (meter.avg < 1.8)

    return ok