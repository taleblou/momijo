# Project:      Momijo
# Module:       learn.metrics.classification
# File:         metrics/classification.mojo
# Path:         src/momijo/learn/metrics/classification.mojo
#
# Description:  Classification metrics for Momijo Learn.
#               Provides accuracy (from labels or logits/probabilities) and F1-score
#               for binary and multiclass settings (macro averaging). Backend-agnostic: 
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# License:      MIT License
# SPDX-License-Identifier: MIT
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand
#
# Notes:
#   - Types: accuracy, f1_score (binary & multiclass)
#   - Helpers: argmax_index, confusion_matrix_from_labels
#   - Overloads:
#       accuracy(y_pred_labels: List[Int], y_true: List[Int])
#       accuracy(y_pred_scores: List[List[Float64]], y_true: List[Int])  # logits/probs
#       f1_score(y_pred_labels: List[Int], y_true: List[Int], n_classes: Int)  # macro
#       f1_score_binary(y_pred_scores: List[Float64], y_true: List[Int], threshold: Float64)

from collections.list import List

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

fn _safe_div(num: Float64, den: Float64) -> Float64:
    if den == 0.0:
        return 0.0
    return num / den

fn argmax_index(xs: List[Float64]) -> Int:
    var best_idx = 0
    var best_val = Float64(-1.7976931348623157e308)  # ~-inf
    var i = 0
    while i < len(xs):
        var v = xs[i]
        if v > best_val:
            best_val = v
            best_idx = i
        i = i + 1
    return best_idx

fn _n_classes_from_labels(y_true: List[Int], y_pred: List[Int]) -> Int:
    var max_c = 0
    var i = 0
    while i < len(y_true):
        if y_true[i] > max_c:
            max_c = y_true[i]
        i = i + 1
    i = 0
    while i < len(y_pred):
        if y_pred[i] > max_c:
            max_c = y_pred[i]
        i = i + 1
    return max_c + 1  # classes are assumed 0..C-1

fn confusion_matrix_from_labels(y_pred: List[Int], y_true: List[Int], n_classes: Int) -> List[List[Int]]:
    # Rows = true class, Cols = predicted class
    var cm = List[List[Int]]()
    var r = 0
    while r < n_classes:
        var row = List[Int]()
        var c = 0
        while c < n_classes:
            row.append(0)
            c = c + 1
        cm.append(row)
        r = r + 1
    var n = len(y_true)
    var i = 0
    while i < n:
        var t = y_true[i]
        var p = y_pred[i]
        if 0 <= t and t < n_classes and 0 <= p and p < n_classes:
            cm[t][p] = cm[t][p] + 1
        i = i + 1
    return cm

# -----------------------------------------------------------------------------
# Accuracy
# -----------------------------------------------------------------------------

# Case 1: y_pred are already label indices (0..C-1)
fn accuracy(y_pred: List[Int], y_true: List[Int]) -> Float64:
    var n_true = len(y_true)
    var n_pred = len(y_pred)
    if n_true == 0 or n_pred == 0 or n_true != n_pred:
        return 0.0
    var correct = 0
    var i = 0
    while i < n_true:
        if y_pred[i] == y_true[i]:
            correct = correct + 1
        i = i + 1
    return _safe_div(Float64(correct), Float64(n_true))

# Case 2: y_pred are per-class scores (logits/probs); pick argmax
fn accuracy(y_pred: List[List[Float64]], y_true: List[Int]) -> Float64:
    var n = len(y_true)
    if n == 0 or n != len(y_pred):
        return 0.0
    var preds = List[Int]()
    var i = 0
    while i < n:
        preds.append(argmax_index(y_pred[i]))
        i = i + 1
    return accuracy(preds, y_true)

# -----------------------------------------------------------------------------
# F1-score
# -----------------------------------------------------------------------------

# Binary F1 (y_pred are scores/probabilities for positive class; threshold->labels)
fn f1_score_binary(y_pred: List[Float64], y_true: List[Int], threshold: Float64 = 0.5) -> Float64:
    var n = len(y_true)
    if n == 0 or n != len(y_pred):
        return 0.0

    var tp = 0
    var fp = 0
    var fn_ = 0
    var i = 0
    while i < n:
        var yhat = 0
        if y_pred[i] >= threshold:
            yhat = 1
        var yt = y_true[i]
        if yhat == 1 and yt == 1:
            tp = tp + 1
        elif yhat == 1 and yt == 0:
            fp = fp + 1
        elif yhat == 0 and yt == 1:
            fn_ = fn_ + 1
        i = i + 1

    var precision = _safe_div(Float64(tp), Float64(tp + fp))
    var recall    = _safe_div(Float64(tp), Float64(tp + fn_))
    return _safe_div(2.0 * precision * recall, precision + recall)

# Multiclass F1 macro (y_pred are label indices; requires n_classes)
fn f1_score(y_pred: List[Int], y_true: List[Int], n_classes: Int) -> Float64:
    var n = len(y_true)
    if n == 0 or n != len(y_pred):
        return 0.0
    var cm = confusion_matrix_from_labels(y_pred, y_true, n_classes)

    var sum_f1 = 0.0
    var c = 0
    while c < n_classes:
        var tp = cm[c][c]
        var fp = 0
        var fn_ = 0

        # fp: sum over true rows for column c, except diagonal counted separately
        var r = 0
        while r < n_classes:
            if r != c:
                fp = fp + cm[r][c]
            r = r + 1

        # fn: sum over predicted cols for row c, except diagonal counted separately
        var col = 0
        while col < n_classes:
            if col != c:
                fn_ = fn_ + cm[c][col]
            col = col + 1

        var precision = _safe_div(Float64(tp), Float64(tp + fp))
        var recall    = _safe_div(Float64(tp), Float64(tp + fn_))
        var f1_c = _safe_div(2.0 * precision * recall, precision + recall)
        sum_f1 = sum_f1 + f1_c
        c = c + 1

    return _safe_div(sum_f1, Float64(n_classes))

# Convenience overload: when n_classes is not provided, infer from labels
fn f1_score(y_pred: List[Int], y_true: List[Int]) -> Float64:
    var n = len(y_true)
    if n == 0 or n != len(y_pred):
        return 0.0
    var C = _n_classes_from_labels(y_true, y_pred)
    return f1_score(y_pred, y_true, C)

# Multiclass F1 macro from logits/probabilities (auto-argmax + infer classes)
fn f1_score(y_pred: List[List[Float64]], y_true: List[Int]) -> Float64:
    var n = len(y_true)
    if n == 0 or n != len(y_pred):
        return 0.0
    var labels = List[Int]()
    var i = 0
    while i < n:
        labels.append(argmax_index(y_pred[i]))
        i = i + 1
    return f1_score(labels, y_true)
