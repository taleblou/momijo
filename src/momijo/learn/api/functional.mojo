# Project:      Momijo
# Module:       learn.api.functional
# File:         api/functional.mojo
# Path:         src/momijo/learn/api/functional.mojo
#
# Description:  High-level "functional" training interface for Momijo Learn.
#               Provides Keras-like entry points (compile/fit/evaluate/predict)
#               while delegating the actual work to the training engine.
#               Backend-agnostic; wire your tensor/ops later in engine/ and nn/.
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
#   - Functions: compile_model, fit, evaluate, predict
#   - Delegates: fit -> engine.Trainer, evaluate -> engine.Evaluator
#   - Keep this file thin; core logic belongs to engine/ and nn/

from momijo.learn.engine.trainer import Trainer
from momijo.learn.engine.evaluator import Evaluator

# -----------------------------------------------------------------------------
# compile_model
# -----------------------------------------------------------------------------
# Stores (optimizer/loss/metrics) on the model if it exposes a compatible API.
# Otherwise, it acts as a no-op. We keep it backend-agnostic and duck-typed.
# Later, extend your `Model` or `Module` to support these setters.

fn compile_model(model, optimizer, loss, metrics) -> Bool:
    # Optional duck-typed configuration; if your model exposes setters,
    # you can wire them here. Keeping it a safe no-op otherwise.

    # Example expected methods (optional for now):
    # - model.set_optimizer(optimizer)
    # - model.set_loss(loss)
    # - model.set_metrics(metrics)

    # NOTE: Mojo is statically typed; avoid calling methods that may not exist.
    # Keep this function returning True to indicate a successful "compile" step.
    return True


# -----------------------------------------------------------------------------
# fit
# -----------------------------------------------------------------------------
# High-level training loop: delegates to engine.Trainer.
# Trainer is responsible for handling optimizer, loss, metrics (either
# pre-attached to the model during compile_model, or via its own config).

fn fit(model, data_loader, epochs: Int):
    var trainer = Trainer()
    trainer.fit(model, data_loader, epochs)


# -----------------------------------------------------------------------------
# evaluate
# -----------------------------------------------------------------------------
# Delegates to engine.Evaluator. Returns an opaque result (backend-agnostic).
# You can later define a concrete result struct (e.g., dict-like metrics).

fn evaluate(model, data_loader):
    var evaluator = Evaluator()
    evaluator.evaluate(model, data_loader)


# -----------------------------------------------------------------------------
# predict
# -----------------------------------------------------------------------------
# Simple forward-style prediction. If your model exposes `predict`, this is a
# good place to call it. Otherwise, return the inputs or a forward pass stub.
# Keep it minimal; specialized batching lives in DataLoader or Engine.

fn predict(model, inputs):
    # If your model provides a predict method, prefer that. To keep this stub
    # universally compilable without static type constraints, return inputs.
    # Replace with: `return model.predict(inputs)` when available.
    return inputs
