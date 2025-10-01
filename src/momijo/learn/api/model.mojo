# Project:      Momijo
# Module:       learn.api.model
# File:         api/model.mojo
# Path:         src/momijo/learn/api/model.mojo
#
# Description:  High-level Model facade (Keras-like) for Momijo Learn.
#               Wraps a low-level `nn.Module` network and provides user-friendly
#               APIs for compile/fit/evaluate/predict, a human-readable summary,
#               and checkpoint save/load helpers that delegate to utils.checkpoint.
#               The implementation is backend-agnostic and can be wired to
#               momijo.tensor ops/autograd as those subsystems mature.
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
#   - Types: Model, CompileConfig
#   - Key fns: compile(...), fit(...), evaluate(...), predict(...),
#              summary(), save(path), load(path)
#   - Dependencies: nn.Module (for the actual network), engine.Trainer (fit loop),
#                   utils.checkpoint (state save/load)
#   - Backend-agnostic: no assumptions about tensor dtype/layout here.

from collections.list import List
from momijo.learn.nn.module import Module
from momijo.learn.engine.trainer import Trainer
from momijo.learn.engine.evaluator import Evaluator
from momijo.learn.utils.checkpoint import (
    save_state_dict,
    load_state_dict,
)

# Optional compile configuration stored on the Model.
struct CompileConfig:
    var optimizer_name: String
    var loss_name: String
    var metric_names: List[String]

    fn __init__(
        out self,
        optimizer_name: String = String(""),
        loss_name: String = String(""),
        metric_names: List[String] = List[String](),
    ):
        self.optimizer_name = optimizer_name
        self.loss_name = loss_name
        self.metric_names = metric_names


# High-level Model facade.
struct Model:
    var name: String
    var net: Module
    var compiled: Bool
    var cfg: CompileConfig
    var training: Bool

    # Construct with an underlying network (nn.Module).
    fn __init__(out self, net: Module, name: String = String("Model")):
        self.name = name
        self.net = net
        self.compiled = False
        self.cfg = CompileConfig()
        self.training = True

    # Keras-like compile: record optimizer/loss/metrics by name.
    # Concrete binding of names → actual objects happens in Trainer.
    fn compile(
        mut self,
        optimizer: String,
        loss: String,
        metrics: List[String] = List[String](),
    ):
        self.cfg = CompileConfig(optimizer, loss, metrics)
        self.compiled = True

    # Switch flags (local to facade; underlying Module can also own such flags later).
    fn train(mut self):
        self.training = True

    fn eval(mut self):
        self.training = False

    # Human-readable string summary. Extend as Module gains parameter introspection.
    fn summary(self) -> String:
        var out = String("=== Momijo Model Summary ===\n")
        out += String("Name: ") + self.name + String("\n")
        out += String("Compiled: ") + (String("True") if self.compiled else String("False")) + String("\n")
        if self.compiled:
            out += String("  Optimizer: ") + self.cfg.optimizer_name + String("\n")
            out += String("  Loss:      ") + self.cfg.loss_name + String("\n")
            out += String("  Metrics:   ")
            if self.cfg.metric_names.size() == 0:
                out += String("(none)\n")
            else:
                # join metric names with comma
                for i in range(self.cfg.metric_names.size()):
                    out += self.cfg.metric_names[i]
                    if i + 1 < self.cfg.metric_names.size():
                        out += String(", ")
                out += String("\n")
        # If Module later exposes param counting, include it here.
        out += String("----------------------------\n")
        return out

    # Save/load delegate to utils.checkpoint (which should understand Module.state_dict()).
    fn save(self, path: String):
        save_state_dict(self.net, path)

    fn load(mut self, path: String):
        load_state_dict(self.net, path)

    # Fit/evaluate/predict: thin wrappers around engine components.
    # Trainer is responsible for resolving optimizer/loss/metrics by name.
    fn fit(self, train_loader, epochs: Int = 1, val_loader = None):
        var trainer = Trainer()
        trainer.fit(self.net, train_loader, epochs)

    fn evaluate(self, data_loader):
        var evaluator = Evaluator()
        evaluator.evaluate(self.net, data_loader)

    # Predict: forwards through the underlying Module. Placeholder until tensor I/O stabilizes.
    # Depending on your Module API, you may rename `forward` to `__call__` later.
    fn predict(self, inputs):
        # By convention you'd call: return self.net.forward(inputs)
        # Placeholder; replace once Module.forward is implemented.
        return inputs
