# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.api.model
# File:         src/momijo/learn/api/model.mojo
#
# Description:
#   High-level Keras-like Model facade for Momijo Learn. Wraps a low-level
#   nn.Module network and exposes user-friendly compile/fit/evaluate/predict
#   APIs plus checkpoint save/load helpers. Backend-agnostic; it can be wired to
#   momijo.tensor ops/autograd as those subsystems mature.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

# -----------------------------------------------------------------------------
# Imports (explicit; no wildcards)
# -----------------------------------------------------------------------------
from collections.list import List
from momijo.learn.nn.module import Module
from momijo.learn.engine.trainer import Trainer
from momijo.learn.engine.evaluator import Evaluator
from momijo.learn.utils.checkpoint import save_state_dict
from momijo.learn.utils.checkpoint import load_state_dict

# Tensor (central import per project standard)
from momijo.tensor.tensor import Tensor
from momijo.tensor.dtypes import Float32
from momijo.tensor.dtypes import Float32

# -----------------------------------------------------------------------------
# Compile configuration (names only; Trainer resolves actual objects)
# -----------------------------------------------------------------------------

struct CompileConfig:
    var optimizer_name: String
    var loss_name: String
    var metric_names: List[String]

    fn __init__(
        out self,
        optimizer_name: String = String(""),
        loss_name: String = String(""),
        metric_names: List[String] = List[String]()
    ):
        self.optimizer_name = optimizer_name
        self.loss_name = loss_name
        self.metric_names = metric_names

# -----------------------------------------------------------------------------
# Model facade
# -----------------------------------------------------------------------------

struct Model:
    var name: String
    var net: Module
    var compiled: Bool
    var cfg: CompileConfig
    var training: Bool

    # Default constructor for generic loaders and simple scripts.
    fn __init__(out self):
        self.name = String("Model")
        self.net = Module()
        self.compiled = False
        self.cfg = CompileConfig()
        self.training = True

    fn __init__(out self, net: Module, name: String = String("Model")):
        self.name = name
        self.net = net
        self.compiled = False
        self.cfg = CompileConfig()
        self.training = True

    # ------------------------- configuration -------------------------

    # Keras-like compile: record optimizer/loss/metrics by name.
    # Concrete binding (names -> objects) is handled by Trainer.
    fn compile(
        mut self,
        optimizer: String,
        loss: String,
        metrics: List[String] = List[String]()
    ):
        self.cfg = CompileConfig(optimizer, loss, metrics)
        self.compiled = True

    # Typed-looking hooks so this facade stays compatible with functional API traits.
    # These are tag-only; use compile(...) to set the real names.
    fn set_optimizer(mut self, _optimizer):
        self.cfg.optimizer_name = String("optimizer")
        self.compiled = True

    fn set_loss(mut self, _loss_fn):
        self.cfg.loss_name = String("loss")
        self.compiled = True

    fn set_metrics(mut self, _metrics):
        self.cfg.metric_names = List[String]()
        self.compiled = True

    # Train/eval mode flags on the facade (and potentially on the wrapped Module).
    fn train(mut self):
        self.training = True

    fn eval(mut self):
        self.training = False

    # --------------------------- I/O & state --------------------------

    # Human-readable string summary of the facade + compile config.
    fn summary(self) -> String:
        var out = String("=== Momijo Model Summary ===\n")
        out = out + String("Name: ") + self.name + String("\n")
        out = out + String("Compiled: ") + (String("True") if self.compiled else String("False")) + String("\n")
        if self.compiled:
            out = out + String("  Optimizer: ") + self.cfg.optimizer_name + String("\n")
            out = out + String("  Loss:      ") + self.cfg.loss_name + String("\n")
            out = out + String("  Metrics:   ")
            var msz = len(self.cfg.metric_names)
            if msz == 0:
                out = out + String("(none)\n")
            else:
                var i = 0
                while i < msz:
                    out = out + self.cfg.metric_names[i]
                    if i + 1 < msz:
                        out = out + String(", ")
                    i = i + 1
                out = out + String("\n")
        out = out + String("----------------------------\n")
        return out

    # Provide state_dict/load_state_dict on the facade too, forwarding to Module.
    fn state_dict(self) -> String:
        return self.net.state_dict()

    fn load_state_dict(mut self, state: String):
        self.net.load_state_dict(state)

    # Save/load delegates to utils.checkpoint (centralized MNP format).
    fn save(self, path: String):
        save_state_dict(self.net, path)

    fn load(mut self, path: String):
        load_state_dict(self.net, path)

    # -------------------------- training API -------------------------

    # Fit without validation loader.
    fn fit(self, train_loader, epochs: Int = 1):
        var trainer = Trainer()
        trainer.fit(self.net, train_loader, epochs)

    # Overload: fit with validation loader (if your Trainer supports it).
    fn fit(self, train_loader, epochs: Int, val_loader):
        var trainer = Trainer()
        trainer.fit(self.net, train_loader, epochs, val_loader)

    fn evaluate(self, data_loader):
        var evaluator = Evaluator()
        evaluator.evaluate(self.net, data_loader)

    # -------------------------- inference API ------------------------
    # Forward-style prediction; delegate to Module.forward when available.
    # Overloads for common float dtypes to keep signatures explicit.

    fn forward(self, x: Tensor[Float32]) -> Tensor[Float32]:
        # Delegate to the wrapped module; assumes compatible signature.
        return self.net.forward(x)

    fn forward(self, x: Tensor[Float32]) -> Tensor[Float32]:
        return self.net.forward(x)


    fn forward(self, x: tensor.GradTensor) -> tensor.GradTensor:
        # Delegate to the wrapped module; assumes compatible signature.
        return self.net.forward(x)

    fn forward(self, x: tensor.GradTensor) -> tensor.GradTensor:
        return self.net.forward(x)

    fn predict(self, inputs: Tensor[Float32]) -> Tensor[Float32]:
        return self.forward(inputs)

    fn predict(self, inputs: Tensor[Float32]) -> Tensor[Float32]:
        return self.forward(inputs)

    # ------------------------- stringification -----------------------

    fn __str__(self) -> String:
        var s = String("Model(")
        s = s + String("name=") + self.name + String(", compiled=")
        s = s + (String("True") if self.compiled else String("False")) + String(")")
        return s
