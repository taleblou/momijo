# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.callbacks.early_stopping
# File:         src/momijo/learn/callbacks/early_stopping.mojo
#
# Description:
#   Early stopping callback for Momijo Learn. Monitors a scalar metric
#   and stops training when no improvement is observed for a configured
#   number of epochs. Supports "min"/"max" modes, min_delta, optional
#   baseline, and restoring the best model weights upon stop.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from pathlib.path import Path
from collections.list import List

# -----------------------------------------------------------------------------
# Trait: StatefulModel
# -----------------------------------------------------------------------------
# Contract for models that can be snapshotted and restored.
trait StatefulModel:
    fn state_dict(self) -> String
    fn load_state_dict(mut self, state: String) -> None

# -----------------------------------------------------------------------------
# EarlyStopping
# -----------------------------------------------------------------------------
struct EarlyStopping:
    # Configuration
    var monitor: String               # e.g. "val_loss"
    var mode: String                  # "min" or "max"
    var patience: Int
    var min_delta: Float64
    var restore_best_weights: Bool
    var verbose: Bool
    var baseline: Optional[Float64]

    # Runtime state
    var best_score: Float64
    var best_epoch: Int
    var wait: Int
    var stopped_epoch: Int
    var best_state: Optional[String]  # serialized state_dict()

    fn __init__(
        out self,
        monitor: String = String("val_loss"),
        mode: String = String("min"),
        patience: Int = 10,
        min_delta: Float64 = 0.0,
        baseline: Optional[Float64] = None,
        restore_best_weights: Bool = True,
        verbose: Bool = False
    ):
        # Config
        self.monitor = monitor

        var m = mode
        if not (m == String("min") or m == String("max")):
            m = String("min")
        self.mode = m

        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.baseline = baseline

        # Runtime defaults
        self.best_epoch = -1
        self.wait = 0
        self.stopped_epoch = -1
        self.best_state = None

        # Initialize best_score based on mode & baseline
        if self.baseline is None:
            if self.mode == String("min"):
                self.best_score = 1.0e300   # +infinity sentinel
            else:
                self.best_score = -1.0e300  # -infinity sentinel
        else:
            self.best_score = self.baseline.value()

    # -------------------------------------------------------------------------
    # Configuration helpers (optional setters)
    # -------------------------------------------------------------------------
    fn set_mode(mut self, mode: String):
        if mode == String("min") or mode == String("max"):
            self.mode = mode

    fn set_monitor(mut self, monitor: String):
        self.monitor = monitor

    fn set_min_delta(mut self, v: Float64):
        self.min_delta = v

    fn set_patience(mut self, v: Int):
        self.patience = v

    fn set_baseline(mut self, v: Optional[Float64]):
        self.baseline = v
        if v is None:
            if self.mode == String("min"):
                self.best_score = 1.0e300
            else:
                self.best_score = -1.0e300
        else:
            self.best_score = v.value()

    fn set_restore_best_weights(mut self, flag: Bool):
        self.restore_best_weights = flag

    fn set_verbose(mut self, flag: Bool):
        self.verbose = flag

    # -------------------------------------------------------------------------
    # Reset runtime state (keeps configuration)
    # -------------------------------------------------------------------------
    fn reset(mut self):
        self.best_epoch = -1
        self.wait = 0
        self.stopped_epoch = -1
        self.best_state = None
        if self.baseline is None:
            if self.mode == String("min"):
                self.best_score = 1.0e300
            else:
                self.best_score = -1.0e300
        else:
            self.best_score = self.baseline.value()

    # -------------------------------------------------------------------------
    # Internal: check improvement
    # -------------------------------------------------------------------------
    fn _is_improved(self, current: Float64) -> Bool:
        if self.mode == String("min"):
            # Require decrease by at least min_delta
            return current < (self.best_score - self.min_delta)
        # mode == "max"
        return current > (self.best_score + self.min_delta)

    # -------------------------------------------------------------------------
    # Best state capture/restore
    # -------------------------------------------------------------------------
    fn capture_best(mut self, model: StatefulModel):
        if self.restore_best_weights:
            var s = model.state_dict()
            self.best_state = Optional[String](s)

    fn restore_best(self, model: StatefulModel):
        if self.restore_best_weights:
            if not (self.best_state is None):
                model.load_state_dict(self.best_state.value())

    # -------------------------------------------------------------------------
    # Epoch end hooks
    # -------------------------------------------------------------------------
    # Use this when you DO NOT need to snapshot model weights.
    fn on_epoch_end(mut self, epoch: Int, current_value: Float64) -> Bool:
        if self._is_improved(current_value):
            self.best_score = current_value
            self.best_epoch = epoch
            self.wait = 0
            if self.verbose:
                var msg = String("[EarlyStopping] Improved ") + self.monitor
                msg = msg + String(" = ") + String(self.best_score)
                print(msg)
            return False

        # No improvement
        self.wait = self.wait + 1
        if self.verbose:
            var msg2 = String("[EarlyStopping] No improvement for ")
            msg2 = msg2 + String(self.wait) + String(" epoch(s)")
            print(msg2)

        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            if self.verbose:
                var msg3 = String("[EarlyStopping] Stopping at epoch ") + String(epoch)
                msg3 = msg3 + String("; best epoch=") + String(self.best_epoch)
                msg3 = msg3 + String(", best ") + self.monitor + String("=") + String(self.best_score)
                print(msg3)
            return True

        return False

    # Use this when you WANT to snapshot/restore model weights.
    fn on_epoch_end_with_model(mut self, epoch: Int, current_value: Float64, model: StatefulModel) -> Bool:
        if self._is_improved(current_value):
            self.best_score = current_value
            self.best_epoch = epoch
            self.wait = 0
            self.capture_best(model)
            if self.verbose:
                var msg = String("[EarlyStopping] Improved ") + self.monitor
                msg = msg + String(" = ") + String(self.best_score)
                print(msg)
            return False

        # No improvement
        self.wait = self.wait + 1
        if self.verbose:
            var msg2 = String("[EarlyStopping] No improvement for ")
            msg2 = msg2 + String(self.wait) + String(" epoch(s)")
            print(msg2)

        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.restore_best(model)
            if self.verbose:
                var msg3 = String("[EarlyStopping] Stopping at epoch ") + String(epoch)
                msg3 = msg3 + String("; best epoch=") + String(self.best_epoch)
                msg3 = msg3 + String(", best ") + self.monitor + String("=") + String(self.best_score)
                print(msg3)
            return True

        return False

    # -------------------------------------------------------------------------
    # Queries / summary
    # -------------------------------------------------------------------------
    fn should_stop(self) -> Bool:
        return self.stopped_epoch >= 0

    fn get_best(self) -> (Float64, Int):
        return (self.best_score, self.best_epoch)

    fn summary(self) -> String:
        var s = String("{ 'monitor': '") + self.monitor + String("', 'mode': '") + self.mode + String("'")
        s = s + String(", 'patience': ") + String(self.patience)
        s = s + String(", 'min_delta': ") + String(self.min_delta)
        s = s + String(", 'best_score': ") + String(self.best_score)
        s = s + String(", 'best_epoch': ") + String(self.best_epoch)
        s = s + String(", 'stopped_epoch': ") + String(self.stopped_epoch) + String(" }")
        return s
