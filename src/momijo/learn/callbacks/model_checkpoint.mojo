# Project:      Momijo
# Module:       learn.callbacks.model_checkpoint
# File:         callbacks/model_checkpoint.mojo
# Path:         src/momijo/learn/callbacks/model_checkpoint.mojo
#
# Description:  Training checkpoint callback for Momijo Learn.
#               Saves "last" checkpoint each call and (optionally) "best" checkpoint
#               based on a monitored metric with min/max policy. Uses centralized
#               utils.checkpoint.save_state_dict for the on-disk format (MNP).
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
#   - Types: ModelCheckpoint
#   - Key fns: set_monitor(name, mode_max), update_metric(value), save(model, step)
#   - Files:
#       * {dir}/{prefix}_step={step}.mnp         (per-step snapshot)
#       * {dir}/last.mnp                         (rolling last)
#       * {dir}/best_{monitor}.mnp               (best by metric, optional)
#   - Storage: delegates to momijo.learn.utils.checkpoint.save_state_dict

from pathlib.path import Path
from momijo.learn.utils.checkpoint import save_state_dict

struct ModelCheckpoint:
    var dirpath: String
    var filename_prefix: String
    var save_last: Bool

    # Monitoring
    var monitor_name: String
    var mode_max: Bool
    var has_monitor: Bool
    var last_metric_value: Float64
    var has_metric_value: Bool
    var best_metric_value: Float64
    var has_best: Bool

    fn __init__(
        out self,
        dirpath: String,
        filename_prefix: String = String("ckpt"),
        save_last: Bool = True
    ):
        self.dirpath = dirpath
        self.filename_prefix = filename_prefix
        self.save_last = save_last

        self.monitor_name = String("")
        self.mode_max = False
        self.has_monitor = False

        self.last_metric_value = 0.0
        self.has_metric_value = False

        self.best_metric_value = 0.0
        self.has_best = False

        # Ensure directory exists
        var p = Path(dirpath)
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)

    # Configure which metric to monitor and whether larger is better.
    # Example: set_monitor("val_accuracy", True)   # maximize
    #          set_monitor("val_loss", False)      # minimize
    fn set_monitor(mut self, name: String, mode_max: Bool = False):
        self.monitor_name = name
        self.mode_max = mode_max
        self.has_monitor = True
        self.has_metric_value = False
        self.has_best = False

    # Update the current metric value before calling save(...).
    fn update_metric(mut self, value: Float64):
        self.last_metric_value = value
        self.has_metric_value = True

    # Internal: compare current vs best according to mode.
    fn _is_better(self, current: Float64, best: Float64) -> Bool:
        if self.mode_max:
            return current > best
        else:
            return current < best

    # Internal: join path pieces safely.
    fn _join(self, a: String, b: String) -> String:
        var pa = Path(a)
        var pb = pa / b
        return String(pb.as_string())

    # Save a checkpoint for the given step.
    # Always writes a per-step file. If save_last==True, also writes last.mnp.
    # If a monitor is configured and a metric value is available, updates best.
    fn save(mut self, model, step: Int):
        # Per-step filename
        var step_str = String("step=") + String(step)
        var base_name = self.filename_prefix + String("_") + step_str + String(".mnp")
        var per_step_path = self._join(self.dirpath, base_name)

        # Write per-step snapshot
        save_state_dict(model, per_step_path)

        # Rolling "last" checkpoint
        if self.save_last:
            var last_path = self._join(self.dirpath, String("last.mnp"))
            save_state_dict(model, last_path)

        # Best checkpoint by monitored metric
        if self.has_monitor and self.has_metric_value:
            if not self.has_best:
                self.best_metric_value = self.last_metric_value
                self.has_best = True
                var best_name = String("best_") + self.monitor_name + String(".mnp")
                var best_path = self._join(self.dirpath, best_name)
                save_state_dict(model, best_path)
            else:
                if self._is_better(self.last_metric_value, self.best_metric_value):
                    self.best_metric_value = self.last_metric_value
                    var best_name2 = String("best_") + self.monitor_name + String(".mnp")
                    var best_path2 = self._join(self.dirpath, best_name2)
                    save_state_dict(model, best_path2)
