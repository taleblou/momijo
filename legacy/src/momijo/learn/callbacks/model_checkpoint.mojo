# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.callbacks.model_checkpoint
# File:         src/momijo/learn/callbacks/model_checkpoint.mojo
#
# Description:
#   Training checkpoint callback for Momijo Learn.
#   Writes a per-step snapshot, an optional rolling "last" checkpoint, and an optional
#   "best" checkpoint based on a monitored metric with min/max policy. Persists using
#   momijo.learn.utils.checkpoint.save_state_dict (MNP format).
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from pathlib.path import Path
from momijo.learn.utils.checkpoint import save_state_dict

# -----------------------------------------------------------------------------
# ModelCheckpoint
# -----------------------------------------------------------------------------

struct ModelCheckpoint:
    # I/O configuration
    var dirpath: String
    var filename_prefix: String
    var save_last: Bool

    # Monitoring configuration/state
    var monitor_name: String
    var mode_max: Bool
    var has_monitor: Bool

    var last_metric_value: Float32
    var has_metric_value: Bool

    var best_metric_value: Float32
    var has_best: Bool

    # Optional bookkeeping (paths of last written files)
    var last_path_written: String
    var best_path_written: String

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

        self.last_path_written = String("")
        self.best_path_written = String("")

        # Try to if destination directory exists (best-effort, no hard failure).
        _ensure_dir(self.dirpath)

    # Configure which metric to monitor and whether larger is better.
    # Example:
    #   set_monitor("val_accuracy", True)  # maximize
    #   set_monitor("val_loss", False)     # minimize
    fn set_monitor(mut self, name: String, mode_max: Bool = False):
        self.monitor_name = name
        self.mode_max = mode_max
        self.has_monitor = True
        # Reset metric/best state when monitor changes
        self.has_metric_value = False
        self.has_best = False
        self.best_path_written = String("")

    # Update the current metric value prior to save(...).
    fn update_metric(mut self, value: Float32):
        self.last_metric_value = value
        self.has_metric_value = True

    # Save a checkpoint for the given step:
    #   - always writes a per-step file
    #   - optionally writes "last.mnp"
    #   - if monitor is set and metric is available, updates "best"
    fn save(mut self, model, step: Int):
        # Compose per-step filename: {prefix}_step={step}.mnp
        var step_str = String("step=") + String(step)
        var base_name = self.filename_prefix + String("_") + step_str + String(".mnp")
        var per_step_path = _join(self.dirpath, base_name)

        # Persist per-step snapshot
        save_state_dict(model, per_step_path)
        self.last_path_written = per_step_path

        # Rolling "last" checkpoint
        if self.save_last:
            var last_path = _join(self.dirpath, String("last.mnp"))
            save_state_dict(model, last_path)
            self.last_path_written = last_path  # keep pointer to the rolling file

        # Best checkpoint by monitored metric
        if self.has_monitor and self.has_metric_value:
            if not self.has_best:
                # First metric becomes the current best
                self.best_metric_value = self.last_metric_value
                self.has_best = True
                var best_name = String("best_") + self.monitor_name + String(".mnp")
                var best_path = _join(self.dirpath, best_name)
                save_state_dict(model, best_path)
                self.best_path_written = best_path
            else:
                if _is_better(self.mode_max, self.last_metric_value, self.best_metric_value):
                    self.best_metric_value = self.last_metric_value
                    var best_name2 = String("best_") + self.monitor_name + String(".mnp")
                    var best_path2 = _join(self.dirpath, best_name2)
                    save_state_dict(model, best_path2)
                    self.best_path_written = best_path2

# -----------------------------------------------------------------------------
# Internal helpers (file system safe-guards and comparisons)
# -----------------------------------------------------------------------------

@always_inline
fn _is_better(mode_max: Bool, current: Float32, best: Float32) -> Bool:
    # Returns True if current is better than best under the selected mode.
    if mode_max:
        return current > best
    else:
        return current < best

@always_inline
fn _join(a: String, b: String) -> String:
    # OS-agnostic path join via pathlib.
    var pa = Path(a)
    var pb = pa / b
    return String(pb.as_string())

fn _ensure_dir(dirpath: String):
    # Best-effort directory creation without raising; avoids keyword-arg features.
    var p = Path(dirpath)
    if p.exists():
        return
    # Try a single-level mkdir; swallow errors to keep training running.
    try:
        p.mkdir()
    except _:
        # As a fallback, attempt to create parent(s) in a simple loop.
        # If pathlib in your Mojo supports parents, replace this logic as needed.
        # Here we simply do nothing further to avoid throwing in environments
        # that restrict filesystem operations.
        pass
