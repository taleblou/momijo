# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.engine.trainer
# File:         src/momijo/learn/engine/trainer.mojo
#
# Description:
#   High-level training loop (Trainer) for Momijo Learn. Provides a backend-
#   agnostic fit() routine with optional AMP grad scaling, gradient accumulation,
#   simple callback hooks, and optional LR scheduler stepping. Designed for
#   duck-typed model/optimizer/dataloader facades.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# Notes:
#   - Types: Trainer
#   - Public fns: fit(model, train_loader, epochs=-1, scaler=None)
#   - Internals: _train_one_epoch_(...), _estimate_num_batches(...), callbacks
#   - Duck-typed expectations:
#       model:
#         - forward(x) or __call__(x)
#       optimizer:
#         - step(mut self)
#         - zero_grad(mut self)
#       loss_fn:
#         - call(loss_fn, preds, targets) -> Float64 or tensor-like scalar
#       dataloader:
#         - fields: dataset, batch_size
#         - dataset.__len__() -> Int
#         - (optionally) iteration yielding (inputs, targets)
#       scheduler (optional):
#         - step(mut self)
#   - AMP:
#       Works with learn.amp.GradScaler: scale(loss), step(optimizer, found_inf), update(found_inf).
#
#   - Tensor note:
#       We import `momijo.tensor.tensor` to align with project conventions.
#       The loop is duck-typed and avoids hard binding to specific tensor APIs,
#       so it compiles even when the actual Tensor methods differ.

# Exact, explicit import per Momijo standards (no wildcard).
from momijo.tensor import tensor

# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------

struct Trainer:
    # Configuration
    var epochs: Int
    var grad_accum_steps: Int
    var verbose: Bool

    # Pluggable components (duck-typed; may be None)
    var optimizer
    var loss_fn
    var scheduler
    var callbacks

    fn __init__(
        out self,
        epochs: Int = 1,
        grad_accum_steps: Int = 1,
        verbose: Bool = True
    ):
        self.epochs = epochs
        self.grad_accum_steps = grad_accum_steps
        self.verbose = verbose
        self.optimizer = None
        self.loss_fn = None
        self.scheduler = None
        self.callbacks = None

    # ------------------------------ Setters -----------------------------------

    fn set_optimizer(mut self, optimizer) -> Trainer:
        self.optimizer = optimizer
        return self

    fn set_loss(mut self, loss_fn) -> Trainer:
        self.loss_fn = loss_fn
        return self

    fn set_scheduler(mut self, scheduler) -> Trainer:
        self.scheduler = scheduler
        return self

    fn set_callbacks(mut self, callbacks) -> Trainer:
        self.callbacks = callbacks
        return self

    # ------------------------------ Public API --------------------------------

    # High-level fit loop. If epochs > 0 is given, it overrides the configured epochs.
    # `scaler` is expected to be a learn.amp.GradScaler or None (duck-typed).
    fn fit(
        mut self,
        model,
        train_loader,
        epochs: Int = -1,
        scaler = None
    ):
        var num_epochs = self.epochs
        if epochs > 0:
            num_epochs = epochs

        self._on_fit_start()

        var e: Int = 0
        while e < num_epochs:
            self._on_epoch_start(e)

            self._train_one_epoch_(model, train_loader, scaler, e)

            if self.scheduler is not None:
                self.scheduler.step()

            self._on_epoch_end(e, String("OK"))
            e = e + 1

        self._on_fit_end()

    # ---------------------------- Internal logic ------------------------------

    fn _train_one_epoch_(mut self, model, train_loader, scaler, epoch: Int):
        var batches: Int = self._estimate_num_batches(train_loader)

        if self.verbose:
            print(String("Epoch ") + String(epoch) + String(" — batches: ") + String(batches))

        if self.optimizer is None and self.verbose:
            print(String("[Trainer] Warning: optimizer is None; updates will be skipped."))
        if self.loss_fn is None and self.verbose:
            print(String("[Trainer] Warning: loss_fn is None; using dummy loss."))

        if self.optimizer is not None:
            self.optimizer.zero_grad()

        # If your DataLoader exposes an iterator/protocol, adapt this block as needed.
        # Duck-typed expectations:
        #   - it.next() -> (inputs, targets)
        #   - model.forward(inputs) or model(inputs)
        #   - loss_fn(preds, targets) -> Float64 or tensor-like scalar
        var step: Int = 0
        while step < batches:
            self._on_batch_start(step)

            # -------------------- Fetch batch (duck-typed) --------------------
            # Placeholder: when no iterator is exposed, users can inject a facade
            # with a `get_batch(i)` method. We try that path first.
            var inputs
            var targets
            var has_batch: Bool = False

            if train_loader is not None:
                # Try duck-typed get_batch(step)
                if train_loader.get_batch is not None:
                    var pair = train_loader.get_batch(step)  # expect (inputs, targets)
                    inputs = pair[0]
                    targets = pair[1]
                    has_batch = True

            if not has_batch:
                # Fallback dummy batch for demonstration (keeps loop valid)
                # Users should provide a working DataLoader in real training.
                inputs = None
                targets = None

            # -------------------- Forward & Loss ------------------------------
            var loss_val: Float64 = 1.0  # default dummy
            if self.loss_fn is not None and model is not None and has_batch:
                # Try calling model; prefer forward(x) if present; else call
                var preds
                if model.forward is not None:
                    preds = model.forward(inputs)
                else:
                    preds = model(inputs)

                var raw_loss = self.loss_fn(preds, targets)

                # Attempt to scalarize the loss for logging/AMP scaling.
                # We keep this conservative to avoid binding to specific tensor APIs.
                loss_val = self._coerce_loss_to_f64(raw_loss)

            # -------------------- AMP scale (optional) ------------------------
            var scaled_loss: Float64 = loss_val
            if scaler is not None:
                scaled_loss = scaler.scale(loss_val)

            # ---------------- Gradient accumulation & Optimizer ---------------
            var do_step: Bool = False
            if self.grad_accum_steps <= 1:
                do_step = True
            else:
                var next_step = step + 1
                if (next_step % self.grad_accum_steps) == 0:
                    do_step = True

            if do_step and self.optimizer is not None:
                var found_inf: Bool = False  # hook for overflow detection
                if scaler is not None:
                    scaler.step(self.optimizer, found_inf)
                    scaler.update(found_inf)
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            self._on_batch_end(step, String("loss=") + String(loss_val))
            step = step + 1

    fn _estimate_num_batches(self, train_loader) -> Int:
        var n: Int = 0
        var bs: Int = 0

        if train_loader is not None:
            # Expect `train_loader.dataset.__len__()` and `train_loader.batch_size`.
            if train_loader.dataset is not None:
                n = train_loader.dataset.__len__()
            bs = train_loader.batch_size

        if bs <= 0: return 0
        if n  <= 0: return 0

        var full: Int = n // bs
        var rem: Int = n - (full * bs)
        if rem > 0:
            return full + 1
        return full

    # ----------------------------- Loss helpers -------------------------------

    # Conservative loss scalarization to Float64 for logging/AMP.
    # We avoid hard-coding tensor methods (like .item()/.mean()) to keep it duck-typed.
    fn _coerce_loss_to_f64(self, loss_any) -> Float64:
        # Common cases first
        if loss_any is Float64:
            return loss_any
        if loss_any is Float32:
            return Float64(loss_any)
        # If a tensor-like scalar is passed, try a duck-typed `to_f64()` adapter
        # expected in user space; otherwise fallback to 1.0 to keep loop alive.
        if loss_any.to_f64 is not None:
            return loss_any.to_f64()
        return 1.0

    # ------------------------------ Callbacks ---------------------------------

    fn _on_fit_start(self):
        if self.callbacks is not None and self.callbacks.on_fit_start is not None:
            self.callbacks.on_fit_start()

    fn _on_fit_end(self):
        if self.callbacks is not None and self.callbacks.on_fit_end is not None:
            self.callbacks.on_fit_end()

    fn _on_epoch_start(self, epoch: Int):
        if self.callbacks is not None and self.callbacks.on_epoch_start is not None:
            self.callbacks.on_epoch_start(epoch)

    fn _on_epoch_end(self, epoch: Int, logs: String):
        if self.callbacks is not None and self.callbacks.on_epoch_end is not None:
            self.callbacks.on_epoch_end(epoch, logs)

    fn _on_batch_start(self, step: Int):
        if self.callbacks is not None and self.callbacks.on_batch_start is not None:
            self.callbacks.on_batch_start(step)

    fn _on_batch_end(self, step: Int, logs: String):
        if self.callbacks is not None and self.callbacks.on_batch_end is not None:
            self.callbacks.on_batch_end(step, logs)
