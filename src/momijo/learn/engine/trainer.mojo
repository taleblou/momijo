# Project:      Momijo
# Module:       learn.engine.trainer
# File:         engine/trainer.mojo
# Path:         src/momijo/learn/engine/trainer.mojo
#
# Description:  High-level training loop (Trainer) for Momijo Learn. Provides
#               a backend-agnostic fit() routine with optional AMP grad scaling,
#               gradient accumulation, simple hooks, and scheduler stepping.
#               Designed to work with duck-typed model/optimizer/dataloader
#               following Momijo Learn facades.
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
#   - Types: Trainer
#   - Key fns: fit(), train_one_epoch_()
#   - Duck-typed expectations:
#       model:
#         - forward(x) or __call__(x)          (optional; loop does not call by default)
#         - state_dict()/load_state_dict()     (optional; for checkpoints outside)
#       optimizer:
#         - step(mut self)
#         - zero_grad(mut self)
#       loss_fn:
#         - call(loss_fn, preds, targets) -> Float64 or tensor-like scalar
#       dataloader:
#         - fields: dataset, batch_size
#         - dataset.__len__() -> Int
#         - optional iteration protocol to yield (inputs, targets)
#       scheduler (optional):
#         - step(mut self)
#
#   - AMP:
#       Integrates with learn.amp.GradScaler: scale(loss), step(...), update(...).
#       Caller provides scaler or None; when provided, it will be used.

# A minimal callback interface (duck-typed). Users can pass any object that
# implements these methods (all optional):
#   on_fit_start(), on_fit_end()
#   on_epoch_start(epoch), on_epoch_end(epoch, logs: String)
#   on_batch_start(step), on_batch_end(step, logs: String)

struct Trainer:
    # Configuration
    var epochs: Int
    var grad_accum_steps: Int
    var verbose: Bool

    # Pluggable components (duck-typed)
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
        # Components are assigned later by setters or fit() parameters.
        self.optimizer = None
        self.loss_fn = None
        self.scheduler = None
        self.callbacks = None

    # --- Setters (fluent-style) ------------------------------------------------

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

    # --- Public API ------------------------------------------------------------

    # Fit loop with optional AMP scaler.
    # The function accepts `epochs` to override configured self.epochs, if desired.
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

        var e = 0
        while e < num_epochs:
            self._on_epoch_start(e)

            self._train_one_epoch_(model, train_loader, scaler, e)

            # Step LR scheduler if provided
            if self.scheduler is not None:
                self.scheduler.step()

            self._on_epoch_end(e, String("OK"))
            e = e + 1

        self._on_fit_end()

    # --- Internal helpers ------------------------------------------------------

    fn _train_one_epoch_(mut self, model, train_loader, scaler, epoch: Int):
        # Since current DataLoader/Dataset are stubs, we compute an estimated
        # number of batches from __len__()/batch_size if available.
        var batches: Int = self._estimate_num_batches(train_loader)

        if self.verbose:
            print(String("Epoch ") + String(epoch) + String(" — batches: ") + String(batches))

        # Ensure optimizer exists
        if self.optimizer is None:
            # No-op but keeps loop consistent
            if self.verbose:
                print(String("[Trainer] Warning: optimizer is None; skipping updates."))

        # Ensure loss_fn exists
        if self.loss_fn is None:
            if self.verbose:
                print(String("[Trainer] Warning: loss_fn is None; loss will be undefined."))

        # Reset gradients at epoch start
        if self.optimizer is not None:
            self.optimizer.zero_grad()

        var step: Int = 0
        while step < batches:
            self._on_batch_start(step)

            # In a real pipeline, we'd fetch a batch:
            # var inputs, targets = next(train_iter)
            # var preds = model.forward(inputs)
            # var loss = self.loss_fn(preds, targets)
            #
            # For now, we synthesize a dummy scalar "loss" for scaffolding.
            var loss: Float64 = 1.0

            # AMP scale if provided
            if scaler is not None:
                loss = scaler.scale(loss)

            # Gradient accumulation stub:
            # In real code, we'd backprop each mini-loss and only call optimizer.step()
            # every grad_accum_steps. Here, we call step directly at accumulation boundary.

            var do_step: Bool = False
            if self.grad_accum_steps <= 1:
                do_step = True
            else:
                if ((step + 1) % self.grad_accum_steps) == 0:
                    do_step = True

            if do_step and self.optimizer is not None:
                var found_inf: Bool = False  # Hook to overflow detector (future)
                if scaler is not None:
                    scaler.step(self.optimizer, found_inf)
                    scaler.update(found_inf)
                else:
                    self.optimizer.step()

                # Zero gradients after update
                self.optimizer.zero_grad()

            self._on_batch_end(step, String("loss=") + String(loss))
            step = step + 1

    fn _estimate_num_batches(self, train_loader) -> Int:
        # Try to derive from dataset length and batch_size; stubs fallback to 0.
        var n: Int = 0
        var bs: Int = 0

        # dataset length
        if train_loader is not None:
            # Expect: train_loader.dataset.__len__() and train_loader.batch_size
            n = train_loader.dataset.__len__()
            bs = train_loader.batch_size

        if bs <= 0:
            return 0
        if n <= 0:
            return 0

        var full: Int = n // bs
        var rem: Int = n - (full * bs)
        if rem > 0:
            return full + 1
        return full

    # --- Callbacks (all optional) ----------------------------------------------

    fn _on_fit_start(self):
        if self.callbacks is not None:
            if self.callbacks.on_fit_start is not None:
                self.callbacks.on_fit_start()

    fn _on_fit_end(self):
        if self.callbacks is not None:
            if self.callbacks.on_fit_end is not None:
                self.callbacks.on_fit_end()

    fn _on_epoch_start(self, epoch: Int):
        if self.callbacks is not None:
            if self.callbacks.on_epoch_start is not None:
                self.callbacks.on_epoch_start(epoch)

    fn _on_epoch_end(self, epoch: Int, logs: String):
        if self.callbacks is not None:
            if self.callbacks.on_epoch_end is not None:
                self.callbacks.on_epoch_end(epoch, logs)

    fn _on_batch_start(self, step: Int):
        if self.callbacks is not None:
            if self.callbacks.on_batch_start is not None:
                self.callbacks.on_batch_start(step)

    fn _on_batch_end(self, step: Int, logs: String):
        if self.callbacks is not None:
            if self.callbacks.on_batch_end is not None:
                self.callbacks.on_batch_end(step, logs)
