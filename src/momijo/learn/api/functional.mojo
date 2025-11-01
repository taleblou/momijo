# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.api.functional
# File:         src/momijo/learn/api/functional.mojo
#
# Description:
#   High-level functional training interface for Momijo Learn.
#   Provides Keras-like entry points (compile/fit/evaluate/predict) while
#   delegating the core loop/mechanics to the training engine.
#   Includes typed traits so models can opt-in to static, safe integration.

from collections.list import List
from momijo.learn.engine.trainer import Trainer
from momijo.learn.engine.evaluator import Evaluator

# -----------------------------------------------------------------------------
# Traits (opt-in static integration)
# -----------------------------------------------------------------------------
# A model can implement these traits to enable typed overloads below.

trait TrainableModel:
    fn set_optimizer(mut self, optimizer)
    fn set_loss(mut self, loss_fn)
    fn set_metrics(mut self, metrics)
    fn state_dict(self) -> String
    fn load_state_dict(mut self, state: String)

trait InferenceModel:
    fn forward(self, x)
    fn __str__(self) -> String

trait PredictableModel:
    fn predict(self, x)

# -----------------------------------------------------------------------------
# Options / Results (engine-agnostic containers)
# -----------------------------------------------------------------------------

struct FitOptions:
    var epochs: Int
    var steps_per_epoch: Int
    var validation_steps: Int
    var callbacks: List[Any]
    var log_every_n_steps: Int

    fn __init__(
        out self,
        epochs: Int = 1,
        steps_per_epoch: Int = 0,
        validation_steps: Int = 0,
        callbacks: List[Any] = List[Any](),
        log_every_n_steps: Int = 0
    ):
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.callbacks = callbacks
        self.log_every_n_steps = log_every_n_steps

struct FitResult:
    var epochs_trained: Int
    var final_step: Int
    var has_history: Bool
    var history_keys: List[String]

    fn __init__(out self):
        self.epochs_trained = 0
        self.final_step = 0
        self.has_history = False
        self.history_keys = List[String]()

struct EvalOptions:
    var steps: Int
    fn __init__(out self, steps: Int = 0):
        self.steps = steps

struct EvalResult:
    var ok: Bool
    fn __init__(out self, ok: Bool = True):
        self.ok = ok

struct PredictOptions:
    var batch_size: Int
    var steps: Int
    fn __init__(out self, batch_size: Int = 0, steps: Int = 0):
        self.batch_size = batch_size
        self.steps = steps

# -----------------------------------------------------------------------------
# compile_model
# -----------------------------------------------------------------------------

# Generic, safe no-op (always returns True).
fn compile_model(model, optimizer, loss, metrics) -> Bool:
    return True

# Typed overload: models that implement TrainableModel receive optimizer/loss/metrics.
fn compile_model(model: TrainableModel, optimizer, loss, metrics) -> Bool:
    var m = model
    m.set_optimizer(optimizer)
    m.set_loss(loss)
    m.set_metrics(metrics)
    return True

# -----------------------------------------------------------------------------
# fit
# -----------------------------------------------------------------------------

# Minimal wrapper over Trainer; returns a generic FitResult.
fn fit(model, data_loader, epochs: Int) -> FitResult:
    var trainer = Trainer()
    trainer.fit(model, data_loader, epochs)
    var r = FitResult()
    r.epochs_trained = epochs
    return r

# Typed overload using options and optional validation loader.
fn fit_with_options(model, train_loader, val_loader, opts: FitOptions) -> FitResult:
    var trainer = Trainer()
    if val_loader is None:
        trainer.fit(model, train_loader, opts.epochs)
    else:
        # If your Trainer has an overload accepting val_loader, this call binds to it.
        trainer.fit(model, train_loader, opts.epochs, val_loader)
    var r = FitResult()
    r.epochs_trained = opts.epochs
    return r

# Typed model overloads (compile-time safety when your model implements TrainableModel)
fn fit(model: TrainableModel, data_loader, epochs: Int) -> FitResult:
    var trainer = Trainer()
    trainer.fit(model, data_loader, epochs)
    var r = FitResult()
    r.epochs_trained = epochs
    return r

fn fit_with_options(model: TrainableModel, train_loader, val_loader, opts: FitOptions) -> FitResult:
    var trainer = Trainer()
    if val_loader is None:
        trainer.fit(model, train_loader, opts.epochs)
    else:
        trainer.fit(model, train_loader, opts.epochs, val_loader)
    var r = FitResult()
    r.epochs_trained = opts.epochs
    return r

# -----------------------------------------------------------------------------
# evaluate
# -----------------------------------------------------------------------------

fn evaluate(model, data_loader) -> EvalResult:
    var evaluator = Evaluator()
    evaluator.evaluate(model, data_loader)
    return EvalResult(True)

fn evaluate_with_options(model, data_loader, opts: EvalOptions) -> EvalResult:
    var evaluator = Evaluator()
    if opts.steps > 0:
        evaluator.evaluate(model, data_loader, opts.steps)
    else:
        evaluator.evaluate(model, data_loader)
    return EvalResult(True)

# Typed overloads for TrainableModel (no change in behavior; ensures static checks).
fn evaluate(model: TrainableModel, data_loader) -> EvalResult:
    var evaluator = Evaluator()
    evaluator.evaluate(model, data_loader)
    return EvalResult(True)

fn evaluate_with_options(model: TrainableModel, data_loader, opts: EvalOptions) -> EvalResult:
    var evaluator = Evaluator()
    if opts.steps > 0:
        evaluator.evaluate(model, data_loader, opts.steps)
    else:
        evaluator.evaluate(model, data_loader)
    return EvalResult(True)

# -----------------------------------------------------------------------------
# predict
# -----------------------------------------------------------------------------

# Generic safe stub: returns inputs; keeps file compilable for arbitrary models.
fn predict(model, inputs):
    return inputs

fn predict_with_options(model, inputs, opts: PredictOptions):
    return predict(model, inputs)

# Typed overload: prefer model.predict when available.
fn predict(model: PredictableModel, inputs):
    return model.predict(inputs)

# Fallback typed overload for models that only implement forward.
fn predict(model: InferenceModel, inputs):
    return model.forward(inputs)

fn predict_with_options(model: PredictableModel, inputs, opts: PredictOptions):
    # Options can be used by higher layers for batching; here we forward directly.
    return model.predict(inputs)

fn predict_with_options(model: InferenceModel, inputs, opts: PredictOptions):
    return model.forward(inputs)


# ----------------------------- INT8 helpers (PTQ) -----------------------------
@always_inline
fn _clamp_int8(v: Int32) -> Int8:
    var t = v
    if t < -128: t = -128
    if t > 127: t = 127
    return Int8(t)
@always_inline
fn _round_to_i32(v: Float64) -> Int32:
    # Round half away from zero using add/sub 0.5 then trunc-to-zero cast.
    var adj = 0.5 if v >= 0.0 else -0.5
    return Int32(v + adj)

@always_inline
fn _numel_shape(shp: List[Int]) -> Int:
    var n = 1
    var i = 0
    var L = len(shp)
    while i < L:
        n = n * shp[i]
        i = i + 1
    return n

# ------------------------ quant/dequant kernels (fixed) ------------------------

fn quantize_symmetric_int8(x: tensor.Tensor[Float64], s: Float64) -> tensor.Tensor[Int8]:
    # y_q = clamp(round(x / s), int8)
    # Guard s to avoid div by zero.
    var scale = s
    if scale == 0.0:
        scale = 1.0

    var shp = x.shape()
    var n = _numel_shape(shp)

    var out = tensor.Tensor[Int8](shp, Int8(0))
    var i = 0
    while i < n:
        var q = _round_to_i32(x._data[i] / scale)
        out._data[i] = _clamp_int8(q)
        i = i + 1
    return out.copy()

fn matmul_i8_i8_to_i32(a: tensor.Tensor[Int8], bt: tensor.Tensor[Int8]) -> tensor.Tensor[Int32]:
    # a: [N,K], bt: [K,M]  (b transposed)
    var ash = a.shape()
    var bsh = bt.shape()
    if len(ash) != 2 or len(bsh) != 2:
        # Fallback: return empty [0,0]
        return tensor.Tensor[Int32]([0, 0], Int32(0))

    var N = ash[0]
    var K = ash[1]
    var KM = bsh[0]
    var M = bsh[1]
    if K != KM:
        # Incompatible shapes; return empty
        return tensor.Tensor[Int32]([0, 0], Int32(0))

    var y = tensor.Tensor[Int32]([N, M], Int32(0))

    var i = 0
    while i < N:
        var j = 0
        while j < M:
            var acc = Int32(0)
            var p = 0
            while p < K:
                acc = acc + Int32(a._data[i * K + p]) * Int32(bt._data[p * M + j])
                p = p + 1
            y._data[i * M + j] = acc
            j = j + 1
        i = i + 1
    return y.copy()

fn dequantize_i32_to_f32_add_bias(
    y_i32: tensor.Tensor[Int32],
    sf: Float64,
    b: tensor.Tensor[Float64]
) -> tensor.Tensor[Float64]:
    # y_f32 = y_i32 * sf + b  (bias broadcast along rows)
    var ysh = y_i32.shape()
    if len(ysh) != 2:
        return tensor.Tensor[Float64]([0, 0], 0.0)

    var N = ysh[0]
    var M = ysh[1]
    var out = tensor.Tensor[Float64]([N, M], 0.0)

    var i = 0
    while i < N:
        var j = 0
        while j < M:
            out._data[i * M + j] = Float64(y_i32._data[i * M + j]) * sf + b._data[j]
            j = j + 1
        i = i + 1
    return out.copy()   