# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo
# File: src/momijo/learn/__init__.mojo
# Description: Public re-exports for Momijo Learn (deduplicated, clean surface).

# -----------------------------------------------------------------------------
# High-level APIs
# -----------------------------------------------------------------------------
from momijo.learn.api.model import Model
from momijo.learn.api.sequential import Sequential
from momijo.learn.api.functional import compile_model, fit, evaluate, predict

# -----------------------------------------------------------------------------
# Core NN building blocks
# -----------------------------------------------------------------------------
from momijo.learn.nn.module import Module

# Layers (keep a single canonical source for each symbol)
from momijo.learn.nn.layers import Linear
from momijo.learn.nn.layers import ReLU
from momijo.learn.nn.layers import LeakyReLU
from momijo.learn.nn.layers import Sigmoid
from momijo.learn.nn.layers import Tanh
from momijo.learn.nn.layers import BatchNorm1d
from momijo.learn.nn.layers import Dropout
# If Flatten exists in layers, export it here (no duplicates elsewhere)
# from momijo.learn.nn.layers import Flatten

# Convolutions / pooling (use conv/* modules as the single source of truth)
from momijo.learn.nn.conv import Conv2d, MaxPool2d
from momijo.learn.nn.avgpool import AvgPool2d
from momijo.learn.nn.conv_transpose import ConvTranspose2d
from momijo.learn.nn.layernorm import LayerNorm1d

# Functional ops (single import site)
from momijo.learn.nn.functional import conv2d, max_pool2d

# Activations (export softmax from activations ONLY to avoid collision with losses)
from momijo.learn.nn.activations import relu, gelu, sigmoid, softmax 
 # from momijo.learn.nn.activations import log_softmax

# -----------------------------------------------------------------------------
# Losses (do NOT re-export softmax/log_softmax here to avoid name clashes)
# -----------------------------------------------------------------------------
from momijo.learn.losses.classification import cross_entropy
from momijo.learn.losses.regression import mse_loss, mae_loss
# Additional loss helpers (unique names; safe to export)
from momijo.learn.losses.losses import softmax_cross_entropy
from momijo.learn.losses.losses import cross_entropy_from_logits
from momijo.learn.losses.losses import cross_entropy_from_probs
# (NOTE) We intentionally do NOT import `softmax` or `log_softmax` from losses.

# -----------------------------------------------------------------------------
# Optimizers & Schedulers (single source per symbol)
# -----------------------------------------------------------------------------
from momijo.learn.optim.sgd import SGD
from momijo.learn.optim.adamw import AdamW
from momijo.learn.optim.rmsprop import RMSprop

from momijo.learn.optim.scheduler_steplr import StepLR
from momijo.learn.optim.scheduler_multistep import MultiStepLR
from momijo.learn.optim.scheduler_cosine import CosineAnnealingLR
# (NOTE) We do NOT import a second scheduler bundle to avoid redefinitions.

# -----------------------------------------------------------------------------
# Data API
# -----------------------------------------------------------------------------
from momijo.learn.data.dataset import Dataset, IterableDataset
from momijo.learn.data.dataloader import DataLoader, DataLoaderOptions
from momijo.learn.data.sampler import RandomSampler

# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
from momijo.learn.metrics.metrics import accuracy_approx
from momijo.learn.metrics.classification import accuracy, f1_score
from momijo.learn.metrics.regression import r2_score

# -----------------------------------------------------------------------------
# Training utilities / callbacks / AMP / engine
# -----------------------------------------------------------------------------
from momijo.learn.callbacks.early_stopping import EarlyStopping
from momijo.learn.callbacks.model_checkpoint import ModelCheckpoint
from momijo.learn.callbacks.lr_monitor import LRMonitor

from momijo.learn.amp.autocast import autocast, GradScaler

from momijo.learn.engine.trainer import Trainer

# -----------------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------------
from momijo.learn.utils.checkpoint import make_checkpoint, apply_checkpoint
from momijo.learn.utils.checkpoint_fileio import save_checkpoint_files, load_checkpoint_files
from momijo.learn.utils.tensor_bytes import (
    pack_f64_to_bytes_binary as pack_f64,
    unpack_bytes_to_f64_binary as unpack_f64,
)


from momijo.learn.distributed.env import get_rank_world
from momijo.learn.distributed.ddp import (
    ProcessGroup,
    init_process_group,
    destroy_process_group,
    DDPLinear,
)
# -----------------------------------------------------------------------------
# Tensor facade re-exports
# -----------------------------------------------------------------------------
from momijo.tensor.tensor import Tensor

# DType utilities
from momijo.tensor.dtype import DType
from momijo.tensor.dtype import DTypeTag
from momijo.tensor.dtype import promote_dtype
from momijo.tensor.dtype import can_cast
from momijo.tensor.dtype import to_code
from momijo.tensor.dtype import from_code
from momijo.tensor.dtype import from_name
from momijo.tensor.dtype import itemsize_for_code

# Creation
from momijo.tensor.creation import linspace_int, linspace_f64, linspace_f32
from momijo.tensor.creation import arange_int, arange_f64, arange_f32
from momijo.tensor.creation import zeros, zeros_int, zeros_f64, zeros_f32
from momijo.tensor.creation import ones_int, ones_f64, ones_f32, ones_like
from momijo.tensor.creation import full, empty, empty_f32, empty_f64
from momijo.tensor.creation import eye_int, eye_f64, eye_f32
from momijo.tensor.creation import randperm_int, randperm_f64, randperm_f32
from momijo.tensor.creation import scalar_f64, scalar_f32, scalar_int

from momijo.tensor.creation import (
    from_list_float64, from_list_float32, from_list_int, from_list_int32, from_list_int16, from_list_bool,
)
from momijo.tensor.creation import (
    from_2d_list_float64, from_2d_list_float32, from_2d_list_int, from_2d_list_int32, from_2d_list_int16, from_2d_list_bool,
)
from momijo.tensor.creation import (
    from_3d_list_float64, from_3d_list_float32, from_3d_list_int, from_3d_list_int32, from_3d_list_int16, from_3d_list_bool,
)

from momijo.tensor.creation import randn, rand, randn_int, randn_f64, randn_f32
from momijo.tensor.creation import zeros_like, normal, full_like
from momijo.tensor.creation import arange_f64 as arange

# Axis ops
from momijo.tensor.axis import moveaxis, moveaxes, swapaxes, roll

# Transforms
from momijo.tensor.transform import pad, fliplr, flipud, sliding_window, sliding_window_step, transpose

# Math
from momijo.tensor.math import mean
from momijo.tensor.math import and_t, complex, complex_abs, complex_real, complex_imag

# Indexing
from momijo.tensor.indexing import slice1d, slice2d, head, tail, slice_dim0, gather, plane, where, col

# Broadcast / reshape / stacking
from momijo.tensor.broadcast import broadcast_to
from momijo.tensor.transform import reshape, reshape_infer, resize_like_with_pad, reshape_like, stack, cat

# Casting
from momijo.tensor.cast import to_string, to_bool, to_int8, to_int16, to_int32, to_int64, to_float32, to_float64

# I/O
from momijo.tensor.io import save_npz_f64
from momijo.tensor.io import (
    write_csv_f64, read_csv_f64,
    write_csv_f32, read_csv_f32,
    write_csv_int, read_csv_int,
    CsvOptions,
    write_json_f64, read_json_f64,
    write_json_f32, read_json_f32,
    write_json_int, read_json_int,
    write_xml_f64, read_xml_f64,
    write_xml_f32, read_xml_f32,
    write_xml_int, read_xml_int,
)

# Autograd
from momijo.tensor.autograd import GradContext, GradTensor, no_grad_begin, no_grad_end
