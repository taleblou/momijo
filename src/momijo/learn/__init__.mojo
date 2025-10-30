    # Project:      Momijo
    # Module:       learn.__init__
    # File:         __init__.mojo
    # Path:         src/momijo/learn/__init__.mojo
    #
    # Description:  learn.__init__ — Skeleton implementation for Momijo Learn high-level training API.
    #               This file is part of the scaffolding that mirrors PyTorch and TensorFlow features.
    #               Replace stubs with real logic progressively while keeping API stable.
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
#   - Skeleton placeholders only
#   - Replace with real logic incrementally

from momijo.learn.api.model import Model
from momijo.learn.api.sequential import Sequential
from momijo.learn.api.functional import fit, evaluate, predict, compile_model

from momijo.learn.nn.module import Module
from momijo.learn.nn.layers import Linear, Conv2d, BatchNorm2d, Dropout, Flatten
from momijo.learn.nn.activations import relu, gelu, softmax, sigmoid
from momijo.learn.nn.functional import conv2d, max_pool2d

from momijo.learn.optim.sgd import SGD
from momijo.learn.optim.adamw import AdamW
from momijo.learn.optim.schedulers import StepLR, CosineAnnealingLR

from momijo.learn.losses.classification import cross_entropy
from momijo.learn.losses.regression import mse_loss, mae_loss

from momijo.learn.metrics.classification import accuracy, f1_score
from momijo.learn.metrics.regression import r2_score

from momijo.learn.data.dataset import Dataset, IterableDataset
from momijo.learn.data.dataloader import DataLoader
from momijo.learn.data.sampler import RandomSampler

from momijo.learn.callbacks.early_stopping import EarlyStopping
from momijo.learn.callbacks.model_checkpoint import ModelCheckpoint
from momijo.learn.callbacks.lr_monitor import LRMonitor

from momijo.learn.amp.autocast import autocast, GradScaler

from momijo.learn.engine.trainer import Trainer






# ---------------------------------------------------------------------
# Tensor
# ---------------------------------------------------------------------
# ----------------------------- Core types -------------------------------------
from momijo.tensor.tensor import Tensor 

# ----------------------------- DType utilities --------------------------------
from momijo.tensor.dtype import DType
from momijo.tensor.dtype import DTypeTag
from momijo.tensor.dtype import promote_dtype
from momijo.tensor.dtype import can_cast
from momijo.tensor.dtype import to_code
from momijo.tensor.dtype import from_code
from momijo.tensor.dtype import from_name
from momijo.tensor.dtype import itemsize_for_code  # if needed by callers

# ----------------------------- Creation (Float64 backends) --------------------
# If your codebase currently provides only *_f64 variants, expose ergonomic names:
from momijo.tensor.creation import linspace_int,linspace_f64,linspace_f32
from momijo.tensor.creation import arange_int,arange_f64,arange_f32
from momijo.tensor.creation import zeros,zeros_int,zeros_f64 ,zeros_f32 
from momijo.tensor.creation import ones_int,ones_f64,ones_f32,ones_like
from momijo.tensor.creation import full,empty,empty_f32,empty_f64
from momijo.tensor.creation import eye_int,eye_f64,eye_f32
from momijo.tensor.creation import randperm_int,randperm_f64,randperm_f32
from momijo.tensor.creation import scalar_f64,scalar_f32,scalar_int

# Random helpers (Float64) 
from momijo.tensor.creation import from_list_float64,from_list_float32,from_list_int ,from_list_int32 ,from_list_int16,from_list_bool 
from momijo.tensor.creation import from_2d_list_float64,from_2d_list_float32,from_2d_list_int ,from_2d_list_int32 ,from_2d_list_int16,from_2d_list_bool 
from momijo.tensor.creation import from_3d_list_float64,from_3d_list_float32,from_3d_list_int ,from_3d_list_int32 ,from_3d_list_int16,from_3d_list_bool 

from momijo.tensor.creation import randn,rand,randn_int,randn_f64,randn_f32
from momijo.tensor.creation import zeros_like,normal,full_like
from momijo.tensor.creation import arange_f64 as arange

# ----------------------------- Axis operations --------------------------------
from momijo.tensor.axis import moveaxis
from momijo.tensor.axis import moveaxes  # multi-axis variant if you need it
from momijo.tensor.axis import swapaxes
from momijo.tensor.axis import roll      # (x, shift, axis)
# Optional specialized helpers:
# from momijo.tensor.axis import roll_one, roll_flat, roll_multi

# ----------------------------- Transforms -------------------------------------
# Constant pad (with fast 1D/2D and spatial paths); signature: pad(x, pad_width, fill_value)
from momijo.tensor.transform import pad
from momijo.tensor.transform import fliplr
from momijo.tensor.transform import flipud
from momijo.tensor.transform import sliding_window
from momijo.tensor.transform import sliding_window_step
from momijo.tensor.transform import transpose

# ----------------------------- Math / Reductions ------------------------------
from momijo.tensor.math import mean
# If you later expose sum/min/max generics, re-export them here similarly:
# from momijo.tensor.math import sum,analytic_jacobian,numeric_jacobian,f_vec
# from momijo.tensor.math import min
# from momijo.tensor.math import max

# ----------------------------- Indexing helpers -------------------------------
# Keep only the most-used helpers at the package surface to avoid API bloat.
from momijo.tensor.indexing import slice1d
from momijo.tensor.indexing import slice2d
from momijo.tensor.indexing import head
from momijo.tensor.indexing import tail 
from momijo.tensor.indexing import slice_dim0
from momijo.tensor.indexing import gather
from momijo.tensor.indexing import plane 
from momijo.tensor.indexing import where  
from momijo.tensor.indexing import col  
# ----------------------------- Printing helpers -------------------------------
 
# ----------------------------- Errors (optional) ------------------------------
# Re-export only what external callers might need for error classification.
from momijo.tensor.errors import TensorError
from momijo.tensor.errors import ErrorKind

 


from momijo.tensor.cast import to_string
from momijo.tensor.cast import to_bool
from momijo.tensor.cast import to_int8
from momijo.tensor.cast import to_int16
from momijo.tensor.cast import to_int32
from momijo.tensor.cast import to_int64
from momijo.tensor.cast import to_float32
from momijo.tensor.cast import to_float64


from momijo.tensor.io import save_npz_f64
# from momijo.tensor.io import load_npy
# from momijo.tensor.io import save_npy
# from momijo.tensor.io import load_npz 
# from momijo.tensor.io import save_npz 
# from momijo.tensor.io import save_csv
# from momijo.tensor.io import load_csv

from momijo.tensor.helpers import write_plane
from momijo.tensor.math import and_t,complex,complex_abs,complex_real,complex_imag


from momijo.tensor.broadcast import broadcast_to


from momijo.tensor.transform import reshape,reshape_infer,resize_like_with_pad,reshape_like
from momijo.tensor.transform import stack,cat

from momijo.tensor.io import (
    # CSV
    write_csv_f64, read_csv_f64,
    write_csv_f32, read_csv_f32,
    write_csv_int, read_csv_int,
    CsvOptions,
    # JSON
    write_json_f64, read_json_f64,
    write_json_f32, read_json_f32,
    write_json_int, read_json_int,
    # XML
    write_xml_f64, read_xml_f64,
    write_xml_f32, read_xml_f32,
    write_xml_int, read_xml_int,
)

from momijo.tensor.autograd import GradContext, GradTensor, no_grad_begin, no_grad_end