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
