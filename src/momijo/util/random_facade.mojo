# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.util
# File: src/momijo/util/random_facade.mojo

from random import Random as _StdRandom
from random import choice as _std_choice
from random import randint as _std_randint
from random import random as _std_random
from random import randrange as _std_randrange
from random import seed as _std_seed
from random import shuffle as _std_shuffle

# Public type alias for users who need an RNG instance.


# Provide thin wrappers so call sites remain uniform.
fn randint(a: Int64, b: Int64) -> Int64:
    # inclusive a..b
    return _std_randint(a, b)
fn random() -> Float64:
    # uniform on [0.0, 1.0)
    return _std_random()
fn randrange(start: Int64, stop: Int64, step: Int64 = 1) -> Int64:
    return _std_randrange(start, stop, step)

# Generic shuffle using in-place protocol for lists/vectors.
# Extend constraints as needed across the project.
fn shuffle[T](xs: mut List[T]) -> None:
    _std_shuffle(xs)

fn choice[T](xs: List[T]) -> T:
    return _std_choice(xs)

# Seeding helper. Prefer calling once at program or test entry points.
fn seed(s: UInt64) -> None:
    _std_seed(s)