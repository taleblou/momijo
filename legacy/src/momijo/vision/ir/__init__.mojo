# MIT License — Short Header
# Project: momijo | Package: vision.ir.__init__
# File: vision/ir/__init__.mojo
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# SPDX-License-Identifier: MIT

# Core API re-exports for vision.ir.__init__

from vision.ir.fusion import _make_resize, _node_kind, _node_oh, _node_ow, _set_nodes
from vision.ir.fusion import fuse_resizes, simplify_convert_color
from vision.ir.ir import add_node, append_resize, append_rgb_to_gray, clear, id, kind
from vision.ir.ir import last_index_of_resize, node_at, node_count, nodes, oh, ow, set_nodes
