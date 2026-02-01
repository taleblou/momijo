# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       momijo.learn.data.image
# File:         src/momijo/learn/data/image.mojo
#
# Description:
#   ImageFolder-style dataset for PNG images.
#   Expects: root/<class_name>/*.png
#   Returns: Pair(x, y) where x is CHW Float32 (normalized by your Compose),
#   and y is Int class index.
#
# Dependencies:
#   - momijo.vision.io.read_image : Path -> Tensor[Float32] (HW or HWC, range 0..255)
#   - momijo.vision.transforms.compose.Compose : callable pipeline returning CHW Float32
#   - tensor ops are handled by the transform pipeline.

from collections.list import List
from pathlib import Path
from os import listdir
from momijo.tensor import tensor
from momijo.vision.io import read_image
from momijo.learn.data.pair import Pair
from momijo.vision.transforms.compose import Compose
from momijo.vision.image import Image
from momijo.vision.convert import to_tensor_float32

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
fn _to_lower_ascii(s: String) -> String:
    var out = String("")
    var n = len(s)
    var i = 0
    while i < n:
        var ch = s[i]
        # naive ASCII lower: 'A'(65) .. 'Z'(90) -> +32
        try:
            var code = Int(ch)
            if code >= 65 and code <= 90:
                code = code + 32
            out = out + String(code)
        except:
            pass

        i = i + 1
    return out

fn _ends_with_ignore_case(name: String, suffix: String) -> Bool:
    var n = len(name)
    var m = len(suffix)
    if m > n:
        return False
    var a = _to_lower_ascii(name[n - m:n])
    var b = _to_lower_ascii(suffix)
    return a == b

fn _join(root: String, name: String) -> String:
    # naive join; fine for our use
    return root + String("/") + name

fn _is_dir(path: String) -> Bool:
    # Heuristic: if we can list it, it's a directory
    try:
        _ = listdir(path)
        return True
    except:
        return False

fn list_dirs_sorted(root: String) -> List[String]:
    var names = List[String]()
    try:
        for nm in listdir(root):                       # nm: String
            if _is_dir(_join(root, nm)):
                names.append(nm)                       # keep just the name
    except:
        pass

    # insertion sort (lexicographic)
    var n = len(names)
    var i = 1
    while i < n:
        var key = names[i]
        var j = i - 1
        while j >= 0 and names[j] > key:
            names[j + 1] = names[j]
            j = j - 1
        names[j + 1] = key
        i = i + 1

    return names.copy()

# If you prefer full paths instead of names:
fn list_dirs_sorted_paths(root: String) -> List[String]:
    var out = List[String]()
    for nm in list_dirs_sorted(root):
        out.append(_join(root, nm))
    return out.copy()

fn list_pngs_sorted(root: String) -> List[String]:
    var names = List[String]()

    try:
        for nm in listdir(root):
            if _ends_with_ignore_case(nm, String(".png")):
                names.append(nm)
    except:
        pass

    # insertion sort by filename (simple lexicographic compare)
    var m = len(names)
    var i = 1
    while i < m:
        var key = names[i]
        var j = i - 1
        while j >= 0 and names[j] > key:
            names[j + 1] = names[j]
            j = j - 1
        names[j + 1] = key
        i = i + 1

    # return full paths
    var out = List[String]()
    for nm in names:
        out.append(root + String("/") + nm)
    return out.copy()

# -----------------------------------------------------------------------------
# ImageFolder Dataset
# -----------------------------------------------------------------------------
struct ImageFolderDataset(Copyable, Movable):
    var files: List[String]
    var labels: List[Int]
    var tf: Compose[tensor.Tensor[Float32]]
    var class_names: List[String]   # index -> class name

    fn __init__(out self, root: String, tf: Compose[tensor.Tensor[Float32]]):
        self.files = List[String]()
        self.labels = List[Int]()
        self.tf = tf.copy()
        self.class_names = List[String]()
        self._scan(root)

    fn __copyinit__(out self, other: Self):
        self.files = other.files.copy()
        self.labels = other.labels.copy()
        self.tf = other.tf.copy()
        self.class_names = other.class_names.copy()

    fn __len__(self) -> Int:
        return len(self.files)

    fn num_classes(self) -> Int:
        return len(self.class_names)

    fn class_name(self, idx: Int) -> String:
        var n = len(self.class_names)
        if idx < 0 or idx >= n: return String("")
        return self.class_names[idx]



    fn __getitem__(self, idx: Int) -> Pair:
        var n = len(self.files)
        if n == 0:
            var z = tensor.zeros([1, 1, 1]).add_scalar(Float32(0.0))
            return Pair.from_xy(z, 0)

        var i = idx
        if i < 0: i = 0
        if i >= n: i = n - 1

        var p = self.files[i]
        var y = self.labels[i]

        var x: tensor.Tensor[Float32] = tensor.zeros([1,1,1]).add_scalar(Float32(0.0))

        try:
            var req = read_image(p)      # Tuple[Bool, Image]
            if req[0]:
                var img_chw = to_tensor_float32(req[1]._tensor.copy())  # ✅ هِلپر جدید
                x = self.tf(img_chw)                     # ترنسفورم‌ها (نرمال‌سازی و ...)
            else:
                print(String("read_image failed for: ") + p)
                x = self.tf(x)      # روی پیش‌فرض اعمال کن تا نوع/شکل ثابت بماند
        except e:
            print(String("read_image exception for: ") + p)
            print(e)
            x = self.tf(x)

        return Pair.from_xy(x, y)


    fn _scan(mut self, root: String):
        var names = list_dirs_sorted(root)                 # فقط اسم‌ها
        self.class_names = List[String]()

        var cid = 0
        while cid < len(names):
            var name = names[cid]                          # مثل "0", "1", ...
            self.class_names.append(name)                  # کلاس‌نیم برای نمایش

            var full = _join(root, name)                   # مسیر کامل
            var pngs = list_pngs_sorted(full)              # ✅ اینجا مسیر کامل بده
            var j = 0
            while j < len(pngs):
                self.files.append(pngs[j])
                self.labels.append(cid)
                j = j + 1
            cid = cid + 1





#  fn initial(self):
#         self._scan(root)

#         var n = len(self.files)
#         if n == 0:
#             var z = tensor.zeros([1, 1, 1]).add_scalar(0.0)
#             return Pair.from_xy(z, 0)

#         var cid = 0
#         while cid < n:
#             var p = self.files[cid]
#             var y = self.labels[cid]
#             var x: tensor.Tensor[Float32] = tensor.zeros([1,1,1]).add_scalar(0.0)
#             try:
#                 var req = read_image(p)      # Tuple[Bool, Image]
#                 if req[0]:
#                     var img_chw = to_tensor_float64(req[1]._tensor.copy())  # ✅ هِلپر جدید
#                     x = self.tf(img_chw)                     # ترنسفورم‌ها (نرمال‌سازی و ...)
#                 else:
#                     print(String("read_image failed for: ") + p)
#                     x = self.tf(x)      # روی پیش‌فرض اعمال کن تا نوع/شکل ثابت بماند
#             except e:
#                 print(String("read_image exception for: ") + p)
#                 print(e)
#                 x = self.tf(x)

#             var pngs = list_pngs_sorted(cdir)
#             var j = 0
#             while j < len(pngs):
#                 self.files.append(pngs[j])
#                 self.labels.append(cid)
#                 j = j + 1
#             cid = cid + 1

#         var i = idx
#         if i < 0: i = 0
#         if i >= n: i = n - 1

#         var p = self.files[i]
#         var y = self.labels[i]

#         var x: tensor.Tensor[Float32] = tensor.zeros([1,1,1]).add_scalar(0.0)

#         try:
#             var req = read_image(p)      # Tuple[Bool, Image]
#             if req[0]:
#                 var img_chw = to_tensor_float64(req[1]._tensor.copy())  # ✅ هِلپر جدید
#                 x = self.tf(img_chw)                     # ترنسفورم‌ها (نرمال‌سازی و ...)
#             else:
#                 print(String("read_image failed for: ") + p)
#                 x = self.tf(x)      # روی پیش‌فرض اعمال کن تا نوع/شکل ثابت بماند
#         except e:
#             print(String("read_image exception for: ") + p)
#             print(e)
#             x = self.tf(x)

#         return Pair.from_xy(x, y)
