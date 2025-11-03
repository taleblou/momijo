# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo
# File: src/momijo/learn/nn/module.mojo
# Description: Tagged-union Module wrapper for heterogeneous layers.

from collections.list import List
from momijo.tensor import tensor
from momijo.learn.nn.layers import Linear, ReLU, LeakyReLU, Sigmoid, Tanh, BatchNorm1d, Dropout
from momijo.learn.nn.conv import Conv2d, MaxPool2d



# -----------------------------
# Base Module
# -----------------------------
struct Module(Copyable, Movable):
    # -------- Variant/tag dispatch (functional "kind" of this module)
    # 0:Linear,1:ReLU,2:Leaky,3:Sigmoid,4:Tanh,5:BN1d,6:Dropout,7:Conv2d,8:MaxPool2d, -1:Generic/container
    var tag: Int
    var linear: Linear
    var relu: ReLU
    var lrelu: LeakyReLU
    var sigmoid: Sigmoid
    var tanh: Tanh
    var bn1d: BatchNorm1d
    var dropout: Dropout
    var conv2d: Conv2d
    var maxpool2d: MaxPool2d

    # -------- Identity / mode
    var name: String
    var training: Bool

    # -------- Hierarchy
    var child_names: List[String]
    var children: List[Module]

    # -------- Learnable params
    var param_names: List[String]
    var param_values: List[tensor.Tensor[Float64]]
    var param_requires_grad: List[Bool]

    # -------- Non-learnable buffers
    var buffer_names: List[String]
    var buffer_values: List[tensor.Tensor[Float64]]

    # --------------------------------
    # Constructors
    # --------------------------------
    fn __init__(
        out self,
        name: String = String(""),
        tag: Int = -1,
        linear: Linear = Linear(1, 1),
        relu: ReLU = ReLU(),
        lrelu: LeakyReLU = LeakyReLU(),
        sigmoid: Sigmoid = Sigmoid(),
        tanh: Tanh = Tanh(),
        bn1d: BatchNorm1d = BatchNorm1d(1),
        dropout: Dropout = Dropout(),
        conv2d: Conv2d = Conv2d(1, 1, (1, 1)),
        maxpool2d: MaxPool2d = MaxPool2d((1, 1))
    ):
        # tag + variants
        self.tag = tag
        self.linear = linear.copy(); self.relu = relu.copy(); self.lrelu = lrelu.copy()
        self.sigmoid = sigmoid.copy(); self.tanh = tanh.copy()
        self.bn1d = bn1d.copy(); self.dropout = dropout.copy()
        self.conv2d = conv2d.copy(); self.maxpool2d = maxpool2d.copy()

        # identity/mode
        self.name = name
        self.training = True

        # hierarchy/params/buffers
        self.child_names = List[String]()
        self.children = List[Module]()

        self.param_names = List[String]()
        self.param_values = List[tensor.Tensor[Float64]]()
        self.param_requires_grad = List[Bool]()

        self.buffer_names = List[String]()
        self.buffer_values = List[tensor.Tensor[Float64]]()

    fn __copyinit__(out self, other: Self):
        # tag + variants
        self.tag = other.tag
        self.linear = other.linear.copy()
        self.relu = other.relu.copy()
        self.lrelu = other.lrelu.copy()
        self.sigmoid = other.sigmoid.copy()
        self.tanh = other.tanh.copy()
        self.bn1d = other.bn1d.copy()
        self.dropout = other.dropout.copy()
        self.conv2d = other.conv2d.copy()
        self.maxpool2d = other.maxpool2d.copy()

        # identity/mode
        self.name = other.name
        self.training = other.training

        # deep-ish copies of lists (Mojo List is value type with copyable elems)
        self.child_names = other.child_names.copy()
        self.children = other.children.copy()

        self.param_names = other.param_names.copy()
        self.param_values = other.param_values.copy()
        self.param_requires_grad = other.param_requires_grad.copy()

        self.buffer_names = other.buffer_names.copy()
        self.buffer_values = other.buffer_values.copy()

    # --------------------------------
    # Static constructors for tagged kinds
    # --------------------------------
    @staticmethod
    fn from_linear(l: Linear, name: String = String("Linear")) -> Self:
        return Self(name, 0, l, ReLU(), LeakyReLU(), Sigmoid(), Tanh(), BatchNorm1d(1), Dropout(), Conv2d(1,1,(1,1)), MaxPool2d((1,1)))

    @staticmethod
    fn from_relu(r: ReLU, name: String = String("ReLU")) -> Self:
        return Self(name, 1, Linear(1,1), r, LeakyReLU(), Sigmoid(), Tanh(), BatchNorm1d(1), Dropout(), Conv2d(1,1,(1,1)), MaxPool2d((1,1)))

    @staticmethod
    fn from_leaky_relu(r: LeakyReLU, name: String = String("LeakyReLU")) -> Self:
        return Self(name, 2, Linear(1,1), ReLU(), r, Sigmoid(), Tanh(), BatchNorm1d(1), Dropout(), Conv2d(1,1,(1,1)), MaxPool2d((1,1)))

    @staticmethod
    fn from_sigmoid(s: Sigmoid, name: String = String("Sigmoid")) -> Self:
        return Self(name, 3, Linear(1,1), ReLU(), LeakyReLU(), s, Tanh(), BatchNorm1d(1), Dropout(), Conv2d(1,1,(1,1)), MaxPool2d((1,1)))

    @staticmethod
    fn from_tanh(t: Tanh, name: String = String("Tanh")) -> Self:
        return Self(name, 4, Linear(1,1), ReLU(), LeakyReLU(), Sigmoid(), t, BatchNorm1d(1), Dropout(), Conv2d(1,1,(1,1)), MaxPool2d((1,1)))

    @staticmethod
    fn from_batchnorm1d(b: BatchNorm1d, name: String = String("BatchNorm1d")) -> Self:
        return Self(name, 5, Linear(1,1), ReLU(), LeakyReLU(), Sigmoid(), Tanh(), b, Dropout(), Conv2d(1,1,(1,1)), MaxPool2d((1,1)))

    @staticmethod
    fn from_dropout(d: Dropout, name: String = String("Dropout")) -> Self:
        return Self(name, 6, Linear(1,1), ReLU(), LeakyReLU(), Sigmoid(), Tanh(), BatchNorm1d(1), d, Conv2d(1,1,(1,1)), MaxPool2d((1,1)))

    @staticmethod
    fn from_conv2d(c: Conv2d, name: String = String("Conv2d")) -> Self:
        return Self(name, 7, Linear(1,1), ReLU(), LeakyReLU(), Sigmoid(), Tanh(), BatchNorm1d(1), Dropout(), c, MaxPool2d((1,1)))

    @staticmethod
    fn from_maxpool2d(p: MaxPool2d, name: String = String("MaxPool2d")) -> Self:
        return Self(name, 8, Linear(1,1), ReLU(), LeakyReLU(), Sigmoid(), Tanh(), BatchNorm1d(1), Dropout(), Conv2d(1,1,(1,1)), p)

    # --------------------------------
    # Forward (dispatch by tag)
    # --------------------------------
    fn forward(mut self, x: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
        if self.tag == 0: return self.linear.forward(x)
        if self.tag == 1: return self.relu.forward(x)
        if self.tag == 2: return self.lrelu.forward(x)
        if self.tag == 3: return self.sigmoid.forward(x)
        if self.tag == 4: return self.tanh.forward(x)
        if self.tag == 5: return self.bn1d.forward(x)
        if self.tag == 6: return self.dropout.forward(x)
        if self.tag == 7: return self.conv2d.forward(x)
        if self.tag == 8: return self.maxpool2d.forward(x)
        # container/generic: forward through children in order if any
        var out = x.copy()
        var i = 0
        while i < len(self.children):
            var m = self.children[i].copy()
            out = m.forward(out)
            self.children[i]= m.copy()
            i = i + 1
        return out.copy()

    fn forward(mut self, mut ctx: GradContext, x: GradTensor) -> GradTensor:
        if self.tag == 0: return self.linear.forward(ctx, x)
        if self.tag == 1: return self.relu.forward(ctx, x)
        if self.tag == 2: return self.lrelu.forward(ctx, x)
        if self.tag == 3: return self.sigmoid.forward(ctx, x)
        if self.tag == 4: return self.tanh.forward(ctx, x)
        if self.tag == 5: return self.bn1d.forward(ctx, x)
        if self.tag == 6: return self.dropout.forward(ctx, x)
        if self.tag == 7: return self.conv2d.forward(ctx, x)
        if self.tag == 8: return self.maxpool2d.forward(ctx, x)

        # container/generic: forward through children in order if any
        var out = x.copy()
        var i = 0
        while i < len(self.children):
            # اگر children حاوی ماژول‌های همین نوع هستند:
            out = self.children[i].forward(ctx, out)
            i = i + 1
        return out.copy()
    # --------------------------------
    # Composition / registration
    # --------------------------------


    fn add_module(mut self, name: String, m: Module):
        var n = len(self.child_names)
        var i = 0
        while i < n:
            if self.child_names[i] == name:
                self.children[i] = m
                return
            i = i + 1
        self.child_names.append(name)
        self.children.append(m)


    fn modules(self) -> List[Module]:
        return self.children

    fn register_parameter(mut self, name: String, value: tensor.Tensor[Float64], requires_grad: Bool = True):
        var i = 0
        while i < len(self.param_names):
            if self.param_names[i] == name:
                self.param_values.__setitem__(i, value)
                self.param_requires_grad.__setitem__(i, requires_grad)
                return
            i = i + 1
        self.param_names.append(name)
        self.param_values.append(value)
        self.param_requires_grad.append(requires_grad)

    fn register_buffer(mut self, name: String, value: tensor.Tensor[Float64]):
        var i = 0
        while i < len(self.buffer_names):
            if self.buffer_names[i] == name:
                self.buffer_values.__setitem__(i, value)
                return
            i = i + 1
        self.buffer_names.append(name)
        self.buffer_values.append(value)

    fn add_parameter(mut self, name: String, value: tensor.Tensor[Float64], requires_grad: Bool = True):
        self.register_parameter(name, value, requires_grad)

    fn add_buffer(mut self, name: String, value: tensor.Tensor[Float64]):
        self.register_buffer(name, value)

    # --------------------------------
    # Train / eval
    # --------------------------------
    fn train(mut self) -> Module:
        self.training = True
        var i = 0
        while i < len(self.children):
            var c = self.children[i].copy()
            c.train()
            self.children.__setitem__(i, c)
            i = i + 1
        return self

    fn eval(mut self) -> Module:
        self.training = False
        var i = 0
        while i < len(self.children):
            var c = self.children[i].copy()
            c.eval()
            self.children.__setitem__(i, c)
            i = i + 1
        return self

    fn is_training(self) -> Bool:
        return self.training

    # --------------------------------
    # Accessors
    # --------------------------------
    fn parameters(self) -> List[tensor.Tensor[Float64]]:
        return self.param_values

    fn buffers(self) -> List[tensor.Tensor[Float64]]:
        return self.buffer_values

    # --------------------------------
    # Introspection
    # --------------------------------
    fn named_parameters(self, prefix: String = String("")) -> List[String]:
        var out = List[String]()

        var i = 0
        while i < len(self.param_names):
            var key = self.param_names[i]
            var full = prefix
            if prefix.__len__() > 0:
                full += String(".")
            full += key
            out.append(full)
            i = i + 1

        var j = 0
        while j < len(self.children):
            var cname = self.child_names[j]
            var child = self.children[j]
            var child_prefix = prefix
            if child_prefix.__len__() > 0:
                child_prefix += String(".")
            child_prefix += cname

            var sub = child.named_parameters(child_prefix)
            var k = 0
            while k < len(sub):
                out.append(sub[k])
                k = k + 1
            j = j + 1
        return out

    fn named_buffers(self, prefix: String = String("")) -> List[String]:
        var out = List[String]()

        var i = 0
        while i < len(self.buffer_names):
            var key = self.buffer_names[i]
            var full = prefix
            if prefix.__len__() > 0:
                full += String(".")
            full += key
            out.append(full)
            i = i + 1

        var j = 0
        while j < len(self.children):
            var cname = self.child_names[j]
            var child = self.children[j]
            var child_prefix = prefix
            if child_prefix.__len__() > 0:
                child_prefix += String(".")
            child_prefix += cname

            var sub = child.named_buffers(child_prefix)
            var k = 0
            while k < len(sub):
                out.append(sub[k])
                k = k + 1
            j = j + 1
        return out

    # --------------------------------
    # Serialization (simple JSON-ish view)
    # --------------------------------
    fn _pairs_flat(self, prefix: String = String("")) -> List[String]:
        var lines = List[String]()

        # params
        var i = 0
        while i < len(self.param_names):
            var k = self.param_names[i].copy()
            var v = self.param_values[i].copy()
            var full = prefix
            if prefix.__len__() > 0:
                full += String(".")
            full += k
            var line = String("\"") + _json_escape(full) + String("\":\"") + _json_escape(_tensor_summary(v)) + String("\"")
            lines.append(line)
            i = i + 1

        # buffers
        var b = 0
        while b < len(self.buffer_names):
            var k2 = self.buffer_names[b].copy()
            var v2 = self.buffer_values[b].copy()
            var full2 = prefix
            if prefix.__len__() > 0:
                full2 += String(".")
            full2 += k2
            var line2 = String("\"") + _json_escape(full2) + String("\":\"") + _json_escape(_tensor_summary(v2)) + String("\"")
            lines.append(line2)
            b = b + 1

        # recurse
        var j = 0
        while j < len(self.children):
            var cname = self.child_names[j].copy()
            var child = self.children[j].copy()
            var child_prefix = prefix
            if child_prefix.__len__() > 0:
                child_prefix += String(".")
            child_prefix += cname

            var sub = child._pairs_flat(child_prefix)
            var k = 0
            while k < len(sub):
                lines.append(sub[k])
                k = k + 1
            j = j + 1

        return lines.copy()

    fn state_dict(self) -> String:
        var pairs = self._pairs_flat(String(""))
        var sb = String("{")
        var i = 0
        while i < len(pairs):
            sb += pairs[i]
            if i + 1 < len(pairs):
                sb += String(",")
            i = i + 1
        sb += String("}")
        return sb

    fn load_state_dict(mut self, state: String):
        # Placeholder: validate braces only; real parser to be added when tensor IO stabilizes.
        if state.__len__() < 2: return
        var first = state[0]
        var last = state[state.__len__() - 1]
        if not (first == String("{") and last == String("}")): return
        return



# -----------------------------
# Minimal JSON string escaping
# -----------------------------
fn _json_escape(s: String) -> String:
    var out = String("")
    var i = 0
    while i < s.__len__():
        var ch = s[i]
        if ch == String("\\"):
            out += String("\\\\")
        elif ch == String("\""):
            out += String("\\\"")
        elif ch == String("\n"):
            out += String("\\n")
        elif ch == String("\r"):
            out += String("\\r")
        elif ch == String("\t"):
            out += String("\\t")
        else:
            out += ch
        i = i + 1
    return out

# -----------------------------
# Small tensor summary (metadata-only)
# -----------------------------
fn _tensor_summary(t: tensor.Tensor[Float64]) -> String:
    # NOTE: print only scalars/String; do not dump raw buffers here.
    var s = String("tensor(")
    s += String("dtype=Float64")
    s += String(", shape=[")
    var nd = t.ndim()
    var d = 0
    while d < nd:
        s += String(t.shape().__str__())
        if d + 1 < nd:
            s += String(",")
        d = d + 1
    s += String("])")
    return s




struct Seq512(Copyable, Movable):
    var l1: Linear
    var a: ReLU
    var l2: Linear

    fn __init__(out self):
        self.l1 = Linear(512, 512)
        self.a  = ReLU()
        self.l2 = Linear(512, 512)

    fn __copyinit__(out self, other: Self):
        self.l1 = other.l1.copy()
        self.a  = other.a.copy()
        self.l2 = other.l2.copy()

    fn forward(self, x: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
        var h = self.l1.forward(x)
        h = self.a.forward(h)
        return self.l2.forward(h)
