# ============================================================================
# MIT License
# (c) 2025 Morteza Talebou & Mitra Daneshmand
# Project: momijo | momijo.ir
# File: src/momijo/codegen/codegen_cuda.mojo
# Description:
#   CUDA C code generator for Momijo IR (enhanced):
#     - vec.add/sub/mul/div.{i64|f64}  → scalar + vectorized (2/4 lanes) kernels
#     - vec.relu.{i64|f64}, vec.sigmoid.f64 (unary)
#     - vec.fused.{i64|f64} with expr_rpn & arity (scalar + 2/4 lanes)
#     - vec.reduce_sum.{i64|f64} (atomicAdd accumulation)
#     - ll/hl.conv2d (NHWC x HWCO) → naïve kernel (strides, padding=valid|same)
#   For vector width > 1, host launchers auto-pick 4→2→1 depending on divisibility.
#   All functions are emitted in a single translation unit with extern "C".
# Notes:
#   - Self-contained fallback; no external deps beyond CUDA headers.
#   - For double atomicAdd requires cc>=6.0; otherwise provide a CAS fallback macro.
# ============================================================================

# -----------------------------
# Minimal IR model
# -----------------------------

struct Location:
    var filename: String
    var line: Int
    var col: Int
    fn __init__(out self, filename: String = String("<unknown>"), line: Int = 0, col: Int = 0):
        self.filename = filename
        self.line = line
        self.col = col

struct TypeDesc:
    var name: String
    fn __init__(out self, name: String): self.name = name

@register_passable
struct Value:
    var id: Int
    var typ: TypeDesc
    fn __init__(out self, id: Int, typ: TypeDesc):
        self.id = id
        self.typ = typ

struct Result:
    var value: Value
    fn __init__(out self, value: Value): self.value = value

struct Operand:
    var value: Value
    fn __init__(out self, value: Value): self.value = value

struct Op:
    var name: String
    var results: List[Result]
    var operands: List[Operand]
    var attrs: List[String]
    var loc: Location

    fn __init__(out self, name: String, loc: Location):
        self.name = name
        self.results = List[Result]()
        self.operands = List[Operand]()
        self.attrs = List[String]()
        self.loc = loc

struct Block:
    var name: String
    var args: List[Value]
    var ops: List[Op]

    fn __init__(out self, name: String):
        self.name = name
        self.args = List[Value]()
        self.ops = List[Op]()

struct Function:
    var name: String
    var arg_types: List[TypeDesc]
    var ret_types: List[TypeDesc]
    var blocks: List[Block]

    fn __init__(out self, name: String):
        self.name = name
        self.arg_types = List[TypeDesc]()
        self.ret_types = List[TypeDesc]()
        self.blocks = List[Block]()

struct Module:
    var name: String
    var functions: List[Function]
    fn __init__(out self, name: String = String("module")):
        self.name = name
        self.functions = List[Function]()

# -----------------------------
# Type/attr helpers
# -----------------------------

struct ShapeInfo:
    var kind: String
    var dtype: String
    var dims: List[Int]
    fn __init__(out self):
        self.kind = String("")
        self.dtype = String("")
        self.dims = List[Int]()

fn _parse_int_token(tok: String, out ok: Bool) -> Int:
    ok = False
    if tok.size() == 0: return 0
    if tok == String("?"): ok = True; return -1
    var sign = 1; var i = 0
    if tok.bytes()[0] == 45: sign = -1; i = 1
    var num = 0
    while i < tok.size():
        var ch = tok.bytes()[i]
        if ch < 48 or ch > 57: return 0
        num = num * 10 + Int(ch - 48); i = i + 1
    ok = True; return sign * num

fn _parse_int_list(br: String, out vals: List[Int], out ok: Bool) -> None:
    ok = False; vals = List[Int]()
    if br.size() < 2: ok = True; return
    var s = br; var start = 0; var end = s.size()
    if s.bytes()[0] == 91 and s.bytes()[s.size()-1] == 93: start = 1; end = s.size() - 1
    var tok = String(""); var i = start
    while i < end:
        var ch = s.bytes()[i]
        if ch == 44:
            var ok1 = False; var v = _parse_int_token(tok, ok1)
            if not ok1: return
            vals.push_back(v); tok = String("")
        else:
            tok = tok + String.from_utf8([ch])
        i = i + 1
    if tok.size() > 0:
        var ok2 = False; var v2 = _parse_int_token(tok, ok2)
        if not ok2: return
        vals.push_back(v2)
    ok = True

fn _parse_shape_from_type(t: TypeDesc, out info: ShapeInfo, out ok: Bool) -> None:
    ok = False; info = ShapeInfo()
    var s = t.name
    if not (s.starts_with(String("tensor<")) or s.starts_with(String("buffer<"))): return
    var lt = s.find(String("<")); if lt < 0: return
    var comma = s.find(String(","), lt+1); if comma < 0: return
    var lbr = s.find(String("["), comma+1); if lbr < 0: return
    var rbr = s.find(String("]"), lbr+1); if rbr < 0: return

    var kind = s.starts_with(String("tensor<")) ? String("tensor") : String("buffer")
    var dt = String("")
    var i = lt + 1
    while i < comma: dt = dt + String.from_utf8([s.bytes()[i]]); i = i + 1
    var dimtxt = String(""); i = lbr + 1
    while i < rbr: dimtxt = dimtxt + String.from_utf8([s.bytes()[i]]); i = i + 1
    var dims = List[Int](); var oklist = False
    _parse_int_list(String("[") + dimtxt + String("]"), dims, oklist)
    info.kind = kind; info.dtype = dt; info.dims = dims; ok = True

fn _attr_value(op: Op, key: String, out found: Bool) -> String:
    found = False; var out_s = String("")
    for i in range(op.attrs.size()):
        var a = op.attrs[i]; var eq = a.find(String("="))
        if eq > 0:
            var k = String(""); var j = 0
            while j < eq: k = k + String.from_utf8([a.bytes()[j]]); j = j + 1
            if k == key:
                var v = String(""); j = eq + 1
                while j < a.size(): v = v + String.from_utf8([a.bytes()[j]]); j = j + 1
                out_s = v; found = True
    return out_s

fn _get_i64_attr(op: Op, key: String, default: Int) -> Int:
    var has = False; var s = _attr_value(op, key, has)
    if not has: return default
    var ok = False; var v = _parse_int_token(s, ok)
    return ok ? v : default

fn _elem_c_type(dtype: String) -> String:
    if dtype == String("i64"): return String("long long")
    if dtype == String("f64"): return String("double")
    return String("long long")

fn _vec_c_type(dtype: String, lanes: Int) -> String:
    if lanes == 2:
        if dtype == String("i64"): return String("longlong2")
        if dtype == String("f64"): return String("double2")
    if lanes == 4:
        if dtype == String("i64"): return String("longlong4")
        if dtype == String("f64"): return String("double4")
    return String("")

fn _sanitize(name: String) -> String:
    var out = String("")
    for i in range(name.size()):
        var ch = name.bytes()[i]
        out = out + String.from_utf8([ch == 46 ? 95 : ch])  # '.' -> '_'
    return out

# -----------------------------
# CUDA generator
# -----------------------------

struct Kernel:
    var name: String
    var body: String
    fn __init__(out self, name: String, body: String):
        self.name = name
        self.body = body

struct CodegenCUDA:
    var kernels: List[Kernel]
    var host_funcs: List[String]
    var counter: Int

    fn __init__(out self):
        self.kernels = List[Kernel]()
        self.host_funcs = List[String]()
        self.counter = 0

    fn _fresh_group(self, base: String) -> String:
        var s = _sanitize(base) + String("_k") + String(self.counter)
        self.counter = self.counter + 1
        return s

    fn _emit_header(self) -> String:
        var s = String("// Generated by Momijo codegen_cuda.mojo\n")
        s = s + String("#include <cuda_runtime.h>\n#include <stdint.h>\n#include <math.h>\n")
        s = s + String("#ifndef __CUDA_ARCH__\n#define __CUDA_ARCH__ 0\n#endif\n")
        s = s + String("#if __CUDA_ARCH__ < 600\n__device__ inline double atomicAdd(double* address, double val){\n")
        s = s + String("  unsigned long long int* addr_as_ull = (unsigned long long int*)address;\n  unsigned long long int old = *addr_as_ull, assumed;\n")
        s = s + String("  do { assumed = old; old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed))); }\n")
        s = s + String("  while (assumed != old);\n  return __longlong_as_double(old);\n}\n#endif\n")
        s = s + String("extern \"C\" {\n")
        return s

    fn _emit_footer(self) -> String:
        return String("} // extern \"C\"\n")

    fn _push_kernel(self, name: String, body: String) -> None:
        self.kernels.push_back(Kernel(name, body))

    # ---------------------
    # Binary: scalar + v2 + v4 + launcher
    # ---------------------
    fn _emit_vec_binary_variants(self, op: Op, base: String, dtype: String) -> None:
        if op.operands.size() < 3: return
        var dt = _elem_c_type(dtype)
        var group = self._fresh_group(op.name)
        var s_name = group + String("_s")
        var v2_name = group + String("_v2")
        var v4_name = group + String("_v4")

        var op_sym = String("+")
        if base == String("sub"): op_sym = String("-")
        elif base == String("mul"): op_sym = String("*")
        elif base == String("div"): op_sym = String("/")

        # scalar
        var body = String("__global__ void ") + s_name + String("(") + dt + String("* out, const ") + dt + String("* a, const ") + dt + String("* b, int N, int offset){\n")
        body = body + String("  int i = blockIdx.x * blockDim.x + threadIdx.x; int stride = gridDim.x * blockDim.x;\n")
        body = body + String("  for (int idx=i; idx<N; idx+=stride){ int j = idx + offset; out[j] = a[j] ") + op_sym + String(" b[j]; }\n}\n")
        self._push_kernel(s_name, body)

        # v2
        var v2t = _vec_c_type(dtype, 2)
        if v2t.size() > 0:
            var b2 = String("__global__ void ") + v2_name + String("(") + v2t + String("* out, const ") + v2t + String("* a, const ") + v2t + String("* b, int Nvec, int offset_vec){\n")
            b2 = b2 + String("  int i = blockIdx.x * blockDim.x + threadIdx.x; int stride = gridDim.x * blockDim.x;\n")
            b2 = b2 + String("  for (int idx=i; idx<Nvec; idx+=stride){ int j = idx + offset_vec; auto va=a[j]; auto vb=b[j]; ")
            b2 = b2 + String("  ") + v2t + String(" r; r.x = va.x ") + op_sym + String(" vb.x; r.y = va.y ") + op_sym + String(" vb.y; out[j]=r; }\n}\n")
            self._push_kernel(v2_name, b2)

        # v4
        var v4t = _vec_c_type(dtype, 4)
        if v4t.size() > 0:
            var b4 = String("__global__ void ") + v4_name + String("(") + v4t + String("* out, const ") + v4t + String("* a, const ") + v4t + String("* b, int Nvec, int offset_vec){\n")
            b4 = b4 + String("  int i = blockIdx.x * blockDim.x + threadIdx.x; int stride = gridDim.x * blockDim.x;\n")
            b4 = b4 + String("  for (int idx=i; idx<Nvec; idx+=stride){ int j = idx + offset_vec; auto va=a[j]; auto vb=b[j]; ")
            b4 = b4 + String("  ") + v4t + String(" r; r.x = va.x ") + op_sym + String(" vb.x; r.y = va.y ") + op_sym + String(" vb.y; r.z = va.z ") + op_sym + String(" vb.z; r.w = va.w ") + op_sym + String(" vb.w; out[j]=r; }\n}\n")
            self._push_kernel(v4_name, b4)

        # launcher
        var L = String("void launch_") + group + String("(") + dt + String("* out, const ") + dt + String("* a, const ") + dt + String("* b, int N, int offset, int vecW){\n")
        L = L + String("  int block=256; int grid = ( (vecW==4? N/4 : (vecW==2? N/2 : N)) + block - 1)/block; if(grid<1) grid=1;\n")
        L = L + String("  if(vecW>=4 && (N%4==0) && (offset%4==0)) { ") + v4_name + String("<<<grid,block>>>(((") + v4t + String("*)out),((") + v4t + String("*)a),((") + v4t + String("*)b), N/4, offset/4); return; }\n")
        L = L + String("  if(vecW>=2 && (N%2==0) && (offset%2==0)) { ") + v2_name + String("<<<grid,block>>>(((") + v2t + String("*)out),((") + v2t + String("*)a),((") + v2t + String("*)b), N/2, offset/2); return; }\n")
        L = L + String("  ") + s_name + String("<<<grid,block>>>(out,a,b,N,offset);\n}\n")
        self.host_funcs.push_back(L)

    # ---------------------
    # Unary: relu/sigmoid (scalar + variants)
    # ---------------------
    fn _emit_vec_unary_variants(self, op: Op, kind: String, dtype: String) -> None:
        if op.operands.size() < 2: return
        var dt = _elem_c_type(dtype)
        var group = self._fresh_group(op.name)
        var s_name = group + String("_s")
        var v2_name = group + String("_v2")
        var v4_name = group + String("_v4")

        fn _expr(x: String) -> String:
            if kind == String("relu"):
                return String("(") + x + String(" > 0 ? ") + x + String(" : 0") + (dtype==String("f64") ? String(".0") : String("")) + String(")")
            else:
                # sigmoid (float64 only ideal)
                return String("(1.0/(1.0 + exp(-(") + x + String("))))")

        # scalar
        var body = String("__global__ void ") + s_name + String("(") + dt + String("* out, const ") + dt + String("* x, int N, int offset){\n")
        body = body + String("  int i=blockIdx.x*blockDim.x+threadIdx.x; int stride=gridDim.x*blockDim.x;\n")
        body = body + String("  for(int idx=i; idx<N; idx+=stride){ int j=idx+offset; out[j]=") + _expr(String("x[j]")) + String("; }\n}\n")
        self._push_kernel(s_name, body)

        # vector lanes emit helper text
        var v2t = _vec_c_type(dtype, 2)
        if v2t.size() > 0:
            var b2 = String("__global__ void ") + v2_name + String("(") + v2t + String("* out, const ") + v2t + String("* x, int Nvec, int offset_vec){\n")
            b2 = b2 + String("  int i=blockIdx.x*blockDim.x+threadIdx.x; int stride=gridDim.x*blockDim.x;\n")
            b2 = b2 + String("  for(int idx=i; idx<Nvec; idx+=stride){ int j=idx+offset_vec; auto vx=x[j]; ") + v2t + String(" r; ")
            b2 = b2 + String("  r.x=") + _expr(String("vx.x")) + String("; r.y=") + _expr(String("vx.y")) + String("; out[j]=r; }\n}\n")
            self._push_kernel(v2_name, b2)

        var v4t = _vec_c_type(dtype, 4)
        if v4t.size() > 0:
            var b4 = String("__global__ void ") + v4_name + String("(") + v4t + String("* out, const ") + v4t + String("* x, int Nvec, int offset_vec){\n")
            b4 = b4 + String("  int i=blockIdx.x*blockDim.x+threadIdx.x; int stride=gridDim.x*blockDim.x;\n")
            b4 = b4 + String("  for(int idx=i; idx<Nvec; idx+=stride){ int j=idx+offset_vec; auto vx=x[j]; ") + v4t + String(" r; ")
            b4 = b4 + String("  r.x=") + _expr(String("vx.x")) + String("; r.y=") + _expr(String("vx.y")) + String("; r.z=") + _expr(String("vx.z")) + String("; r.w=") + _expr(String("vx.w")) + String("; out[j]=r; }\n}\n")
            self._push_kernel(v4_name, b4)

        var L = String("void launch_") + group + String("(") + dt + String("* out, const ") + dt + String("* x, int N, int offset, int vecW){\n")
        L = L + String("  int block=256; int grid=((vecW==4?N/4:(vecW==2?N/2:N))+block-1)/block; if(grid<1)grid=1;\n")
        L = L + String("  if(vecW>=4 && (N%4==0) && (offset%4==0)) { ") + v4_name + String("<<<grid,block>>>(((") + v4t + String("*)out),((") + v4t + String("*)x), N/4, offset/4); return; }\n")
        L = L + String("  if(vecW>=2 && (N%2==0) && (offset%2==0)) { ") + v2_name + String("<<<grid,block>>>(((") + v2t + String("*)out),((") + v2t + String("*)x), N/2, offset/2); return; }\n")
        L = L + String("  ") + s_name + String("<<<grid,block>>>(out,x,N,offset);\n}\n")
        self.host_funcs.push_back(L)

    # ---------------------
    # Fused: scalar + v2 + v4 + launcher
    # ---------------------
    fn _compile_rpn_to_infix(self, rpn: String) -> String:
        var stack = List[String](); var tok = String(""); var i = 0
        while i <= rpn.size():
            if i == rpn.size() or rpn.bytes()[i] == 32:
                if tok.size() > 0:
                    if tok.starts_with(String("x")): stack.push_back(tok)
                    elif tok == String("add") or tok == String("+"):
                        var b = stack.pop_back(); var a = stack.pop_back(); stack.push_back(String("(")+a+String(" + ")+b+String(")"))
                    elif tok == String("sub") or tok == String("-"):
                        var b2 = stack.pop_back(); var a2 = stack.pop_back(); stack.push_back(String("(")+a2+String(" - ")+b2+String(")"))
                    elif tok == String("mul") or tok == String("*"):
                        var b3 = stack.pop_back(); var a3 = stack.pop_back(); stack.push_back(String("(")+a3+String(" * ")+b3+String(")"))
                    elif tok == String("div") or tok == String("/"):
                        var b4 = stack.pop_back(); var a4 = stack.pop_back(); stack.push_back(String("(")+a4+String(" / ")+b4+String(")"))
                    else: stack.push_back(tok)
                    tok = String("")
            else:
                tok = tok + String.from_utf8([rpn.bytes()[i]])
            i = i + 1
        if stack.size() == 0: return String("x0")
        return stack[stack.size()-1]

    fn _emit_vec_fused_variants(self, op: Op, dtype: String) -> None:
        if op.operands.size() < 2: return
        var dt = _elem_c_type(dtype)
        var arity = _get_i64_attr(op, String("arity"), op.operands.size() - 1)
        if arity + 1 > op.operands.size(): arity = op.operands.size() - 1
        var has = False; var rpn = _attr_value(op, String("expr_rpn"), has)
        var expr = (has and rpn.size()>0) ? self._compile_rpn_to_infix(rpn) : String("")

        var group = self._fresh_group(op.name)
        var s_name = group + String("_s")
        var v2_name = group + String("_v2")
        var v4_name = group + String("_v4")

        # scalar
        var sig = dt + String("* out")
        for i in range(arity): sig = sig + String(", const ") + dt + String("* in") + String(i)
        sig = sig + String(", int N, int offset)")
        var body = String("__global__ void ") + s_name + String("(") + sig + String("{\n")
        body = body + String("  int i=blockIdx.x*blockDim.x+threadIdx.x; int stride=gridDim.x*blockDim.x;\n")
        body = body + String("  for(int idx=i; idx<N; idx+=stride){ int j=idx+offset; ")
        for i in range(arity):
            body = body + String("auto x") + String(i) + String(" = in") + String(i) + String("[j]; ")
        body = body + String("out[j] = ") + (expr.size()>0 ? expr : String("(") + (".0+".join([String("x")+String(k) for k in range(arity)])) + String(")")) + String("; }\n}\n")
        self._push_kernel(s_name, body)

        # helper to emit vector lanes bodies
        fn _vx(i: Int) -> String: return String("vx")+String(i)
        fn _emit_vec_body(lanes: Int) -> String:
            var vt = _vec_c_type(dtype, lanes)
            var sigv = vt + String("* out")
            for i in range(arity): sigv = sigv + String(", const ") + vt + String("* in") + String(i)
            sigv = sigv + String(", int Nvec, int offset_vec)")
            var b = String("__global__ void ") + (lanes==2 ? v2_name : v4_name) + String("(") + sigv + String("{\n")
            b = b + String("  int i=blockIdx.x*blockDim.x+threadIdx.x; int stride=gridDim.x*blockDim.x;\n")
            b = b + String("  for(int idx=i; idx<Nvec; idx+=stride){ int j=idx+offset_vec; ")
            for i in range(arity): b = b + String("auto ")+_vx(i)+String(" = in")+String(i)+String("[j]; ")
            b = b + String(vt + String(" r; "))
            # Emit lanes
            var comps = (lanes==2) ? ["x","y"] : ["x","y","z","w"]
            for c in comps:
                var repl = String(expr)
                if expr.size()==0:
                    # sum default
                    var sum = String("0") + (dtype==String("f64") ? String(".0") : String(""))
                    for i in range(arity): sum = sum + String(" + ") + _vx(i) + String(".") + String(c)
                    repl = sum
                else:
                    # replace x0.. with vx0.c
                    var e = String("")
                    var p = 0
                    while p < repl.size():
                        # simple pass: when 'x' then number; else copy
                        var ch = repl.bytes()[p]
                        if ch == 120 and p+1 < repl.size() and repl.bytes()[p+1] >= 48 and repl.bytes()[p+1] <= 57:
                            # read digits
                            var q = p + 1; var num = 0
                            while q < repl.size() and repl.bytes()[q] >= 48 and repl.bytes()[q] <= 57:
                                num = num*10 + Int(repl.bytes()[q]-48); q = q + 1
                            e = e + _vx(num) + String(".") + String(c); p = q; continue
                        e = e + String.from_utf8([ch]); p = p + 1
                    repl = e
                b = b + String("r.") + String(c) + String(" = ") + repl + String("; ")
            b = b + String(" out[j]=r; }\n}\n")
            return b

        self._push_kernel(v2_name, _emit_vec_body(2))
        self._push_kernel(v4_name, _emit_vec_body(4))

        # launcher
        var L = String("void launch_") + group + String("(") + dt + String("* out")
        for i in range(arity): L = L + String(", const ") + dt + String("* in") + String(i)
        L = L + String(", int N, int offset, int vecW){\n  int block=256; int grid=((vecW==4?N/4:(vecW==2?N/2:N))+block-1)/block; if(grid<1)grid=1;\n")
        L = L + String("  if(vecW>=4 && (N%4==0) && (offset%4==0)) { ") + v4_name + String("<<<grid,block>>>(((") + _vec_c_type(dtype,4) + String("*)out)")
        for i in range(arity): L = L + String(", ((") + _vec_c_type(dtype,4) + String("*)in") + String(i) + String(")")
        L = L + String(", N/4, offset/4); return; }\n")
        L = L + String("  if(vecW>=2 && (N%2==0) && (offset%2==0)) { ") + v2_name + String("<<<grid,block>>>(((") + _vec_c_type(dtype,2) + String("*)out)")
        for i in range(arity): L = L + String(", ((") + _vec_c_type(dtype,2) + String("*)in") + String(i) + String(")")
        L = L + String(", N/2, offset/2); return; }\n")
        L = L + String("  ") + s_name + String("<<<grid,block>>>(out")
        for i in range(arity): L = L + String(", in") + String(i)
        L = L + String(", N, offset);\n}\n")
        self.host_funcs.push_back(L)

    # ---------------------
    # reduce_sum (atomicAdd)
    # ---------------------
    fn _emit_reduce_sum(self, op: Op, dtype: String) -> None:
        if op.operands.size() < 2: return
        var dt = _elem_c_type(dtype)
        var group = self._fresh_group(op.name)
        var kname = group + String("_rs")
        var body = String("__global__ void ") + kname + String("(const ") + dt + String("* x, ") + dt + String("* out, int N){\n")
        body = body + String("  ") + dt + String(" acc = 0; int i=blockIdx.x*blockDim.x+threadIdx.x; int stride=gridDim.x*blockDim.x;\n")
        body = body + String("  for(int idx=i; idx<N; idx+=stride) acc += x[idx];\n")
        body = body + String("  atomicAdd(out, acc);\n}\n")
        self._push_kernel(kname, body)
        var L = String("void launch_") + group + String("(const ") + dt + String("* x, ") + dt + String("* out, int N){ int block=256; int grid=(N+block-1)/block; if(grid<1)grid=1; ")
        L = L + kname + String("<<<grid,block>>>(x,out,N); }\n")
        self.host_funcs.push_back(L)

    # ---------------------
    # conv2d NHWC x HWCO (naïve)
    # ---------------------
    fn _emit_conv2d_nhwc(self, op: Op, dtype: String) -> None:
        if op.operands.size() < 2: return
        var dt = _elem_c_type(dtype)

        # Parse shapes & attrs
        var in_info = ShapeInfo(); var okI = False
        _parse_shape_from_type(op.operands[0].value.typ, in_info, okI); if not okI: return
        var k_info = ShapeInfo(); var okK = False
        _parse_shape_from_type(op.operands[1].value.typ, k_info, okK); if not okK: return
        var N = (in_info.dims.size()>0) ? in_info.dims[0] : -1
        var H = (in_info.dims.size()>1) ? in_info.dims[1] : -1
        var W = (in_info.dims.size()>2) ? in_info.dims[2] : -1
        var C = (in_info.dims.size()>3) ? in_info.dims[3] : -1
        var KH = (k_info.dims.size()>0) ? k_info.dims[0] : -1
        var KW = (k_info.dims.size()>1) ? k_info.dims[1] : -1
        var KC = (k_info.dims.size()>2) ? k_info.dims[2] : -1
        var OC = (k_info.dims.size()>3) ? k_info.dims[3] : -1

        var strides = List[Int](); var hasS = False
        _parse_int_list(_attr_value(op, String("strides"), hasS), strides, hasS)
        var sH = (strides.size()>=2) ? strides[0] : 1
        var sW = (strides.size()>=2) ? strides[1] : 1
        var hasP = False; var padding = _attr_value(op, String("padding"), hasP)
        if not hasP: padding = String("valid")

        # Compute OH/OW and pads (embed constants)
        fn _ceil_div(a:Int,b:Int)->Int: return (b<=0 or a<0) ? -1 : (a + b - 1) / b
        var OH = (padding==String("same")) ? _ceil_div(H,sH) : ((H-KH)/sH + 1)
        var OW = (padding==String("same")) ? _ceil_div(W,sW) : ((W-KW)/sW + 1)
        var padH = (padding==String("same") and OH>0) ? ((OH-1)*sH + KH - H) : 0
        var padW = (padding==String("same") and OW>0) ? ((OW-1)*sW + KW - W) : 0
        var padTop = padH/2; var padLeft = padW/2

        var group = self._fresh_group(op.name)
        var kname = group + String("_conv")

        var body = String("__global__ void ") + kname + String("(") + dt + String("* out, const ") + dt + String("* inp, const ") + dt + String("* ker){\n")
        body = body + String("  const int N=")+String(N)+String(", H=")+String(H)+String(", W=")+String(W)+String(", C=")+String(C)+String(", KH=")+String(KH)+String(", KW=")+String(KW)+String(", OC=")+String(OC)+String(";\n")
        body = body + String("  const int sH=")+String(sH)+String(", sW=")+String(sW)+String(", OH=")+String(OH)+String(", OW=")+String(OW)+String(", padTop=")+String(padTop)+String(", padLeft=")+String(padLeft)+String(";\n")
        body = body + String("  int i=blockIdx.x*blockDim.x+threadIdx.x; int stride=gridDim.x*blockDim.x; int total=N*OH*OW*OC;\n")
        body = body + String("  for(int idx=i; idx<total; idx+=stride){ int t=idx; int oc=t%OC; t/=OC; int ow=t%OW; t/=OW; int oh=t%OH; t/=OH; int n=t;\n")
        body = body + String("    ")+dt+String(" acc=0; int in_h0 = oh*sH - padTop; int in_w0 = ow*sW - padLeft;\n")
        body = body + String("    for(int kh=0; kh<KH; ++kh){ int ih=in_h0+kh; if(ih<0||ih>=H) continue; for(int kw=0; kw<KW; ++kw){ int iw=in_w0+kw; if(iw<0||iw>=W) continue; for(int c=0;c<C;++c){\n")
        body = body + String("      ")+dt+String(" x = inp[ ((n*H + ih)*W + iw)*C + c ];\n")
        body = body + String("      ")+dt+String(" w = ker[ ((kh*KW + kw)*C + c)*OC + oc ]; acc += x*w; }} }\n")
        body = body + String("    out[ ((n*OH + oh)*OW + ow)*OC + oc ] = acc;\n  }\n}\n")
        self._push_kernel(kname, body)

        var L = String("void launch_") + group + String("(") + dt + String("* out, const ") + dt + String("* inp, const ") + dt + String("* ker){ int block=128; int grid=((")+String(N)+String("*")+String(OH)+String("*")+String(OW)+String("*")+String(OC)+String(")+block-1)/block; if(grid<1)grid=1; ")
        L = L + kname + String("<<<grid,block>>>(out,inp,ker); }\n")
        self.host_funcs.push_back(L)

    fn _dtype_of_value(self, v: Value) -> String:
        var info = ShapeInfo(); var ok = False
        _parse_shape_from_type(v.typ, info, ok)
        if ok: return info.dtype
        return String("i64")

    fn _handle_op(self, op: Op) -> None:
        if op.name.starts_with(String("vec.")):
            var i = op.name.find(String(".")); var j = op.name.find(String("."), i+1)
            if i >= 0 and j > i:
                var base = String(""); var k = i + 1
                while k < j: base = base + String.from_utf8([op.name.bytes()[k]]); k = k + 1
                var dtype = String(""); k = j + 1
                while k < op.name.size(): dtype = dtype + String.from_utf8([op.name.bytes()[k]]); k = k + 1

                if base == String("add") or base == String("sub") or base == String("mul") or base == String("div"):
                    self._emit_vec_binary_variants(op, base, dtype)
                elif base == String("relu") or base == String("sigmoid"):
                    self._emit_vec_unary_variants(op, base, dtype)
                elif base == String("fused"):
                    self._emit_vec_fused_variants(op, dtype)
                elif base == String("reduce_sum"):
                    self._emit_reduce_sum(op, dtype)
        elif op.name.find(String("conv2d")) >= 0:
            # Support ll./hl. conv2d
            var dtype = self._dtype_of_value(op.operands[0].value)
            self._emit_conv2d_nhwc(op, dtype)
        elif op.name == String("buf.memcpy"):
            # keep simple memcpy scalar kernel + launcher
            if op.operands.size() < 2: return
            var dtype = self._dtype_of_value(op.operands[0].value)
            var dt = _elem_c_type(dtype)
            var group = self._fresh_group(op.name)
            var kname = group + String("_cpy")
            var body = String("__global__ void ") + kname + String("(") + dt + String("* dst, const ") + dt + String("* src, int N, int offset){\n")
            body = body + String("  int i=blockIdx.x*blockDim.x+threadIdx.x; int stride=gridDim.x*blockDim.x;\n")
            body = body + String("  for(int idx=i; idx<N; idx+=stride){ int j=idx+offset; dst[j]=src[j]; }\n}\n")
            self._push_kernel(kname, body)
            var L = String("void launch_") + group + String("(") + dt + String("* dst, const ") + dt + String("* src, int N, int offset){ int block=256; int grid=(N+block-1)/block; if(grid<1)grid=1; ")
            L = L + kname + String("<<<grid,block>>>(dst,src,N,offset); }\n")
            self.host_funcs.push_back(L)

    fn generate(self, m: Module) -> String:
        for fi in range(m.functions.size()):
            var f = m.functions[fi]
            for bi in range(f.blocks.size()):
                var b = f.blocks[bi]
                for oi in range(b.ops.size()):
                    self._handle_op(b.ops[oi])

        var out = self._emit_header()
        # Host launchers
        for i in range(self.host_funcs.size()): out = out + self.host_funcs[i] + String("\n")
        # Device kernels
        for i in range(self.kernels.size()): out = out + self.kernels[i].body + String("\n")
        out = out + self._emit_footer()
        return out

# -----------------------------
# Tiny demo & self-test
# -----------------------------

struct IdGen:
    var next_id: Int
    fn __init__(out self): self.next_id = 0
    fn fresh(self) -> Int:
        var r = self.next_id; self.next_id = self.next_id + 1; return r

fn _buffer(dtype: String, dims: List[Int]) -> TypeDesc:
    var s = String("buffer<") + dtype + String(",[")
    for i in range(dims.size()):
        s = s + String(dims[i]); if i + 1 < dims.size(): s = s + String(",")
    s = s + String("]>"); return TypeDesc(s)

fn _demo_module() -> Module:
    var m = Module(String("demo_codegen_cuda_plus"))
    var f = Function(String("main"))
    var b = Block(String("entry"))
    var idg = IdGen()

    var out0 = Value(idg.fresh(), _buffer(String("i64"), [64]))
    var a0 = Value(idg.fresh(), _buffer(String("i64"), [64]))
    var b0 = Value(idg.fresh(), _buffer(String("i64"), [64]))
    var add = Op(String("vec.add.i64"), Location(String("d"), 1, 1))
    add.operands.push_back(Operand(out0)); add.operands.push_back(Operand(a0)); add.operands.push_back(Operand(b0))
    add.attrs.push_back(String("N=64")); add.attrs.push_back(String("vector=4"))
    b.ops.push_back(add)

    var out1 = Value(idg.fresh(), _buffer(String("f64"), [32]))
    var x1 = Value(idg.fresh(), _buffer(String("f64"), [32]))
    var relu = Op(String("vec.relu.f64"), Location(String("d"), 1, 2))
    relu.operands.push_back(Operand(out1)); relu.operands.push_back(Operand(x1))
    relu.attrs.push_back(String("N=32")); relu.attrs.push_back(String("offset=0")); relu.attrs.push_back(String("vector=2"))
    b.ops.push_back(relu)

    var out2 = Value(idg.fresh(), _buffer(String("f64"), [18]))
    var a2 = Value(idg.fresh(), _buffer(String("f64"), [18]))
    var b2 = Value(idg.fresh(), _buffer(String("f64"), [18]))
    var c2 = Value(idg.fresh(), _buffer(String("f64"), [18]))
    var fused = Op(String("vec.fused.f64"), Location(String("d"), 1, 3))
    fused.operands.push_back(Operand(out2)); fused.operands.push_back(Operand(a2)); fused.operands.push_back(Operand(b2)); fused.operands.push_back(Operand(c2))
    fused.attrs.push_back(String("N=18")); fused.attrs.push_back(String("arity=3")); fused.attrs.push_back(String("expr_rpn=x0 x1 add x2 mul")); fused.attrs.push_back(String("offset=0")); fused.attrs.push_back(String("vector=4"))
    b.ops.push_back(fused)

    var x3 = Value(idg.fresh(), _buffer(String("f64"), [128]))
    var out3 = Value(idg.fresh(), _buffer(String("f64"), [1]))
    var red = Op(String("vec.reduce_sum.f64"), Location(String("d"), 1, 4))
    red.operands.push_back(Operand(out3)); red.operands.push_back(Operand(x3))
    red.attrs.push_back(String("N=128"))
    b.ops.push_back(red)

    var I = Value(idg.fresh(), _buffer(String("f64"), [1, 8, 8, 3]))
    var K = Value(idg.fresh(), _buffer(String("f64"), [3, 3, 3, 4]))
    var conv = Op(String("ll.conv2d"), Location(String("d"), 1, 5))
    var O = Value(idg.fresh(), _buffer(String("f64"), [1, 8, 8, 4])); conv.results.push_back(Result(O))
    conv.operands.push_back(Operand(I)); conv.operands.push_back(Operand(K))
    conv.attrs.push_back(String("strides=[1,1]")); conv.attrs.push_back(String("padding=same"))
    b.ops.push_back(conv)

    var ret = Op(String("ll.ret"), Location(String("d"), 1, 6)); b.ops.push_back(ret)

    f.blocks.push_back(b); m.functions.push_back(f); return m

fn _self_test_codegen_cuda() -> Bool:
    var m = _demo_module()
    var cg = CodegenCUDA()
    var cu = cg.generate(m)

    var ok = True
    if cu.find(String("void launch_vec_add_i64_k0")) < 0: ok = False
    if cu.find(String("__global__ void vec_add_i64_k0_v4")) < 0: ok = False
    if cu.find(String("void launch_vec_relu_f64_k1")) < 0: ok = False
    if cu.find(String("__global__ void vec_fused_f64_k2_v4")) < 0: ok = False
    if cu.find(String("__global__ void vec_reduce_sum_f64_k3_rs")) < 0: ok = False
    if cu.find(String("void launch_ll_conv2d_k4")) < 0: ok = False
    if ok: print(String("OK")) else: print(String("FAIL"))
    return ok

 
