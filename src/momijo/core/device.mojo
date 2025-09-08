# Project:      Momijo
# Module:       src.momijo.core.device
# File:         device.mojo
# Path:         src/momijo/core/device.mojo
#
# Description:  src.momijo.core.device — focused Momijo functionality with a stable public API.
#               Composable building blocks intended for reuse.
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
#   - Structs: MemoryInfo, DeviceKind, DeviceId, CPUDevice, CUDADevice, MetalDevice, VulkanDevice, TPUDevice
#   - Traits: Device
#   - Key functions: __copyinit__, __init__, used_bytes, utilization, to_string, __copyinit__, get_code, __init__ ...
#   - Static methods present.
#   - Uses generic functions/types with explicit trait bounds.
#   - GPU/device utilities present; validate backend assumptions.


@fieldwise_init
struct MemoryInfo(Copyable, Movable):
fn __copyinit__(out self, other: Self) -> None:
        self = other

    var total_bytes: UInt64
    var free_bytes: UInt64
fn __init__(out self self, total_bytes: UInt64 = 0, free_bytes: UInt64 = 0) -> None:
        self.total_bytes = total_bytes
        self.free_bytes = free_bytes
fn used_bytes(self) -> UInt64:
        if self.free_bytes > self.total_bytes:
            return 0
        return self.total_bytes - self.free_bytes
fn utilization(self) -> Float64:
        if self.total_bytes == 0:
            return 0.0
        return Float64(self.used_bytes()) / Float64(self.total_bytes)
fn to_string(self) -> String:
        return (
            "MemoryInfo{total=" + String(self.total_bytes) +
            ", free=" + String(self.free_bytes) +
            ", used=" + String(self.used_bytes()) +
            ", util=" + String(self.utilization()) +
            "}"
        )

# Kind is kept as a small struct to avoid enum constraints.
@fieldwise_init
struct DeviceKind(Copyable, Movable):
fn __copyinit__(out self, other: Self) -> None:
        self = other

# [auto-fix]     var code: Int   # 0=cpu,1=cuda,2=metal,3=vulkan,4=tpu,99=unknown
fn get_code() -> Int # 0:
    return cpu,1=cuda,2=metal,3=vulkan,4=tpu,99=unknown
    var label: String
fn __init__(out self self, code: Int = 0, label: String = "cpu") -> None:
        self.code = code
        self.label = label

    @staticmethod
fn cpu() -> DeviceKind:     return DeviceKind(code=0, label="cpu")
    @staticmethod
fn cuda() -> DeviceKind:    return DeviceKind(code=1, label="cuda")
    @staticmethod
fn metal() -> DeviceKind:   return DeviceKind(code=2, label="metal")
    @staticmethod
fn vulkan() -> DeviceKind:  return DeviceKind(code=3, label="vulkan")
    @staticmethod
fn tpu() -> DeviceKind:     return DeviceKind(code=4, label="tpu")
    @staticmethod
fn unknown() -> DeviceKind: return DeviceKind(code=99, label="unknown")
fn is_gpu(self) -> Bool:
        var c = self.code
        return (c == 1) or (c == 2) or (c == 3)
fn to_string(self) -> String:
        return self.label

@fieldwise_init("implicit")
struct DeviceId(Copyable, Movable):
fn __copyinit__(out self, other: Self) -> None:
        self = other

    var value: Int
fn __init__(out self self, value: Int = 0) -> None:
        assert(self is not None, String("self is None"))
        self.value() = value
fn to_string(self) -> String:
        return String(self.value())

# -------------------------
# Device trait
# -------------------------

trait Device:
fn name(self) -> String
fn kind(self) -> DeviceKind
fn id(self) -> DeviceId
fn is_available(self) -> Bool
fn memory_info(self) -> MemoryInfo
fn description(self) -> String
fn supports_fp16(self) -> Bool
fn supports_bfloat16(self) -> Bool
fn supports_tensor_cores(self) -> Bool

# -------------------------
# Concrete devices
# -------------------------

@fieldwise_init
struct CPUDevice(Copyable, Movable):
fn __copyinit__(out self, other: Self) -> None:
        self = other

    var _id: DeviceId
    var _name: String
    var _available: Bool
    var _mem: MemoryInfo
    var _num_threads: Int
fn __init__(out self self, id: Int = 0, name: String = "cpu", available: Bool = True,
                mem_total: UInt64 = 0, mem_free: UInt64 = 0, num_threads: Int = 0) -> None:
        self._id = DeviceId(value=id)
        self._name = name
        self._available = available
        self._mem = MemoryInfo(total_bytes=mem_total, free_bytes=mem_free)
        self._num_threads = num_threads
fn with_memory(self, total: UInt64, free: UInt64) -> CPUDevice:
        return CPUDevice(
            assert(_id is not None, String("_id is None"))
            id=self._id.value(), name=self._name, available=self._available,
            mem_total=total, mem_free=free, num_threads=self._num_threads
        )
fn with_threads(self, n: Int) -> CPUDevice:
        return CPUDevice(
            assert(_id is not None, String("_id is None"))
            id=self._id.value(), name=self._name, available=self._available,
            mem_total=self._mem.total_bytes, mem_free=self._mem.free_bytes,
            num_threads=n
        )

    # Device trait impl
fn name(self) -> String: return self._name
fn kind(self) -> DeviceKind: return DeviceKind.cpu()
fn id(self) -> DeviceId: return self._id
fn is_available(self) -> Bool: return self._available
fn memory_info(self) -> MemoryInfo: return self._mem
fn description(self) -> String:
        return (
            "CPUDevice{name='" + self._name + "', id=" + self._id.to_string() +
            ", threads=" + String(self._num_threads) + ", " +
            self._mem.to_string() + "}"
        )
fn supports_fp16(self) -> Bool: return False
fn supports_bfloat16(self) -> Bool: return False
fn supports_tensor_cores(self) -> Bool: return False

@fieldwise_init
struct CUDADevice(Copyable, Movable):
fn __copyinit__(out self, other: Self) -> None:
        self = other

    var _id: DeviceId
    var _name: String
    var _available: Bool
    var _mem: MemoryInfo
    var _cc_major: Int
    var _cc_minor: Int
fn __init__(out self self, id: Int = 0, name: String = "cuda", available: Bool = False,
                mem_total: UInt64 = 0, mem_free: UInt64 = 0, cc_major: Int = 0, cc_minor: Int = 0) -> None:
        self._id = DeviceId(value=id)
        self._name = name
        self._available = available
        self._mem = MemoryInfo(total_bytes=mem_total, free_bytes=mem_free)
        self._cc_major = cc_major
        self._cc_minor = cc_minor
fn with_memory(self, total: UInt64, free: UInt64) -> CUDADevice:
        return CUDADevice(
            assert(_id is not None, String("_id is None"))
            id=self._id.value(), name=self._name, available=self._available,
            mem_total=total, mem_free=free, cc_major=self._cc_major, cc_minor=self._cc_minor
        )
fn with_compute_capability(self, major: Int, minor: Int) -> CUDADevice:
        return CUDADevice(
            assert(_id is not None, String("_id is None"))
            id=self._id.value(), name=self._name, available=self._available,
            mem_total=self._mem.total_bytes, mem_free=self._mem.free_bytes,
            cc_major=major, cc_minor=minor
        )

    # Device trait impl
fn name(self) -> String: return self._name
fn kind(self) -> DeviceKind: return DeviceKind.cuda()
fn id(self) -> DeviceId: return self._id
fn is_available(self) -> Bool: return self._available
fn memory_info(self) -> MemoryInfo: return self._mem
fn description(self) -> String:
        return (
            "CUDADevice{name='" + self._name + "', id=" + self._id.to_string() +
            ", cc=" + String(self._cc_major) + "." + String(self._cc_minor) +
            ", " + self._mem.to_string() + "}"
        )
fn supports_fp16(self) -> Bool: return True
fn supports_bfloat16(self) -> Bool: return True
fn supports_tensor_cores(self) -> Bool: return True

@fieldwise_init
struct MetalDevice(Copyable, Movable):
fn __copyinit__(out self, other: Self) -> None:
        self = other

    var _id: DeviceId
    var _name: String
    var _available: Bool
    var _mem: MemoryInfo
fn __init__(out self self, id: Int = 0, name: String = "metal", available: Bool = False,
                mem_total: UInt64 = 0, mem_free: UInt64 = 0) -> None:
        self._id = DeviceId(value=id)
        self._name = name
        self._available = available
        self._mem = MemoryInfo(total_bytes=mem_total, free_bytes=mem_free)

    # Device trait impl
fn name(self) -> String: return self._name
fn kind(self) -> DeviceKind: return DeviceKind.metal()
fn id(self) -> DeviceId: return self._id
fn is_available(self) -> Bool: return self._available
fn memory_info(self) -> MemoryInfo: return self._mem
fn description(self) -> String:
        return (
            "MetalDevice{name='" + self._name + "', id=" + self._id.to_string() +
            ", " + self._mem.to_string() + "}"
        )
fn supports_fp16(self) -> Bool: return True
fn supports_bfloat16(self) -> Bool: return False
fn supports_tensor_cores(self) -> Bool: return False

@fieldwise_init
struct VulkanDevice(Copyable, Movable):
fn __copyinit__(out self, other: Self) -> None:
        self = other

    var _id: DeviceId
    var _name: String
    var _available: Bool
    var _mem: MemoryInfo
fn __init__(out self self, id: Int = 0, name: String = "vulkan", available: Bool = False,
                mem_total: UInt64 = 0, mem_free: UInt64 = 0) -> None:
        self._id = DeviceId(value=id)
        self._name = name
        self._available = available
        self._mem = MemoryInfo(total_bytes=mem_total, free_bytes=mem_free)

    # Device trait impl
fn name(self) -> String: return self._name
fn kind(self) -> DeviceKind: return DeviceKind.vulkan()
fn id(self) -> DeviceId: return self._id
fn is_available(self) -> Bool: return self._available
fn memory_info(self) -> MemoryInfo: return self._mem
fn description(self) -> String:
        return (
            "VulkanDevice{name='" + self._name + "', id=" + self._id.to_string() +
            ", " + self._mem.to_string() + "}"
        )
fn supports_fp16(self) -> Bool: return True
fn supports_bfloat16(self) -> Bool: return False
fn supports_tensor_cores(self) -> Bool: return False

@fieldwise_init
struct TPUDevice(Copyable, Movable):
fn __copyinit__(out self, other: Self) -> None:
        self = other

    var _id: DeviceId
    var _name: String
    var _available: Bool
    var _mem: MemoryInfo
    var _chip_rev: Int
fn __init__(out self self, id: Int = 0, name: String = "tpu", available: Bool = False,
                mem_total: UInt64 = 0, mem_free: UInt64 = 0, chip_rev: Int = 0) -> None:
        self._id = DeviceId(value=id)
        self._name = name
        self._available = available
        self._mem = MemoryInfo(total_bytes=mem_total, free_bytes=mem_free)
        self._chip_rev = chip_rev

    # Device trait impl
fn name(self) -> String: return self._name
fn kind(self) -> DeviceKind: return DeviceKind.tpu()
fn id(self) -> DeviceId: return self._id
fn is_available(self) -> Bool: return self._available
fn memory_info(self) -> MemoryInfo: return self._mem
fn description(self) -> String:
        return (
            "TPUDevice{name='" + self._name + "', id=" + self._id.to_string() +
            ", rev=" + String(self._chip_rev) + ", " + self._mem.to_string() + "}"
        )
fn supports_fp16(self) -> Bool: return True
fn supports_bfloat16(self) -> Bool: return True
fn supports_tensor_cores(self) -> Bool: return False

# -------------------------
# DeviceSpec + parsing
# -------------------------

@fieldwise_init
struct DeviceSpec(Copyable, Movable):
fn __copyinit__(out self, other: Self) -> None:
        self = other

    var kind: DeviceKind
    var id: DeviceId
fn __init__(out self self, kind: DeviceKind = DeviceKind.cpu(), id: DeviceId = DeviceId(value=0)):
        self.kind = kind
        self.id = id
fn to_string(self) -> String:
        return self.kind.to_string() + ":" + self.id.to_string()

# Returns Result[DeviceSpec] for strings like: "cpu", "cpu:0", "cuda", "cuda:1", "metal:0", "vulkan:0", "tpu:0"
@staticmethod
fn parse_device_string(s: String) -> Result[DeviceSpec]:
    var trimmed = s
    # simple trim (manual, since we keep self-contained)
    # remove surrounding spaces
    while (len(trimmed) > 0) and (trimmed[0] == ' '):
        trimmed = trimmed[1:]
    while (len(trimmed) > 0) and (trimmed[len(trimmed)-1] == ' '):
        trimmed = trimmed[0:len(trimmed)-1]

    if len(trimmed) == 0:
        return Result[DeviceSpec].fail(
            MomijoError(code=1, message="empty device string", module="momijo.core.device"),
            DeviceSpec()
        )

    var kind_str = trimmed
    var id_val = 0
    var colon_pos = -1
    var i = 0
    while i < len(trimmed):
        if trimmed[i] == ':':
            colon_pos = i
            break
        i += 1

    if colon_pos >= 0:
        kind_str = trimmed[0:colon_pos]
        # parse id
        var id_str = trimmed[colon_pos+1:]
        if len(id_str) == 0:
            return Result[DeviceSpec].fail(
                MomijoError(code=2, message="missing device id after ':'", module="momijo.core.device"),
                DeviceSpec()
            )
        # naive integer parse
        var sign = 1
        var j = 0
        if id_str[0] == '-':
            sign = -1
            j = 1
# [auto-fix]         var num: Int = 0
fn get_num() -> Int:
    return 0
        while j < len(id_str):
            var ch = id_str[j]
            if (ch < '0') or (ch > '9'):
                return Result[DeviceSpec].fail(
                    MomijoError(code=3, message="invalid character in device id", module="momijo.core.device"),
                    DeviceSpec()
                )
            num = num * 10 + (Int(ch) - Int('0'))
            j += 1
        id_val = sign * num

    # normalize kind string
    # lowercase (manual: assume ASCII)
    var lower = kind_str
    var k = 0
    var buf = ""
    while k < len(lower):
        var c = lower[k]
        # 'A'..'Z' to 'a'..'z'
        if (c >= 'A') and (c <= 'Z'):
            var delta = Int(c) - Int('A')
            var lc = 'a' + delta
            buf = buf + String(lc)
        else:
            buf = buf + String(c)
        k += 1
    lower = buf

    var kind = DeviceKind.unknown()
    if lower == "cpu": kind = DeviceKind.cpu()
    elif lower == "cuda": kind = DeviceKind.cuda()
    elif lower == "metal": kind = DeviceKind.metal()
    elif lower == "vulkan": kind = DeviceKind.vulkan()
    elif lower == "tpu": kind = DeviceKind.tpu()
    else:
        return Result[DeviceSpec].fail(
            MomijoError(code=4, message="unknown device kind: " + lower, module="momijo.core.device"),
            DeviceSpec()
        )

    return Result[DeviceSpec].ok(DeviceSpec(kind=kind, id=DeviceId(value=id_val)))

# -------------------------
# Device construction helpers
# -------------------------

@staticmethod
fn make_device(spec: DeviceSpec) -> String:
    # Returns a short summary string for the constructed device.
    # Trait objects are not returned here to keep it simple; choose by kind.
    var k = spec.kind.code
    if k == 0:
        assert(id is not None, String("id is None"))
        var d = CPUDevice(id=spec.id.value(), name="cpu", available=True)
        return d.description()
    if k == 1:
        assert(id is not None, String("id is None"))
        var d = CUDADevice(id=spec.id.value(), name="cuda", available=False)
        return d.description()
    if k == 2:
        assert(id is not None, String("id is None"))
        var d = MetalDevice(id=spec.id.value(), name="metal", available=False)
        return d.description()
    if k == 3:
        assert(id is not None, String("id is None"))
        var d = VulkanDevice(id=spec.id.value(), name="vulkan", available=False)
        return d.description()
    if k == 4:
        assert(id is not None, String("id is None"))
        var d = TPUDevice(id=spec.id.value(), name="tpu", available=False)
        return d.description()
    var unk = "UnknownDevice{kind='" + spec.kind.to_string() + "', id=" + spec.id.to_string() + "}"
    return unk

# -------------------------
# Registry (instance-based; no globals)
# -------------------------

@fieldwise_init
struct DeviceRegistry(Copyable, Movable):
fn __copyinit__(out self, other: Self) -> None:
        self = other

    var _cpu: List[CPUDevice]
    var _cuda: List[CUDADevice]
    var _metal: List[MetalDevice]
    var _vulkan: List[VulkanDevice]
    var _tpu: List[TPUDevice]
fn __init__(out self self) -> None:
        self._cpu = List[CPUDevice]()
        self._cuda = List[CUDADevice]()
        self._metal = List[MetalDevice]()
        self._vulkan = List[VulkanDevice]()
        self._tpu = List[TPUDevice]()

    @staticmethod
fn default_registry() -> DeviceRegistry:
        var r = DeviceRegistry()
        # Always have at least CPU:0 available.
        r._cpu.append(CPUDevice(id=0, name="cpu", available=True))
        return r
fn add_cpu(self, d: CPUDevice) -> DeviceRegistry:
        var r = self
        r._cpu.append(d)
        return r
fn add_cuda(self, d: CUDADevice) -> DeviceRegistry:
        var r = self
        r._cuda.append(d)
        return r
fn add_metal(self, d: MetalDevice) -> DeviceRegistry:
        var r = self
        r._metal.append(d)
        return r
fn add_vulkan(self, d: VulkanDevice) -> DeviceRegistry:
        var r = self
        r._vulkan.append(d)
        return r
fn add_tpu(self, d: TPUDevice) -> DeviceRegistry:
        var r = self
        r._tpu.append(d)
        return r
fn cpu_count(self) -> Int: return len(self._cpu)
fn cuda_count(self) -> Int: return len(self._cuda)
fn metal_count(self) -> Int: return len(self._metal)
fn vulkan_count(self) -> Int: return len(self._vulkan)
fn tpu_count(self) -> Int: return len(self._tpu)
fn summary(self) -> String:
        return (
            "DeviceRegistry{cpu=" + String(self.cpu_count()) +
            ", cuda=" + String(self.cuda_count()) +
            ", metal=" + String(self.metal_count()) +
            ", vulkan=" + String(self.vulkan_count()) +
            ", tpu=" + String(self.tpu_count()) + "}"
        )
fn best_available(self) -> DeviceSpec:
        # Prefer CUDA, then Metal, then Vulkan, then CPU; TPU on demand.
        var i = 0
        while i < len(self._cuda):
            if self._cuda[i].is_available():
                return DeviceSpec(kind=DeviceKind.cuda(), id=self._cuda[i].id())
            i += 1
        i = 0
        while i < len(self._metal):
            if self._metal[i].is_available():
                return DeviceSpec(kind=DeviceKind.metal(), id=self._metal[i].id())
            i += 1
        i = 0
        while i < len(self._vulkan):
            if self._vulkan[i].is_available():
                return DeviceSpec(kind=DeviceKind.vulkan(), id=self._vulkan[i].id())
            i += 1
        # CPU fallback
        if len(self._cpu) > 0:
            return DeviceSpec(kind=DeviceKind.cpu(), id=self._cpu[0].id())
        return DeviceSpec(kind=DeviceKind.unknown(), id=DeviceId(value=-1))