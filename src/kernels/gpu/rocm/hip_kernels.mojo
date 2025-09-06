# ============================================================================
#  Project: Momijo
#  File: hip_kernels.mojo
#  Description: HIP custom kernel wrappers for ROCm GPU execution (with CPU fallback)
#  Authors: Morteza Taleblou, Mitra Daneshmand
#  License: MIT (https://opensource.org/licenses/MIT)
#  Website: https://taleblou.ir/
# ============================================================================

from momijo.tensor.tensor import Tensor
from momijo.core.device import Device
from momijo.core.error import Error

# HIP elementwise add kernel (with CPU fallback)
fn hip_elemwise_add(alpha: Float64, A: Tensor, beta: Float64, B: Tensor, mut C: Tensor, device: Device) raises -> Error:
    assert A.numel() == B.numel() and B.numel() == C.numel(), "Size mismatch in hip_elemwise_add"
    var n = A.numel()

    if device.is_cpu():
        for i in range(n):
            var val = alpha * A.get_item(i) + beta * B.get_item(i)
            C.set_item(i, val)
    else:
        # Real HIP kernel launch would be invoked here
        # For now, fallback to CPU path
        return hip_elemwise_add(alpha, A, beta, B, C, Device("cpu"))

    return Error.ok()


# HIP elementwise multiply kernel (with CPU fallback)
fn hip_elemwise_mul(A: Tensor, B: Tensor, mut C: Tensor, device: Device) raises -> Error:
    assert A.numel() == B.numel() and B.numel() == C.numel(), "Size mismatch in hip_elemwise_mul"
    var n = A.numel()

    if device.is_cpu():
        for i in range(n):
            var val = A.get_item(i) * B.get_item(i)
            C.set_item(i, val)
    else:
        # Real HIP kernel launch would be invoked here
        # For now, fallback to CPU path
        return hip_elemwise_mul(A, B, C, Device("cpu"))

    return Error.ok()


# --- Minimal self-test for smoke testing ---
fn _self_test() -> Bool:
    var ok = True
    var dev = Device("cpu")
    var A = Tensor([1.0, 2.0, 3.0], shape=[3])
    var B = Tensor([4.0, 5.0, 6.0], shape=[3])
    var C = Tensor([0.0, 0.0, 0.0], shape=[3])

    try:
        var err1 = hip_elemwise_add(1.0, A, 1.0, B, C, dev)
    except e:
        return False
    if C.get_item(0) != 5.0 or C.get_item(2) != 9.0:
        ok = False

    try:
        var err2 = hip_elemwise_mul(A, B, C, dev)
    except e:
        return False
    if C.get_item(0) != 4.0 or C.get_item(2) != 18.0:
        ok = False

    return ok

 
