# ============================================================================
#  Project: Momijo
#  File: matmul_ref.mojo
#  Description: Reference matrix multiplication implementation for correctness testing
#  Authors: Morteza Taleblou, Mitra Daneshmand
#  License: MIT (https://opensource.org/licenses/MIT)
#  Website: https://taleblou.ir/
# ============================================================================

from momijo.tensor.tensor import Tensor
from momijo.core.error import Error

# Reference matrix multiplication: C = A x B
fn matmul_ref(A: Tensor, B: Tensor, mut C: Tensor) raises -> Error:
    assert A.shape().rank == 2 and B.shape().rank == 2 and C.shape().rank == 2, "All matrices must be 2D"
    var m = A.shape()[0]
    var k = A.shape()[1]
    var n = B.shape()[1]

    assert B.shape()[0] == k, "Inner dimension mismatch"
    assert C.shape()[0] == m and C.shape()[1] == n, "Output shape mismatch"

    for i in range(m):
        for j in range(n):
            var sum_val: Float64 = 0.0
            for p in range(k):
                var a_val = A.get_item(i * k + p)
                var b_val = B.get_item(p * n + j)
                sum_val += a_val * b_val
            C.set_item(i * n + j, sum_val)

    return Error.ok()


# --- Minimal self-test for smoke testing ---
fn _self_test() -> Bool:
    var ok = True

    var A = Tensor([1.0, 2.0,
                    3.0, 4.0], shape=[2,2])
    var B = Tensor([5.0, 6.0,
                    7.0, 8.0], shape=[2,2])
    var C = Tensor([0.0, 0.0,
                    0.0, 0.0], shape=[2,2])

    try:
        var err = matmul_ref(A, B, C)
    except e:
        return False

    # Expected result: [[19, 22], [43, 50]]
    if C.get_item(0) != 19.0 or C.get_item(1) != 22.0:
        ok = False
    if C.get_item(2) != 43.0 or C.get_item(3) != 50.0:
        ok = False

    return ok

 
