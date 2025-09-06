# ============================================================================
#  Project: Momijo
#  File: threading.mojo
#  Description: Threading and parallel execution helpers for kernel implementations
#  Authors: Morteza Taleblou, Mitra Daneshmand
#  License: MIT (https://opensource.org/licenses/MIT)
#  Website: https://taleblou.ir/
# ============================================================================

from momijo.utils.thread_pool import ThreadPool

# Parallel for utility: executes fn(i) for i in [0, n)
fn parallel_for(n: Int, fn_body: fn(Int) -> None, num_threads: Int = 0):
    var pool = ThreadPool(num_threads)
    for i in range(n):
        pool.submit(fn_body, i)
    pool.join()


# Map utility: applies fn(x) to each element in list xs
fn parallel_map[T: Copyable & Movable, R: Copyable & Movable](xs: List[T], fn_body: fn(T) -> R, num_threads: Int = 0) -> List[R]:
    var results = List[R]()
    results.reserve(len(xs))
    var pool = ThreadPool(num_threads)
    for x in xs:
        pool.submit(lambda (arg: T) -> None:
            results.append(fn_body(arg)), x)
    pool.join()
    return results


# Reduce utility: applies fn(acc, x) in parallel to reduce list xs
fn parallel_reduce[T: Copyable & Movable](xs: List[T], fn_body: fn(T, T) -> T, init: T, num_threads: Int = 0) -> T:
    if len(xs) == 0:
        return init
    var acc = init
    var pool = ThreadPool(num_threads)
    for x in xs:
        pool.submit(lambda (a: T, b: T) -> None:
            acc = fn_body(a, b), acc, x)
    pool.join()
    return acc


# --- Minimal self-test for smoke testing ---
fn _self_test() -> Bool:
    var ok = True
    var data = [1, 2, 3, 4, 5]
    var results = parallel_map[Int, Int](data, lambda (x: Int) -> Int: x * 2, 2)
    if len(results) != len(data):
        ok = False
    return ok

 
