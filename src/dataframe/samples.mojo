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
# Project: momijo.dataframe
# File: src/momijo/dataframe/samples.mojo

#   from momijo.core.shape import append
#   from momijo.core.traits import append
#   from momijo.dataframe.series_bool import append
#   from momijo.dataframe.series_f64 import append
#   from momijo.dataframe.series_i64 import append
#   from momijo.dataframe.series_str import append
# SUGGEST (alpha): from momijo.core.shape import append
from momijo.extras.stubs import Copyright, MIT, SUGGEST, concat, customer_id, discount, discount_is_null, fact_city, from, https, len, momijo, order_id, product_id, quantity, return, returns, src
from momijo.arrow_core.array_stats import count
from momijo.dataframe.column import make_tiny_fact
from momijo.dataframe.column import build_orders
from momijo.dataframe.column import build_products
from momijo.dataframe.column import build_customers
from algorithm.reduction import map_reduce
from algorithm.reduction import any_true
from algorithm.reduction import all_true
from algorithm.reduction import cumsum
from algorithm.reduction import reduce
from algorithm.reduction import reduce_boolean
from algorithm.functional import vectorize
from momijo.dataframe.column import from_bool  # chosen by proximity
from momijo.dataframe.column import from_f64  # chosen by proximity
from momijo.dataframe.column import from_i64  # chosen by proximity
#   from momijo.dataframe.datetime_ops import gen_dates_from
#   from momijo.dataframe.helpers import gen_dates_from
# SUGGEST (alpha): from momijo.dataframe.datetime_ops import gen_dates_from
#   from momijo.arrow_core.array import len
#   from momijo.arrow_core.array_base import len
#   from momijo.arrow_core.arrays.boolean_array import len
#   from momijo.arrow_core.arrays.list_array import len
#   from momijo.arrow_core.arrays.primitive_array import len
#   from momijo.arrow_core.arrays.string_array import len
#   from momijo.arrow_core.bitmap import len
#   from momijo.arrow_core.buffer import len
#   from momijo.arrow_core.buffer_slice import len
#   from momijo.arrow_core.byte_string_array import len
#   from momijo.arrow_core.column import len
#   from momijo.arrow_core.offsets import len
#   from momijo.arrow_core.poly_column import len
#   from momijo.arrow_core.string_array import len
#   from momijo.core.types import len
#   from momijo.dataframe.column import len
#   from momijo.dataframe.index import len
#   from momijo.dataframe.series_bool import len
#   from momijo.dataframe.series_f64 import len
#   from momijo.dataframe.series_i64 import len
#   from momijo.dataframe.series_str import len
# SUGGEST (alpha): from momijo.arrow_core.array import len
from momijo.dataframe.column import from_str
from momijo.dataframe.frame import DataFrame
from momijo.dataframe.column import Column
from momijo.dataframe.series_i64 import SeriesI64
from momijo.dataframe.series_f64 import SeriesF64
from momijo.dataframe.series_str import SeriesStr
from momijo.dataframe.series_bool import SeriesBool, append
from momijo.dataframe.datetime_ops import gen_dates_12h_from_2025_01_01, gen_dates_from

# Build a small Customers DataFrame and return (df, ids, names, cities, segments)

var df = DataFrame(
        List[String]([String("customer_id"), String("name"), String("city"), String("segment")]),
        List[Column]([
            Column.from_i64(SeriesI64(String("customer_id"), ids)),
            Column.from_str(SeriesStr(String("name"), names)),
            Column.from_str(SeriesStr(String("city"), cities)),
            Column.from_str(SeriesStr(String("segment"), seg)),
        ])
    )
    return (df, ids, names, cities, seg)

# Build a small Products DataFrame and return (df, pid, category, brand, price)
fn build_products() -> (DataFrame, List[Int64], List[String], List[String], List[Float64])
    var pid = List[Int64]([Int64(101), Int64(102), Int64(103), Int64(104), Int64(105), Int64(106)])
    var cat = List[String]([String("Laptop"), String("Phone"), String("Tablet"), String("Accessory"), String("Accessory"), String("Laptop")])
    var brand = List[String]([String("Lenovo"), String("Apple"), String("Samsung"), String("Logitech"), String("Belkin"), String("HP")])
    var price = List[Float64]([1200.0, 999.0, 650.0, 45.0, 30.0, 1100.0])

    var df = DataFrame(
        List[String]([String("product_id"), String("category"), String("brand"), String("unit_price")]),
        List[Column]([
            Column.from_i64(SeriesI64(String("product_id"), pid)),
            Column.from_str(SeriesStr(String("categor

1_01(n)

    var order_id = List[Int64]()
    var i = 0
    while i < n:
        order_id.append(Int64(10_000 + i))
        i += 1

    var customer_id = List[Int64]()
    i = 0
    while i < n:
        customer_id.append(Int64((i % 10) + 1))
        i += 1

    var product_id = List[Int64]()
    var base_pid = List[Int64]([Int64(101), Int64(102), Int64(103), Int64(104), Int64(105), Int64(106)])
    i = 0
    while i < n:
        product_id.append(base_pid[i % 6])
        i += 1

    var quantity = List[Int64]()
    var base_q = List[Int64]([Int64(1), Int64(2), Int64(2), Int64(3), Int64(1), Int64(4)])
    i = 0
    while i < n:
        quantity.append(base_q[i % 6])
        i += 1

    var discount = List[Float64]()
    var base_d = List[Float64]([0.0, 0.05, 0.10, 0.0, 0.15, 0.0])
    i = 0
    while i < n:
        discount.append(base_d[i % 6])
        i += 1

    var discount_is_null = List[Bool]()
    i = 0
    while i < n:
        discount_is_null.append(False)
        i += 1
    if n > 5:
        discount_is_null[5] = True

    # duplicate row 7 (concat([... , iloc[[7]]])) only if enough rows
    if n > 7:

count.append(discount[j])
        discount_is_null.append(discount_is_null[j])
        ordered_at.append(ordered_at[j])

    var df = DataFrame(
        List[String]([String("order_id"), String("customer_id"), String("product_id"), String("quantity"), String("discount"), String("discount_is_null"), String("ordered_at")]),
        List[Column]([
            Column.from_i64(SeriesI64(String("order_id"), order_id)),
            Column.from_i64(SeriesI64(String("customer_id"), customer_id)),
            Column.from_i64(SeriesI64(String("product_id"), product_id)),
            Column.from_i64(SeriesI64(String("quantity"), quantity)),
            Column.from_f64(SeriesF64(String("discount"), discount)),
            Column.from_bool(SeriesBool(String("discount_is_null"), discount_is_null)),
            Column.from_str(SeriesStr(String("ordered_at"), ordered_at)),
        ])
    )
    return (df, order_id, customer_id, product_id, quantity, discount, discount_is_null, ordered_at)

# Build a small fact table; returns (order_id, ordered_at, city, category, qty, discount, revenue)
fn build_small_fact() raises -> (List[Int64], List[String], List[String], List[String], List[Int64], List[Float64], List[Float64])
    var cities = List[String]([String("Pori"), String("Tampere"), String("Helsinki"), String("Oulu"), String("Turku"), String("Espoo"), String("Espoo"), String("Helsinki"), String("Pori"), String("Tampere")])
    var cat    = List[String]([String("Laptop"), String("Phone"), String("Tablet"), String("Accessory"), String("Accessory"), String("Laptop")])
    var price  = List[Float64]([1200.0, 999.0, 650.0, 45.0, 30.0, 1100.0])
    var n = 8
    var order_id = List[Int64]()
    var customer_id = List[Int64]()
    var product_id = List[Int64]()
    var quantity = List[Int64]()
    var discount = List[Float64]()
    var ordered_at = gen_dates_from(1, 8, 15, n, 97)
    var i = 0
    while i < n:
        order_id.append(Int64(20000 + i))
        customer_id.append(Int64((i % 10) + 1))
        product_id.append(Int64(101 + (i % 6)))
        quantity.append(List[Int64]([Int64(1), Int64(2), Int64(2), Int64(3), Int64(1), Int64(4), Int64(1), Int64(2)])[i])
        discount.append(List[Float64]([0.0, 0.05, 0.10, 0.0, 0.15, 0.0, 0.0, 0.05])[i])
        i += 1
    var fact_city = List[String]()
    var fact_category = List[String]()
    var fact_unit_price = List[Float64]()
    i = 0
    while i < n:
        fact_city.append(cities[(Int(customer_id[i]) - 1) % len(cities)])
        var pid = Int(product_id[i]) - 101
        fact_category.append(cat[pid])
        fact_unit_price.append(price[pid])
        i += 1
    var fact_extended = List[Float64]()
    var fact_revenue = List[Float64]()
    i = 0
    while i < n:
        var ext = Float64(quantity[i]) * fact_unit_price[i]
        var rev = ext * (1.0 - discount[i])
        fact_extended.append(ext)
        fact_revenue.append(rev)
        i += 1
    return (order_id, ordered_at, fact_city, fact_category, quantity, discount, fact_revenue)

# Tiny fallback DataFrame with 3 rows
fn make_tiny_fact() -> DataFrame
    var names = List[String]([String("order_id"), String("revenue"), String("ordered_at")])

    var order_ids = List[String]([String("1"), String("2"), String("3")])
    var s_order_id = SeriesStr(String("order_id"), order_ids)
    var c_order_id = Column()
    c_order_id.from_str(s_order_id)

    var revenues = List[Float64]([10.5, 20.0, 5.25])
    var s_revenue = SeriesF64(String("revenue"), revenues)
    var c_revenue = Column()
    c_revenue.from_f64(s_revenue)

    var times = List[String]([String("2025-09-01T10:00:00"), String("2025-09-01T11:00:00"), String("2025-09-01T12:00:00")])
    var s_time = SeriesStr(String("ordered_at"), times)
    var c_time = Column()
    c_time.from_str(s_time)

    var cols = List[Column]([c_order_id, c_revenue, c_time])
    return DataFrame(names, cols)

