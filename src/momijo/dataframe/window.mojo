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
# File: src/momijo/dataframe/window.mojo

from momijo.core.error import module
from momijo.core.ndarray import product
from momijo.dataframe.column import Column, from_str
from momijo.dataframe.frame import DataFrame
from momijo.dataframe.logical_plan import window
from momijo.dataframe.rolling import rolling_mean
from momijo.dataframe.series_f64 import SeriesF64
from momijo.dataframe.series_i64 import SeriesI64
from momijo.dataframe.series_str import SeriesStr
from momijo.dataframe.window import cum_mean, cum_sum, dense_rank, lag, lead, rolling_mean, row_number
from pathlib import Path
from pathlib.path import Path
from utils.index import product

fn print_df(title: String, df: DataFrame, max_rows: Int = 20) -> None:
    print(String("\n== ") + title + String(" =="))
    print(df.to_string(max_rows))
fn main() -> None:
    var prod_vals = List[String]()
    prod_vals.append(String("A")); prod_vals.append(String("A"))
    prod_vals.append(String("A")); prod_vals.append(String("B"))
    prod_vals.append(String("B")); prod_vals.append(String("B"))

    var day_vals = List[Int64]()
    day_vals.append(1); day_vals.append(2); day_vals.append(3)
    day_vals.append(1); day_vals.append(2); day_vals.append(3)

    var sales_vals = List[Float64]()
    sales_vals.append(10.0); sales_vals.append(20.0); sales_vals.append(15.0)
    sales_vals.append(5.0);  sales_vals.append(7.0);  sales_vals.append(30.0)

    var s_prod = SeriesStr(String("product"), prod_vals)
    var s_day = SeriesI64(String("day"), day_vals)
    var s_sales = SeriesF64(String("sales"), sales_vals)

    var c_prod = Column(); c_prod.from_str(s_prod)
    var c_day = Column();  c_day.from_i64(s_day)
    var c_sales = Column(); c_sales.from_f64(s_sales)

    var names = List[String]()
    names.append(String("product"))
    names.append(String("day"))
    names.append(String("sales"))

    var cols = List[Column]()
    cols.append(c_prod)
    cols.append(c_day)
    cols.append(c_sales)

    var df = DataFrame(names, cols)

    print_df(String("Original DataFrame"), df)

    var df_rm = rolling_mean(df, String("sales"), 2, String("day"), List[String]([String("product")]))
    print_df(String("rolling_mean(sales, window=2, partition=product)"), df_rm)

    # 2) row_number
    var df_rn = row_number(df, String("day"), List[String]([String("product")]))
    print_df(String("row_number over product partition"), df_rn)

    # 3) lag(sales, 1)
    var df_lag = lag(df, String("sales"), 1, String("day"), List[String]([String("product")]), String("lag1"))
    print_df(String("lag(sales,1) over product partition"), df_lag)

    # 4) lead(sales, 1)
    var df_lead = lead(df, String("sales"), 1, String("day"), List[String]([String("product")]), String("lead1"))
    print_df(String("lead(sales,1) over product partition"), df_lead)

    # 5) cum_sum(sales)
    var df_cs = cum_sum(df, String("sales"), String("day"), List[String]([String("product")]), String("cumsum"))
    print_df(String("cumulative sum of sales"), df_cs)

    # 6) cum_mean(sales)
    var df_cm = cum_mean(df, String("sales"), String("day"), List[String]([String("product")]), String("cummean"))
    print_df(String("cumulative mean of sales"), df_cm)

    # 7) dense_rank(sales)
    var df_dr = dense_rank(df, String("sales"), List[String]([String("product")]), String("drank"))
    print_df(String("dense_rank of sales within product"), df_dr)