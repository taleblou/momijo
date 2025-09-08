# Project:      Momijo
# Module:       src.momijo.dataframe.lazy
# File:         lazy.mojo
# Path:         src/momijo/dataframe/lazy.mojo
#
# Description:  src.momijo.dataframe.lazy — focused Momijo functionality with a stable public API.
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
#   - Key functions: make_df, main


from momijo.dataframe.column import Column
from momijo.dataframe.frame import DataFrame
from momijo.dataframe.lazy import LazyFrame
from momijo.dataframe.logical_plan import AggSpec

fn make_df() -> DataFrame:
    # Build a tiny DataFrame: cols: ["id", "grp", "val"]
    var c_id = Column.from_i64_name(String("id"), [1, 2, 3, 4, 5])
    var c_g  = Column.from_str_name(String("grp"), [String("A"), String("A"), String("B"), String("B"), String("B")])
    var c_v  = Column.from_f64_name(String("val"), [10.0, 11.0, 5.0, 7.0, 9.0])
    return DataFrame([String("id"), String("grp"), String("val")], [c_id, c_g, c_v])
fn main() -> None:
    var df = make_df()
    var lf = LazyFrame.scan(df)

    # filter + select
    var lf2 = lf.filter(Expr.col(String("val")).gt(Expr.lit_f64(8.0))).select([String("id"), String("grp"), String("val")])

    # groupby + aggregate
    var specs = List[AggSpec]()
    specs.push(AggSpec.sum(String("val"), String("sum_val")))
    specs.push(AggSpec.mean(String("val"), String("mean_val")))

    var gb = lf2.groupby([String("grp")]).agg(specs)

    # sort
    var out = gb.sort_by([String("sum_val")], [False]).collect()
    out.show()

    # window example: row_number within grp ordered by val desc
    var lf3 = lf.row_number(String("val"), [String("grp")])
    var out2 = lf3.collect()
    out2.show()