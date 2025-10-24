# Project:      Momijo
# Module:       dataframe.__init__
# File:         __init__.mojo
# Path:         dataframe/__init__.mojo
#
# Description:  dataframe.__init__ —   Init   module for Momijo DataFrame.
#               Implements core data structures, algorithms, and convenience APIs for production use.
#               Designed as a stable, composable building block within the Momijo public API.
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
#   - Structs: —
#   - Key functions: —

from momijo.dataframe.api import df_make as _df_make
from momijo.dataframe.column import Column as Column
from momijo.dataframe.column import Value,set_col_strings,col_from_list_with_tag
from momijo.dataframe.column import make_broadcast_col_by_name,make_broadcast_col_by_pos

from momijo.dataframe.utils import compute_stats as _compute_stats
from momijo.dataframe.utils import rolling_mean_f64,rolling_sum_f64,rolling_std_f64,expanding_mean_f64,ewm_var_f64,ewm_mean_f64,ewm_mean_f64_span,ewm_var_f64_span

from momijo.dataframe.datetime_ops import datetime_year as datetime_year
from momijo.dataframe.datetime_ops import datetime_year_df
from momijo.dataframe._groupby_core import groupby_transform as groupby_transform
from momijo.dataframe._groupby_core import Agg,pivot_table
# ---- Core types ----
from momijo.dataframe.frame import DataFrame as DataFrame

# ---- Series & helpers ---- 
from momijo.dataframe.series import series_from_list as series_from_list
from momijo.dataframe.helpers import isna as _isna

# Demo expects (values, name, index); real impl is (index, values, name)
from momijo.dataframe.api import df_from_pairs as df_from_pairs
from momijo.dataframe.api import df_from_columns as df_from_columns
from momijo.dataframe.api import df_shape as df_shape
from momijo.dataframe.api import df_dtypes as df_dtypes
from momijo.dataframe.api import DType,int32,int64,bool,float32,float64,string
from momijo.dataframe.api import Series,Index,ToDataFrame,df_describe,range,date_range,range_f64,ToDataFrameNullable

# from momijo.dataframe.api import to_category

# ---- Pairs helpers (functional, no inout) ----
# alias ColPair = (String, List[String])
# from momijo.dataframe.io_facade import read_csv as read_csv
# from momijo.dataframe.io_facade import to_csv as to_csv
# # JSONL hooks can be added when implemented:
# from momijo.dataframe.io_facade import to_json_lines as to_json_lines
# from momijo.dataframe.io_facade import read_json_lines as read_json_lines

# ---- Basic selection/filter ----
from momijo.dataframe.frame import head as head
from momijo.dataframe.missing import dropna_any as dropna_any
from momijo.dataframe.selection import select as select
from momijo.dataframe.selection import loc as loc
from momijo.dataframe.selection import iloc as iloc
from momijo.dataframe.selection import RowRange as RowRange
from momijo.dataframe.selection import ColRange as ColRange

from momijo.dataframe.selection import col_ge as col_ge
from momijo.dataframe.selection import col_isin as col_isin
from momijo.dataframe.selection import mask_and as mask_and
from momijo.dataframe.selection import filter as filter
from momijo.dataframe.selection import slice_labels,slice_rows,slice_cols
from momijo.dataframe.selection import rows_all,cols_all,rows,pslice
# Print-friendly variant of head
from momijo.dataframe.frame import set_index as set_index
from momijo.dataframe.frame import reset_index as reset_index

# Provide rename with Pythonic keywords used by the demo
from momijo.dataframe.api import df_rename as _df_rename
# Pandas-like: set a single cell. This stub returns the original df for demo stability.
# Returns a printable summary like:
# name: 0\nage: 2\n...
# Replace exact matches in a given column; returns a new DataFrame.
# Note: value is Float64; stored back as String for simplicity.
# Interpolates NA/empty cells linearly; leading/trailing NAs use nearest known value.
# Currently supports fmt="%Y-%m-%d" by normalizing to 10-char date.
# Supports 1 or 2 keys; ops: mean, max, count, nunique
# Returns a List[String] to be used with assign(...).

from momijo.dataframe.string_ops import col_str_concat as col_str_concat
from momijo.dataframe.series_str import df_label_where 
from momijo.dataframe.api import make_pairs

from momijo.dataframe.series import series_index
from momijo.dataframe.api import pairs_append
from momijo.dataframe.api import rename

from momijo.dataframe.api import copy,set_null
from momijo.dataframe.api import set_value
from momijo.dataframe.missing import isna_count_by_col
from momijo.dataframe.api import replace_values,coerce_str_to_f64
from momijo.dataframe.stats_core import col_mean
from momijo.dataframe._groupby_core import groupby 
from momijo.dataframe.string_ops import NaN
from momijo.dataframe.string_ops import NoneStr
from momijo.dataframe.categorical import to_category

from momijo.dataframe.api import to_datetime
from momijo.dataframe.api import take_rows
from momijo.dataframe.api import concat_rows

from momijo.dataframe.series_str import df_label_where as where

from momijo.dataframe.missing import fillna_value

from momijo.dataframe.missing import ffill

from momijo.dataframe.missing import interpolate_numeric

from momijo.dataframe.api import astype

from momijo.dataframe.api import concat_cols

from momijo.dataframe.api import melt

from momijo.dataframe.frame import sort_values

from momijo.dataframe.missing import drop_duplicates

from momijo.dataframe.api import cut_numeric



from momijo.dataframe.api import assign
from momijo.dataframe.string_ops import str_title
from momijo.dataframe.string_ops import str_upper
from momijo.dataframe.string_ops import str_contains
from momijo.dataframe.string_ops import str_slice
from momijo.dataframe.string_ops import str_extract
from momijo.dataframe.string_ops import str_replace_regex



from momijo.dataframe.join import merge,Join

 

# ---- Pairs helpers (functional, no inout) ----
# alias ColPair = (String, List[String])
from momijo.dataframe.io_csv import read_csv as read_csv
from momijo.dataframe.io_csv import write_csv as to_csv
# JSONL hooks can be added when implemented:
from momijo.dataframe.io_json_min import read_json as read_json
from momijo.dataframe.io_json_min import write_json as to_json
from momijo.dataframe.io_json_min import write_json as to_json_lines
from momijo.dataframe.io_json_min import read_json as read_json_lines

