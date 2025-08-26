# MIT License
# Project: momijo.arrow_core
# File: momijo/arrow_core/poly_table.mojo
#
# Works with the current list-backed PolyColumn & PolyRecordBatch implementations.
# Tag selection is done via POLYTAG_* helpers imported from poly_column.

from momijo.arrow_core.poly_record_batch import PolyRecordBatch
from momijo.arrow_core.poly_column import (
    PolyColumn,
    POLYTAG_UNKNOWN,
    POLYTAG_INT,
    POLYTAG_F64,
    POLYTAG_STR,
)

fn __module_name__() -> String:
    return String("momijo/arrow_core/poly_table.mojo")

fn __self_test__() -> Bool:
    var t = PolyTable()
    return t.n_batches() == 0 and t.n_rows() == 0 and len(t) == 0

struct PolyTable(Copyable, Movable, Sized):
    var batches: List[PolyRecordBatch]

    # ---------- Constructors ----------
    fn __init__(out self):
        self.batches = List[PolyRecordBatch]()

    fn from_batches(mut self, bs: List[PolyRecordBatch]):
        self.batches = bs

    # ---------- Sized ----------
    fn __len__(self) -> Int:
        return self.n_rows()

    # ---------- Properties ----------
    fn n_batches(self) -> Int:
        return len(self.batches)

    fn n_rows(self) -> Int:
        var total: Int = 0
        var i = 0
        while i < len(self.batches):
            total += self.batches[i].n_rows()
            i += 1
        return total

    fn n_columns(self) -> Int:
        if len(self.batches) == 0:
            return 0
        return self.batches[0].n_columns()

    fn column_names(self) -> List[String]:
        if len(self.batches) == 0:
            return List[String]()
        return self.batches[0].column_names()

    # ---------- Access ----------
    fn get_batch(self, i: Int) -> PolyRecordBatch:
        if i < 0 or i >= len(self.batches):
            var dummy = PolyRecordBatch()
            return dummy
        return self.batches[i]

    fn get_row_as_strings(self, row: Int) -> List[String]:
        var offset: Int = 0
        var k = 0
        while k < len(self.batches):
            var b = self.batches[k]
            var n = b.n_rows()
            if row < offset + n:
                return b.get_row_as_strings(row - offset)
            offset += n
            k += 1
        return List[String]()

    fn get_column(self, name: String) -> List[String]:
        var out = List[String]()
        var bi = 0
        while bi < len(self.batches):
            var b = self.batches[bi]
            var col = b.get_column_by_name(name)
            var n = b.n_rows()
            var i = 0
            while i < n:
                if col.tag == POLYTAG_INT():
                    out.append(String(col.get_int(i)))
                elif col.tag == POLYTAG_F64():
                    out.append(String(col.get_f64(i)))
                elif col.tag == POLYTAG_STR():
                    out.append(col.get_string(i))
                else:
                    out.append(String(""))
                i += 1
            bi += 1
        return out

    # ---------- Mutation ----------
    fn add_batch(mut self, b: PolyRecordBatch):
        self.batches.append(b)

    fn clear(mut self):
        self.batches = List[PolyRecordBatch]()

    # ---------- Conversion ----------
    fn to_table_strings(self) -> List[List[String]]:
        var rows = List[List[String]]()
        var bi = 0
        while bi < len(self.batches):
            var batch_rows = self.batches[bi].to_table_strings()
            var i = 0
            while i < len(batch_rows):
                rows.append(batch_rows[i])
                i += 1
            bi += 1
        return rows
