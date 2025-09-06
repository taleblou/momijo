# ============================================================================
#  Momijo Visualization - spec/spec.mojo
#  Copyright (c) 2025  Morteza Talebou  (https://taleblou.ir/)
#  Licensed under the MIT License. See LICENSE in the project root.
# ============================================================================

struct Enc:
    var field: String
    var dtype: String  # "Q" quantitative, "N" nominal, "O" ordinal
    fn __init__(out self, s: String):
        var i = 0
        var colon = -1
        var n = len(s)
        while i < n:
            if s[i] == 58:  # ':'
                colon = i
                break
            i += 1
        if colon >= 0:
            self.field = String(s[0:colon])
            self.dtype = String(s[colon+1:n])
        else:
            self.field = s
            self.dtype = String("Q")

fn enc(spec: String) -> Enc:
    var e = Enc(spec)
    return e

struct MarkKind:
    var value: Int
    fn __init__(out self, value: Int):
        self.value = value

struct MarkKinds:
    @staticmethod
    fn point() -> MarkKind: return MarkKind(0)
    @staticmethod
    fn line() -> MarkKind:  return MarkKind(1)
    @staticmethod
    fn rect() -> MarkKind:  return MarkKind(2)

# Minimal columnar data reference
struct DataRef:
    var headers: List[String]
    var cols_num: Dict[String, List[Float64]]
    var cols_str: Dict[String, List[String]]

    fn __init__(out self):
        self.headers = List[String]()
        self.cols_num = Dict[String, List[Float64]]()
        self.cols_str = Dict[String, List[String]]()

    @staticmethod
    fn from_csv(path: String) -> DataRef:
        var d = DataRef()
        var f = open(path, String("r"))
        if f.is_null():
            return d
        var line = String()
        var row = 0
        while f.readline(out line):
            # simple CSV: comma-separated, no quotes/escapes
            if len(line) == 0: continue
            if line[-1] == 10: line = String(line[0:len(line)-1])  # trim LF
            var parts = line.split(String(","))
            if row == 0:
                for p in parts:
                    d.headers.append(p)
                    d.cols_num[p] = List[Float64]()
                    d.cols_str[p] = List[String]()
            else:
                var j = 0
                for p in parts:
                    var h = d.headers[j]
                    var ok = True
                    var v: Float64 = 0.0
                    # parse float
                    var k = 0
                    # very naive parse: use built-in converting constructor if available
                    try:
                        v = Float64(p)
                    except e:
                        ok = False
                    if ok:
                        d.cols_num[h].append(v)
                        d.cols_str[h].append(String(""))
                    else:
                        d.cols_num[h].append(0.0)
                        d.cols_str[h].append(p)
                    j += 1
            row += 1
        f.close()
        return d

    @staticmethod
    fn from_arrays(headers: List[String], data_num: Dict[String, List[Float64]], data_str: Dict[String, List[String]]) -> DataRef:
        var d = DataRef()
        d.headers = headers
        d.cols_num = data_num
        d.cols_str = data_str
        return d

struct Encodings:
    var x: Enc
    var y: Enc
    var color: Enc
    fn __init__(out self):
        self.x = Enc(String(""))
        self.y = Enc(String(""))
        self.color = Enc(String(""))

struct Spec:
    var data: DataRef
    var mark: MarkKind
    var enc: Encodings
    var width: Int
    var height: Int
    var padding: Int
    var theme: String

    fn __init__(out self, data: DataRef):
        self.data = data
        self.mark = MarkKinds.point()
        self.enc = Encodings()
        self.width = 800
        self.height = 600
        self.padding = 48
        self.theme = String("scientific")


struct LayerSpec:
    var mark: MarkKind
    var enc: Encodings
    fn __init__(out self):
        self.mark = MarkKinds.point()
        self.enc = Encodings()

struct FacetSpecSpec:
    var by: String
    var cols: Int
    fn __init__(out self):
        self.by = String("")
        self.cols = 2


    var layers: List[LayerSpec]
    var facet: FacetSpecSpec

        self.layers = List[LayerSpec]()
        self.facet = FacetSpecSpec()
