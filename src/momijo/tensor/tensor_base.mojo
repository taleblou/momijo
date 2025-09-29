struct TensorBase[T: Copyable & Movable](ExplicitlyCopyable, Movable):
    var _shape: List[Int]
    var _strides: List[Int]
    var _data: List[T]
    var _offset: Int
    var _dtype: DType

    fn __init__(out self, shape: List[Int], fill: T, dtype: DType = float64()):
        var size = shape_product(shape)
        self._shape = shape
        self._strides = compute_strides_rowmajor(shape)
        self._data = List[T]()
        var i = 0
        while i < size:
            self._data.append(fill)
            i += 1
        self._offset = 0
        self._dtype = dtype

    fn __copyinit__(out self, other: Self):
        self._shape = other._shape
        self._strides = other._strides
        self._data = other._data
        self._offset = other._offset
        self._dtype = other._dtype

    fn copy(self) -> Self:
        var out = Tensor[T](self._shape, self._data[0], self._dtype)
        out._data = List[T]()
        var i = 0
        while i < len(self._data):
            out._data.append(self._data[i])
            i += 1
        out._strides = self._strides
        out._offset = self._offset
        return out

    fn __str__(self) -> String:
        return "Tensor(shape=" + String(self._shape) + ", dtype=" + self._dtype.__str__() + ")"

    fn shape(self) -> List[Int]:
        return self._shape

    fn ndim(self) -> Int:
        return len(self._shape)

    fn size(self) -> Int:
        return shape_product(self._shape)

    fn get(self, idx: List[Int]) -> T:
        var off = index_to_offset(idx, self._shape, self._strides, self._offset)
        return self._data[off]

    fn set(mut self, idx: List[Int], v: T) -> None:
        var off = index_to_offset(idx, self._shape, self._strides, self._offset)
        self._data[off] = v

    fn get_flat(self, i: Int) -> T:
        return self._data[self._offset + i]

    fn set_flat(mut self, i: Int, v: T) -> None:
        self._data[self._offset + i] = v

    fn __getitem__(self, i: Int) -> T:
        assert len(self._shape) == 1
        return self._data[self._offset + i * self._strides[0]]

    fn __setitem__(mut self, i: Int, v: T) -> None:
        assert len(self._shape) == 1
        self._data[self._offset + i * self._strides[0]] = v
