# ============================================================================
#  Momijo Visualization - spec/api.mojo
#  Copyright (c) 2025  Morteza Talebou  (https://taleblou.ir/)
#  Licensed under the MIT License. See LICENSE in the project root.
# ============================================================================

from momijo.visual.spec.spec import Spec, DataRef, Enc, enc, MarkKind, MarkKinds
from momijo.visual.spec.compiler import compile_spec_to_scene
from momijo.visual.runtime.backend_select import select_backend_from_ext, BackendKinds
from momijo.visual.render.svg.svg_backend import render_scene
from momijo.visual.render.raster.png_backend import render_scene_raster
from momijo.visual.spec.exporters.vega_lite_json import to_vega_lite
from momijo.visual.spec.exporters.plotly_json import to_plotly

struct Chart:
    var _spec: Spec

    fn __init__(out self, data: DataRef):
        self._spec = Spec(data)

    fn mark(mut self, kind: MarkKind) -> Chart:
        self._spec.mark = kind
        return self

    fn encode(mut self, x: Enc, y: Enc, color: Enc = Enc(String(""))) -> Chart:
        self._spec.enc.x = x
        self._spec.enc.y = y
        self._spec.enc.color = color
        return self

    fn size(mut self, width: Int, height: Int) -> Chart:
        self._spec.width = width
        self._spec.height = height
        return self

    fn theme(mut self, name: String) -> Chart:
        self._spec.theme = name
        return self

    fn save(self, path: String):
        let scene = compile_spec_to_scene(self._spec)
        let backend = select_backend_from_ext(path)
        if backend.value == BackendKinds.svg().value:
            render_scene(scene, backend, path)
        else:
            render_scene_raster(scene, backend, path)

    fn to_vega_lite_json(self) -> String:
        return to_vega_lite(self._spec)

    fn to_plotly_json(self) -> String:
        return to_plotly(self._spec)


    fn layer(mut self, mark: MarkKind, x: Enc, y: Enc, color: Enc = Enc(String(""))) -> Chart:
        var L = LayerSpec()
        L.mark = mark
        L.enc.x = x; L.enc.y = y; L.enc.color = color
        self._spec.layers.append(L)
        return self

    fn facet_by(mut self, by: String, cols: Int = 2) -> Chart:
        self._spec.facet.by = by
        self._spec.facet.cols = cols
        return self
