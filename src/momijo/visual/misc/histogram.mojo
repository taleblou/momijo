# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.math
# File: src/momijo/math/histogram.mojo
# Description: histogram binning

fn hist1d(samples: List[Float64], bins: Int, minv: Float64, maxv: Float64) -> List[Int]:
    var counts=List[Int](); var i=0
    while i<bins: counts.append(0); i+=1
    var scale=Float64(bins)/(maxv-minv)
    var j=0
    while j<len(samples):
        var idx=Int((samples[j]-minv)*scale)
        if idx>=0 and idx<bins: counts[idx]+=1
        j+=1
    return counts