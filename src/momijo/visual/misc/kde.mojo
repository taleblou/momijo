# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.math
# File: src/momijo/math/kde.mojo
# Description: KDE

fn kde_1d(samples: List[Float64], xs: List[Float64], bandwidth: Float64) -> List[Float64]:
    var out=List[Float64](); var i=0
    while i<len(xs):
        var s=0.0; var j=0
        while j<len(samples):
            var u=(xs[i]-samples[j])/bandwidth
            s+= (1.0/sqrt(2.0*3.14159)) * exp(-0.5*u*u)
            j+=1
        out.append(s/Float64(len(samples)*bandwidth))
        i+=1
    return out