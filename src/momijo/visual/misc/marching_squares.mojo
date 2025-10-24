# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.math
# File: src/momijo/math/marching_squares.mojo
# Description: marching squares

fn marching_squares(grid: List[List[Float64]], level: Float64) -> List[List[List[Float64]]]:
    var lines = List[List[List[Float64]]]()
    var rows=len(grid); var cols=len(grid[0])
    var r=0
    while r<rows-1:
        var c=0
        while c<cols-1:
            var square=[grid[r][c],grid[r][c+1],grid[r+1][c+1],grid[r+1][c]]
            var idx=0; var k=0
            while k<4: if square[k]>level: idx|=(1<<k); k+=1
            if idx==0 or idx==15: c+=1; continue
            var poly=List[List[Float64]](); poly.append([Float64(c),Float64(r)]); poly.append([Float64(c+1),Float64(r+1)])
            lines.append(poly)
            c+=1
        r+=1
    return lines