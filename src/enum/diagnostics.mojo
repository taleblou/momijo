# MIT License
# Copyright (c) 2025 Morteza Talebou (https://taleblou.ir/)
# Module: momijo.enum.diagnostics

from momijo.enum.`match` import Case, RangeCase

fn _check_exhaustive(min_tag: Int, max_tag: Int, tags: List[Int], rstarts: List[Int], rends: List[Int]) -> Bool:
    if max_tag < min_tag:
        print("warning: empty domain")
        return True

    var size = (max_tag - min_tag) + 1
    var covered = List[Bool](length=size, fill=False)

    # precise tags
    for i in range(len(tags)):
        var t = tags[i]
        if t >= min_tag and t <= max_tag:
            covered[t - min_tag] = True

    # ranges
    var n = min(len(rstarts), len(rends))
    var idx = 0
    while idx < n:
        var s = rstarts[idx]
        var e = rends[idx]
        if e < s:
            var tmp = s
            s = e
            e = tmp
        if e >= min_tag and s <= max_tag:
            if s < min_tag: s = min_tag
            if e > max_tag: e = max_tag
            var k = s
            while k <= e:
                covered[k - min_tag] = True
                k += 1
        idx += 1

    var all_ok = True
    var j = 0
    while j < size:
        if not covered[j]:
            var miss = min_tag + j
            print("warning: uncovered tag ", miss)
            all_ok = False
        j += 1
    return all_ok

# Legacy API wrapper using Case/RangeCase
fn assert_exhaustive_or_warn(cases: List[Case], ranges: List[RangeCase], min_tag: Int, max_tag: Int) -> Bool:
    var tags = List[Int](capacity=len(cases))
    for i in range(len(cases)):
        tags.append(cases[i].tag)
    var rstarts = List[Int](capacity=len(ranges))
    var rends = List[Int](capacity=len(ranges))
    for j in range(len(ranges)):
        rstarts.append(ranges[j].start)
        rends.append(ranges[j].end)
    return _check_exhaustive(min_tag, max_tag, tags, rstarts, rends)

# Direct tags API
fn assert_exhaustive_or_warn_tags(tags: List[Int], rstarts: List[Int], rends: List[Int], min_tag: Int, max_tag: Int) -> Bool:
    return _check_exhaustive(min_tag, max_tag, tags, rstarts, rends)
