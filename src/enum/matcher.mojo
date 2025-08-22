# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
struct Matcher:
    var tags: List[Int]
    var results: List[Int]

    fn __init__(out self, tags: List[Int], results: List[Int]):
        self.tags = tags
        self.results = results

fn build_matcher(tags: List[Int], results: List[Int]) -> Matcher:
    return Matcher(tags=tags, results=results)

fn match_with_selector(selector: Int, default: Int, m: Matcher) -> Int:
    for i in range(len(m.tags)):
        if m.tags[i] == selector:
            return m.results[i]
    return default