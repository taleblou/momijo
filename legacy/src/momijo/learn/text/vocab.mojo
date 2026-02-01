# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.text.vocab
# File:         src/momijo/learn/text/vocab.mojo
#
# Description:
#   Minimal Vocab without external deps.
#   - Keep insertion order: specials first, then unique tokens from iterator.
#   - Linear search stoi (no hashmap) – fine for small demos.
#   - Default index for OOV tokens.

from collections.list import List

struct Vocab(Copyable, Movable):
    var itos: List[String]      # index -> token
    var default_index: Int

    fn __init__(out self, specials: List[String] = List[String]()):
        self.itos = List[String]()
        # Insert specials (unique, in order)
        var i = 0
        while i < len(specials):
            var sp = specials[i]
            if not _contains(self.itos, sp):
                self.itos.append(sp)
            i = i + 1
        # Default to first special if exists; otherwise -1 (error sentinel)
        self.default_index = 0 if len(self.itos) > 0 else -1

    fn __copyinit__(out self, other: Self):
        self.itos = other.itos.copy()
        self.default_index = other.default_index

    fn _find_index(self, tok: String) -> Int:
        var n = len(self.itos)
        var i = 0
        while i < n:
            if self.itos[i] == tok:
                return i
            i = i + 1
        return -1

    fn add_token(mut self, tok: String):
        if self._find_index(tok) == -1:
            self.itos.append(tok)

    fn add_tokens(mut self, toks: List[String]):
        var i = 0
        var n = len(toks)
        while i < n:
            self.add_token(toks[i])
            i = i + 1

    fn set_default_index(mut self, idx: Int):
        self.default_index = idx

    fn get(self, tok: String) -> Int:
        var i = self._find_index(tok)
        return i if i != -1 else self.default_index

    fn lookup(self, toks: List[String]) -> List[Int]:
        var out = List[Int]()
        var n = len(toks)
        var i = 0
        while i < n:
            out.append(self.get(toks[i]))
            i = i + 1
        return out.copy()

# ----------------------------- builder ----------------------------------------

fn _contains(xs: List[String], s: String) -> Bool:
    var n = len(xs)
    var i = 0
    while i < n:
        if xs[i] == s:
            return True
        i = i + 1
    return False

fn build_vocab_from_iterator(
    token_sequences: List[List[String]],
    specials: List[String] = List[String]()
) -> Vocab:
    var v = Vocab(specials)
    var m = len(token_sequences)
    var i = 0
    while i < m:
        v.add_tokens(token_sequences[i])
        i = i + 1
    return v.copy()
