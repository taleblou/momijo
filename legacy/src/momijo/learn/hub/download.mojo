# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.hub.download
# File:         src/momijo/learn/hub/download.mojo
#
# Description:
#   Cached download utilities for Momijo Learn Hub.
#   - Backend-agnostic API to fetch weights/artifacts with an on-disk cache.
#   - Atomic commit (tmp → final), optional resumable hooks, integrity checks:
#       * FNV-1a 64 (hex)
#       * SHA-256  (hex)
#   - Pure Mojo implementation for hashing; I/O is abstracted via local shims.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from pathlib.path import Path
from collections.list import List

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------

struct DownloadOptions:
    var cache_enabled: Bool
    var retries: Int
    var timeout_sec: Int
    var allow_resume: Bool
    var expected_fnv64: String      # optional: "" means skip verification
    var expected_sha256: String     # optional: "" means skip verification
    var user_agent: String

    fn __init__(
        out self,
        cache_enabled: Bool = True,
        retries: Int = 2,
        timeout_sec: Int = 60,
        allow_resume: Bool = True,
        expected_fnv64: String = String(""),
        expected_sha256: String = String(""),
        user_agent: String = String("MomijoLearn/0.1")
    ):
        self.cache_enabled = cache_enabled
        self.retries = retries
        self.timeout_sec = timeout_sec
        self.allow_resume = allow_resume
        self.expected_fnv64 = expected_fnv64
        self.expected_sha256 = expected_sha256
        self.user_agent = user_agent


# -----------------------------------------------------------------------------
# Minimal FS shims (replace these with your project's real FS utils)
# -----------------------------------------------------------------------------
# NOTE: 
#  - If your project already has write_all_bytes/read_all_bytes etc., call them here.

fn _exists(p: Path) -> Bool:
    # If your Path has .exists(), use it; otherwise wire to your FS:
    # e.g., return exists(String(p))
    return p.exists()

fn _mkdirs(p: Path):
    # Create directory hierarchy if not present. 
    # Fallback: do nothing if Path API handles auto-create on write.
    pass

fn _remove(p: Path): 
    # This shim silently ignores errors to keep behavior side-effect free.
    pass

fn _write_bytes(p: Path, data: List[UInt8]):
    # If available: return write_all_bytes(String(p), data)
    # Otherwise implement with your FileHandle writer.
    pass

fn _read_bytes(p: Path) -> List[UInt8]:
    # If available: return read_all_bytes(String(p))
    var out = List[UInt8]()
    return out

fn _copy_file(src: Path, dst: Path):
    var data = _read_bytes(src)
    _write_bytes(dst, data)

fn _atomic_move(src: Path, dst: Path):
    # Prefer a real atomic rename if available; fallback to copy+remove.
    _copy_file(src, dst)
    _remove(src)


# -----------------------------------------------------------------------------
# Helpers: byte/hex utilities
# -----------------------------------------------------------------------------

@always_inline
fn _u8_to_string(b: UInt8) -> String:
    var s = String("")
    # Construct a single-character string from a byte (UTF-8 safe for ASCII hex)
    return s + String.from_utf8(List[UInt8]([b]))

fn _hex_of_u64(x: UInt64) -> String:
    var digits = "0123456789abcdef"
    var buf = List[UInt8]()
    var n = x
    if n == 0:
        buf.append(UInt8(digits[0].ord()))
    else:
        while n > 0:
            var idx = Int(n & 0xF)
            buf.append(UInt8(digits[idx].ord()))
            n = n >> 4
        # reverse
        var i: Int = 0
        var j: Int = len(buf) - 1
        while i < j:
            var t = buf[i]
            buf[i] = buf[j]
            buf[j] = t
            i = i + 1
            j = j - 1
    var s = String("")
    var k: Int = 0
    while k < len(buf):
        s = s + _u8_to_string(buf[k])
        k = k + 1
    return s

fn _hex_of_bytes(bs: List[UInt8]) -> String:
    var digits = "0123456789abcdef"
    var out = String("")
    var i: Int = 0
    while i < Int(bs):
        var v = UInt8(bs[i])
        var hi = Int((v >> 4) & 0xF)
        var lo = Int(v & 0xF)
        out = out + _u8_to_string(UInt8(digits[hi].ord()))
        out = out + _u8_to_string(UInt8(digits[lo].ord()))
        i = i + 1
    return out


# -----------------------------------------------------------------------------
# Checksums: FNV-1a 64 and SHA-256 (pure Mojo)
# -----------------------------------------------------------------------------

fn _fnv1a64(data: List[UInt8]) -> String:
    var hash: UInt64 = 0xcbf29ce484222325
    var prime: UInt64 = 0x100000001b3
    var i: Int = 0
    while i < len(data):
        hash = hash ^ UInt64(data[i])
        hash = hash * prime
        i = i + 1
    return _hex_of_u64(hash)

@always_inline
fn _rotr32(x: UInt32, n: Int) -> UInt32:
    return (x >> UInt32(n)) | (x << UInt32(32 - n))

@always_inline
fn _sha256_constants() -> List[UInt32]:
    # No globals: return constants via a local List on demand.
    return List[UInt32]([
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
    ])

fn _sha256(data: List[UInt8]) -> String:
    var h0: UInt32 = 0x6a09e667
    var h1: UInt32 = 0xbb67ae85
    var h2: UInt32 = 0x3c6ef372
    var h3: UInt32 = 0xa54ff53a
    var h4: UInt32 = 0x510e527f
    var h5: UInt32 = 0x9b05688c
    var h6: UInt32 = 0x1f83d9ab
    var h7: UInt32 = 0x5be0cd19

    # Padding
    var msg = List[UInt8]()
    var i: Int = 0
    while i < len(data):
        msg.append(data[i])
        i = i + 1
    msg.append(UInt8(0x80))
    while ((len(msg) % 64) != 56):
        msg.append(UInt8(0x00))
    var bit_len: UInt64 = UInt64(len(data)) * UInt64(8)
    var s: Int = 56
    while s >= 0:
        var b = UInt8((bit_len >> UInt64(s)) & UInt64(0xff))
        msg.append(b)
        s = s - 8

    # Process chunks
    var chunk: Int = 0
    while chunk < len(msg):
        # message schedule
        var w = List[UInt32]()
        var t: Int = 0
        while t < 16:
            var off = chunk + (t * 4)
            var val = (UInt32(msg[off]) << 24) | (UInt32(msg[off+1]) << 16) | (UInt32(msg[off+2]) << 8) | UInt32(msg[off+3])
            w.append(val)
            t = t + 1
        while t < 64:
            var s0 = _rotr32(w[t-15], 7) ^ _rotr32(w[t-15], 18) ^ (w[t-15] >> 3)
            var s1 = _rotr32(w[t-2], 17) ^ _rotr32(w[t-2], 19) ^ (w[t-2] >> 10)
            var val2 = (w[t-16] &+ s0 &+ w[t-7] &+ s1)
            w.append(val2)
            t = t + 1

        var a = h0
        var b = h1
        var c = h2
        var d = h3
        var e = h4
        var f = h5
        var g = h6
        var h = h7

        t = 0
        var K = _sha256_constants()
        while t < 64:
            var S1 = _rotr32(e, 6) ^ _rotr32(e, 11) ^ _rotr32(e, 25)
            var ch = (e & f) ^ ((~e) & g)
            var temp1 = h &+ S1 &+ ch &+ K[t] &+ w[t]
            var S0 = _rotr32(a, 2) ^ _rotr32(a, 13) ^ _rotr32(a, 22)
            var maj = (a & b) ^ (a & c) ^ (b & c)
            var temp2 = S0 &+ maj

            h = g
            g = f
            f = e
            e = d &+ temp1
            d = c
            c = b
            b = a
            a = temp1 &+ temp2
            t = t + 1

        h0 = h0 &+ a
        h1 = h1 &+ b
        h2 = h2 &+ c
        h3 = h3 &+ d
        h4 = h4 &+ e
        h5 = h5 &+ f
        h6 = h6 &+ g
        h7 = h7 &+ h

        chunk = chunk + 64

    # Serialize digest big-endian
    var digest = List[UInt8]()
    var hs = List[UInt32]([h0,h1,h2,h3,h4,h5,h6,h7])
    var k: Int = 0
    while k < 8:
        var v = hs[k]
        digest.append(UInt8((v >> 24) & 0xff))
        digest.append(UInt8((v >> 16) & 0xff))
        digest.append(UInt8((v >> 8) & 0xff))
        digest.append(UInt8(v & 0xff))
        k = k + 1
    return _hex_of_bytes(digest)


# -----------------------------------------------------------------------------
# URL helpers
# -----------------------------------------------------------------------------

fn _starts_with(s: String, prefix: String) -> Bool:
    if len(s) < len(prefix):
        return False
    var i: Int = 0
    while i < len(prefix):
        if s[i] != prefix[i]:
            return False
        i = i + 1
    return True

fn _basename_from_url(url: String) -> String:
    var last = String("")
    var i: Int = 0
    var part = String("")
    while i < len(url):
        var c = url[i]
        if c == '/':
            if len(part) > 0:
                last = part
            part = String("")
        else:
            part = part + _u8_to_string(UInt8(c))
        i = i + 1
    if len(part) > 0:
        last = part
    return last


# -----------------------------------------------------------------------------
# Network backend hook (TODO)
# -----------------------------------------------------------------------------

fn _download_http_to_file(url: String, tmp_path: Path, opts: DownloadOptions) -> Bool:
    # TODO: implement real HTTP(S) download with retries/timeout/user-agent.
    # Leave this as False for now (not implemented).
    return False


# -----------------------------------------------------------------------------
# Check & verify
# -----------------------------------------------------------------------------

fn _verify_checksums(data: List[UInt8], opts: DownloadOptions) -> Bool:
    if len(opts.expected_fnv64) > 0:
        var got = _fnv1a64(data)
        if got != opts.expected_fnv64:
            return False
    if len(opts.expected_sha256) > 0:
        var got2 = _sha256(data)
        if got2 != opts.expected_sha256:
            return False
    return True


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
# cached_download:
# - If path exists and (optional) checksum matches → return.
# - Else download/copy to temp and atomic-move to path.
# - Supports file:// now; http/https via hook to be implemented.

fn cached_download(url: String, path: String, opts: DownloadOptions = DownloadOptions()) -> Path:
    var dst = Path(path)

    # 1) Cache hit
    if _exists(dst):
        if not (len(opts.expected_fnv64) > 0 or len(opts.expected_sha256) > 0):
            return dst
        var data = _read_bytes(dst)
        if _verify_checksums(data, opts):
            return dst
        _remove(dst)  # evict and re-fetch

    # 2) if parent dir
    var parent = dst.parent()
    _mkdirs(parent)

    # 3) Temp path
    var tmp = Path(String(path) + String(".tmp"))

    # 4) Backend selection
    if _starts_with(url, String("file://")):
        # local copy: file:///abs/path/to/file
        var local = Path(url.substr(Int(7)))
        _copy_file(local, tmp)
    else:
        var ok = _download_http_to_file(url, tmp, opts)
        if not ok:
            _remove(tmp)
            return dst  # not created; caller can diagnose

    # 5) Integrity checks
    var data2 = _read_bytes(tmp)
    if not _verify_checksums(data2, opts):
        _remove(tmp)
        return dst

    # 6) Commit
    _atomic_move(tmp, dst)
    return dst
