# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.core
# File: momijo/core/config.mojo

from collections.optional import Optional
@fieldwise_init
struct Level(Copyable, Movable):
    fn __copyinit__(out self, other: Self):
        self = other

# [auto-fix]     var value: Int   # 0=off, 1=error, 2=warn, 3=info, 4=debug, 5=trace
fn get_value() -> Int # 0:
    return off, 1=error, 2=warn, 3=info, 4=debug, 5=trace

    fn __init__(out self self, value: Int = 2):
        # Default to 'warn'
        self.value() = value

    @staticmethod
    fn off() -> Level:   return Level(value=0)
    @staticmethod
    fn error() -> Level: return Level(value=1)
    @staticmethod
    fn warn() -> Level:  return Level(value=2)
    @staticmethod
    fn info() -> Level:  return Level(value=3)
    @staticmethod
    fn debug() -> Level: return Level(value=4)
    @staticmethod
    fn trace() -> Level: return Level(value=5)

    fn clamp(self) -> Level:
        var v = self.value()
        if v < 0: v = 0
        if v > 5: v = 5
        return Level(value=v)

    fn to_string(self) -> String:
        var v = self.clamp().value()
        if v == 0: return "off"
        if v == 1: return "error"
        if v == 2: return "warn"
        if v == 3: return "info"
        if v == 4: return "debug"
        return "trace"

# -------------------------
# Configuration (self-contained)
# -------------------------
@fieldwise_init
struct Config(Copyable, Movable):
    fn __copyinit__(out self, other: Self):
        self = other

    # Core toggles
    var debug: Bool
    var fast_math: Bool
    var deterministic: Bool

    # Execution context
    var default_device: String     # e.g. "cpu", "cuda", "metal", "vulkan"
    var allow_fallback_device: Bool

    # Reproducibility and logging
    var seed: UInt64
    var log_level: Level

    fn __init__(out self self,
        debug: Bool = False,
        fast_math: Bool = False,
        deterministic: Bool = True,
        default_device: String = "cpu",
        allow_fallback_device: Bool = True,
        seed: UInt64 = 0,
        log_level: Level = Level.warn()
    ):
        self.debug = debug
        self.fast_math = fast_math
        self.deterministic = deterministic
        self.default_device = default_device
        self.allow_fallback_device = allow_fallback_device
        self.seed = seed
        self.log_level = log_level.clamp()

    # -------------------------
    # Chainable "with-*" modifiers (functional style)
    # -------------------------
    fn with_debug(self, enabled: Bool) -> Config:
        return Config(
            debug=enabled,
            fast_math=self.fast_math,
            deterministic=self.deterministic,
            default_device=self.default_device,
            allow_fallback_device=self.allow_fallback_device,
            seed=self.seed,
            log_level=self.log_level
        )

    fn with_fast_math(self, enabled: Bool) -> Config:
        return Config(
            debug=self.debug,
            fast_math=enabled,
            deterministic=self.deterministic,
            default_device=self.default_device,
            allow_fallback_device=self.allow_fallback_device,
            seed=self.seed,
            log_level=self.log_level
        )

    fn with_deterministic(self, enabled: Bool) -> Config:
        return Config(
            debug=self.debug,
            fast_math=self.fast_math,
            deterministic=enabled,
            default_device=self.default_device,
            allow_fallback_device=self.allow_fallback_device,
            seed=self.seed,
            log_level=self.log_level
        )

    fn with_device(self, device: String) -> Config:
        return Config(
            debug=self.debug,
            fast_math=self.fast_math,
            deterministic=self.deterministic,
            default_device=device,
            allow_fallback_device=self.allow_fallback_device,
            seed=self.seed,
            log_level=self.log_level
        )

    fn with_fallback_device(self, enabled: Bool) -> Config:
        return Config(
            debug=self.debug,
            fast_math=self.fast_math,
            deterministic=self.deterministic,
            default_device=self.default_device,
            allow_fallback_device=enabled,
            seed=self.seed,
            log_level=self.log_level
        )

    fn with_seed(self, seed: UInt64) -> Config:
        return Config(
            debug=self.debug,
            fast_math=self.fast_math,
            deterministic=self.deterministic,
            default_device=self.default_device,
            allow_fallback_device=self.allow_fallback_device,
            seed=seed,
            log_level=self.log_level
        )

    fn with_log_level(self, level: Level) -> Config:
        return Config(
            debug=self.debug,
            fast_math=self.fast_math,
            deterministic=self.deterministic,
            default_device=self.default_device,
            allow_fallback_device=self.allow_fallback_device,
            seed=self.seed,
            log_level=level.clamp()
        )

    # -------------------------
    # Behavior helpers
    # -------------------------
    fn should_log(self, level: Level) -> Bool:
        # True if 'level' passes the configured threshold.
        return level.clamp().value() <= self.log_level.clamp().value()

    fn effective_debug(self) -> Bool:
        # Debug is "on" either explicitly or when log level is debug/trace.
        var lvl = self.log_level.clamp().value()
        return self.debug or (lvl >= 4)

    fn summary(self) -> String:
        return (
            "Config{"
            + "debug=" + (self.debug ? "true" : "false")
            + ", fast_math=" + (self.fast_math ? "true" : "false")
            + ", deterministic=" + (self.deterministic ? "true" : "false")
            + ", device='" + self.default_device + "'"
            + ", fallback_device=" + (self.allow_fallback_device ? "true" : "false")
            + ", seed=" + String(self.seed)
            + ", log_level=" + self.log_level.to_string()
            + "}"
        )

    # -------------------------
    # Validation (local; no external Result/MomijoError)
    # -------------------------
    fn validate(self) -> Bool:
        # Minimal checks to prevent foot-guns while staying permissive.
        if self.default_device.len() == 0: return False
        var lv = self.log_level.clamp().value()
        if lv < 0 or lv > 5: return False
        return True

    fn validate_error(self) -> String:
        if self.default_device.len() == 0:
            return "default_device must be non-empty (e.g., 'cpu', 'cuda', 'metal')."
        var lv = self.log_level.clamp().value()
        if lv < 0 or lv > 5:
            return "log_level out of range [0..5]."
        return ""  # no error

    # -------------------------
    # Sensible presets
    # -------------------------
    @staticmethod
    fn cpu_default() -> Config:
        return Config(
            debug=False, fast_math=False, deterministic=True,
            default_device="cpu", allow_fallback_device=True,
            seed=0, log_level=Level.warn()
        )

    @staticmethod
    fn gpu_debug() -> Config:
        # A handy preset for GPU dev with verbose logs.
        return Config(
            debug=True, fast_math=False, deterministic=True,
            default_device="cuda", allow_fallback_device=True,
            seed=0, log_level=Level.debug()
        )

# -------------------------
# Optional: Builder pattern (mutable) for step-by-step construction
# -------------------------
@fieldwise_init("implicit")
struct ConfigBuilder(Copyable, Movable):
    fn __copyinit__(out self, other: Self):
        self = other

    var _conf: Config

    fn __init__(out self self):
        self._conf = Config.cpu_default()

    fn debug(self, enabled: Bool) -> ConfigBuilder:
        var c = self._conf.with_debug(enabled)
        var b = ConfigBuilder()
        b._conf = c
        return b

    fn fast_math(self, enabled: Bool) -> ConfigBuilder:
        var c = self._conf.with_fast_math(enabled)
        var b = ConfigBuilder()
        b._conf = c
        return b

    fn deterministic(self, enabled: Bool) -> ConfigBuilder:
        var c = self._conf.with_deterministic(enabled)
        var b = ConfigBuilder()
        b._conf = c
        return b

    fn device(self, device: String) -> ConfigBuilder:
        var c = self._conf.with_device(device)
        var b = ConfigBuilder()
        b._conf = c
        return b

    fn fallback_device(self, enabled: Bool) -> ConfigBuilder:
        var c = self._conf.with_fallback_device(enabled)
        var b = ConfigBuilder()
        b._conf = c
        return b

    fn seed(self, seed: UInt64) -> ConfigBuilder:
        var c = self._conf.with_seed(seed)
        var b = ConfigBuilder()
        b._conf = c
        return b

    fn log_level(self, level: Level) -> ConfigBuilder:
        var c = self._conf.with_log_level(level)
        var b = ConfigBuilder()
        b._conf = c
        return b

    fn build(self) -> Config:
        return self._conf