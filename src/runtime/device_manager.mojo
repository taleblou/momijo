# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.runtime
# File: src/momijo/runtime/device_manager.mojo

# Minimal runtime device manager: keeps track of available compute devices
# (CPU, GPU, accelerator backends). This is a simplified placeholder.

@fieldwise_init
struct Device:
    var id: Int
    var kind: String
    var name: String

    fn __init__(out self, id: Int, kind: String, name: String):
        self.id = id
        self.kind = kind
        self.name = name

    fn summary(self) -> String:
        return String("Device(") + String(self.id) + String(", ") + self.kind + String(", ") + self.name + String(")")


@fieldwise_init
struct DeviceManager:
    var devices: List[Device]

    fn __init__(out self):
        self.devices = List[Device]()

    fn add_device(mut self, d: Device):
        self.devices.push_back(d)

    fn count(self) -> Int:
        return len(self.devices)

    fn get(self, idx: Int) -> Device:
        return self.devices[idx]

    fn list_summaries(self) -> List[String]:
        var out = List[String]()
        var i = 0
        while i < len(self.devices):
            out.push_back(self.devices[i].summary())
            i += 1
        return out


fn _self_test() -> Bool:
    var mgr = DeviceManager()
    var cpu = Device(0, String("CPU"), String("Generic CPU"))
    mgr.add_device(cpu)
    var ok = True
    if mgr.count() < 1:
        ok = False
    if not mgr.get(0).kind == String("CPU"):
        ok = False
    if len(mgr.list_summaries()) == 0:
        ok = False
    return ok
