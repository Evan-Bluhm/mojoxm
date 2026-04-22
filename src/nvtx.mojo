# ======================================================================
# Lightweight NVTX v3 wrapper for Nsight Systems timeline annotation
# ======================================================================
#
# Loads `libnvtx3interop.so.1` (or fallbacks) at runtime via dlopen.
# When the library is unavailable, push_range / pop_range / mark are
# silent no-ops, mirroring NVTX's native "enabled only when a profiler
# is attached" model.
#
# Usage:
#   var nvtx = NvtxContext()
#   nvtx.push_range("time loop")
#   ... work ...
#   nvtx.pop_range()
#   nvtx.mark("checkpoint written")
#
# Implementation note
# -------------------
# Mojo 0.26.2 only accepts the `thin` / `abi("C")` function-type
# modifiers inside template-parameter positions (e.g. the argument of
# `get_function[...]`), not as top-level type aliases or struct fields.
# So we cache the `OwnedDLHandle` and re-resolve each function pointer
# on every call.  dlsym is an O(1) hashtable lookup on the cached
# library handle -- cheap enough for the frequency of NVTX calls we
# issue (per-frame and per-stage, not per-element).
# ======================================================================

from std.ffi import OwnedDLHandle

struct NvtxContext(Movable):
    var _lib: OwnedDLHandle
    var _enabled: Bool

    def __init__(out self) raises:
        var candidates = [
            String("libnvtx3interop.so.1"),
            String("libnvtx3interop.so"),
            String("libnvToolsExt.so.1"),
            String("libnvToolsExt.so"),
        ]
        var opt_lib = Optional[OwnedDLHandle](None)
        for i in range(len(candidates)):
            try:
                opt_lib = OwnedDLHandle(candidates[i])
                break
            except:
                pass
        if opt_lib:
            self._lib = opt_lib.take()
            self._enabled = True
        else:
            self._lib = OwnedDLHandle()   # RTLD_DEFAULT placeholder
            self._enabled = False

    def push_range(mut self, name: String) raises:
        if not self._enabled:
            return
        var cstr = name + String("\0")
        var f = self._lib.get_function[
            def(UnsafePointer[UInt8, ImmutAnyOrigin]) -> Int32
        ]("nvtxRangePushA")
        _ = f(
            rebind[UnsafePointer[UInt8, ImmutAnyOrigin]](cstr.unsafe_ptr())
        )

    def pop_range(mut self) raises:
        if not self._enabled:
            return
        var f = self._lib.get_function[def() -> Int32]("nvtxRangePop")
        _ = f()

    def mark(mut self, name: String) raises:
        if not self._enabled:
            return
        var cstr = name + String("\0")
        var f = self._lib.get_function[
            def(UnsafePointer[UInt8, ImmutAnyOrigin]) -> NoneType
        ]("nvtxMarkA")
        f(
            rebind[UnsafePointer[UInt8, ImmutAnyOrigin]](cstr.unsafe_ptr())
        )

    def is_enabled(self) -> Bool:
        return self._enabled
