//! ONNX Runtime Memory Info
//!
//! MemoryInfo describes where tensor data resides (CPU, GPU, etc.)
//! Used with IoBinding for zero-copy device placement.
//! Supports generic c_api modules via the MemoryInfo(CApi) factory.

const std = @import("std");
const default_c_api = @import("c_api.zig");
const errors_mod = @import("errors.zig");

const OrtError = errors_mod.OrtError;

/// Generic MemoryInfo factory for any c_api module
pub fn MemoryInfo(comptime CApi: type) type {
    const Errs = errors_mod.Errors(CApi);

    return struct {
        ptr: *CApi.OrtMemoryInfo,
        api: *const CApi.OrtApi,

        const Self = @This();

        /// Create CPU memory info with default settings
        pub fn initCpu(api: *const CApi.OrtApi) OrtError!Self {
            var memory_info: ?*CApi.OrtMemoryInfo = null;
            const status = api.CreateCpuMemoryInfo.?(
                CApi.OrtArenaAllocator,
                CApi.OrtMemTypeDefault,
                &memory_info,
            );
            try Errs.checkStatus(api, status);

            return Self{
                .ptr = memory_info.?,
                .api = api,
            };
        }

        /// Release memory info
        pub fn deinit(self: *Self) void {
            self.api.ReleaseMemoryInfo.?(self.ptr);
            self.ptr = undefined;
        }

        /// Get the underlying pointer
        pub fn getPtr(self: Self) *CApi.OrtMemoryInfo {
            return self.ptr;
        }
    };
}

// =============================================================================
// Backward-compatible exports using default c_api
// =============================================================================

/// Default MemoryInfo type using built-in c_api (backward compatible)
pub const DefaultMemoryInfo = MemoryInfo(default_c_api);

test "MemoryInfo CPU creation" {
    const api = default_c_api.getApi() orelse return error.SkipZigTest;
    var mem_info = try DefaultMemoryInfo.initCpu(api);
    defer mem_info.deinit();
}
