//! ONNX Runtime Memory Info
//!
//! MemoryInfo describes where tensor data resides (CPU, GPU, etc.)
//! Used with IoBinding for zero-copy device placement.

const std = @import("std");
const c_api = @import("c_api.zig");
const errors = @import("errors.zig");

const OrtError = errors.OrtError;

/// Memory information describing device and allocator type
pub const MemoryInfo = struct {
    ptr: *c_api.OrtMemoryInfo,
    api: *const c_api.OrtApi,

    const Self = @This();

    /// Create CPU memory info with default settings
    pub fn initCpu(api: *const c_api.OrtApi) OrtError!Self {
        var memory_info: ?*c_api.OrtMemoryInfo = null;
        const status = api.CreateCpuMemoryInfo.?(
            c_api.c.OrtArenaAllocator,
            c_api.c.OrtMemTypeDefault,
            &memory_info,
        );
        try errors.checkStatus(api, status);

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
    pub fn getPtr(self: Self) *c_api.OrtMemoryInfo {
        return self.ptr;
    }
};

test "MemoryInfo CPU creation" {
    const api = c_api.getApi() orelse return error.SkipZigTest;
    var mem_info = try MemoryInfo.initCpu(api);
    defer mem_info.deinit();
}
