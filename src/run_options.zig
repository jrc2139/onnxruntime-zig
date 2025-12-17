//! ONNX Runtime Run Options
//!
//! RunOptions control inference execution behavior including
//! run tags for profiling and termination of in-flight runs.

const std = @import("std");
const c_api = @import("c_api.zig");
const errors = @import("errors.zig");

const OrtError = errors.OrtError;

/// Options controlling inference execution
pub const RunOptions = struct {
    ptr: *c_api.OrtRunOptions,
    api: *const c_api.OrtApi,

    const Self = @This();

    /// Create run options with default settings
    pub fn init(api: *const c_api.OrtApi) OrtError!Self {
        var run_options: ?*c_api.OrtRunOptions = null;
        const status = api.CreateRunOptions.?(&run_options);
        try errors.checkStatus(api, status);

        return Self{
            .ptr = run_options.?,
            .api = api,
        };
    }

    /// Release run options
    pub fn deinit(self: *Self) void {
        self.api.ReleaseRunOptions.?(self.ptr);
        self.ptr = undefined;
    }

    /// Flag to terminate any currently executing Run calls
    pub fn setTerminate(self: Self) OrtError!void {
        const status = self.api.RunOptionsSetTerminate.?(self.ptr);
        try errors.checkStatus(self.api, status);
    }

    /// Clear the terminate flag
    pub fn unsetTerminate(self: Self) OrtError!void {
        const status = self.api.RunOptionsUnsetTerminate.?(self.ptr);
        try errors.checkStatus(self.api, status);
    }

    /// Set a tag for this run (useful for profiling)
    pub fn setRunTag(self: Self, tag: [:0]const u8) OrtError!void {
        const status = self.api.RunOptionsSetRunTag.?(self.ptr, tag.ptr);
        try errors.checkStatus(self.api, status);
    }

    /// Get the underlying pointer
    pub fn getPtr(self: Self) *c_api.OrtRunOptions {
        return self.ptr;
    }
};

test "RunOptions creation" {
    const api = c_api.getApi() orelse return error.SkipZigTest;
    var opts = try RunOptions.init(api);
    defer opts.deinit();
}

test "RunOptions set tag" {
    const api = c_api.getApi() orelse return error.SkipZigTest;
    var opts = try RunOptions.init(api);
    defer opts.deinit();

    try opts.setRunTag("test_run");
}
