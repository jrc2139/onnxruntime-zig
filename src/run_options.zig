//! ONNX Runtime Run Options
//!
//! RunOptions control inference execution behavior including
//! run tags for profiling and termination of in-flight runs.
//! Supports generic c_api modules via the RunOptions(CApi) factory.

const std = @import("std");
const default_c_api = @import("c_api.zig");
const errors_mod = @import("errors.zig");

const OrtError = errors_mod.OrtError;

/// Generic RunOptions factory for any c_api module
pub fn RunOptions(comptime CApi: type) type {
    const Errs = errors_mod.Errors(CApi);

    return struct {
        ptr: *CApi.OrtRunOptions,
        api: *const CApi.OrtApi,

        const Self = @This();

        /// Create run options with default settings
        pub fn init(api: *const CApi.OrtApi) OrtError!Self {
            var run_options: ?*CApi.OrtRunOptions = null;
            const status = api.CreateRunOptions.?(&run_options);
            try Errs.checkStatus(api, status);

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
            try Errs.checkStatus(self.api, status);
        }

        /// Clear the terminate flag
        pub fn unsetTerminate(self: Self) OrtError!void {
            const status = self.api.RunOptionsUnsetTerminate.?(self.ptr);
            try Errs.checkStatus(self.api, status);
        }

        /// Set a tag for this run (useful for profiling)
        pub fn setRunTag(self: Self, tag: [:0]const u8) OrtError!void {
            const status = self.api.RunOptionsSetRunTag.?(self.ptr, tag.ptr);
            try Errs.checkStatus(self.api, status);
        }

        /// Get the underlying pointer
        pub fn getPtr(self: Self) *CApi.OrtRunOptions {
            return self.ptr;
        }
    };
}

// =============================================================================
// Backward-compatible exports using default c_api
// =============================================================================

/// Default RunOptions type using built-in c_api (backward compatible)
pub const DefaultRunOptions = RunOptions(default_c_api);

test "RunOptions creation" {
    const api = default_c_api.getApi() orelse return error.SkipZigTest;
    var opts = try DefaultRunOptions.init(api);
    defer opts.deinit();
}

test "RunOptions set tag" {
    const api = default_c_api.getApi() orelse return error.SkipZigTest;
    var opts = try DefaultRunOptions.init(api);
    defer opts.deinit();

    try opts.setRunTag("test_run");
}
