//! ONNX Runtime Environment
//!
//! The Environment holds the global state for ONNX Runtime.
//! It should be created once and used for all sessions.

const std = @import("std");
const c_api = @import("c_api.zig");
const errors = @import("errors.zig");

const OrtError = errors.OrtError;

/// ONNX Runtime Environment
///
/// The environment maintains the global threadpool and other resources.
/// Create one environment per application and reuse it for all sessions.
pub const Environment = struct {
    ptr: *c_api.OrtEnv,
    api: *const c_api.OrtApi,

    const Self = @This();

    /// Initialize options for creating an environment
    pub const InitOptions = struct {
        /// Logging severity level
        log_level: c_api.LoggingLevel = .warning,
        /// Logger identifier (appears in log messages)
        log_id: [:0]const u8 = "onnxruntime-zig",
    };

    /// Create a new ONNX Runtime environment
    pub fn init(options: InitOptions) (OrtError || error{ApiNotAvailable})!Self {
        const api = c_api.getApi() orelse return error.ApiNotAvailable;

        var env: ?*c_api.OrtEnv = null;
        const status = api.CreateEnv.?(
            options.log_level.toC(),
            options.log_id.ptr,
            &env,
        );

        try errors.checkStatus(api, status);

        return Self{
            .ptr = env.?,
            .api = api,
        };
    }

    /// Release the environment
    pub fn deinit(self: *Self) void {
        self.api.ReleaseEnv.?(self.ptr);
        self.ptr = undefined;
    }

    /// Get the underlying OrtApi pointer
    pub fn getApi(self: Self) *const c_api.OrtApi {
        return self.api;
    }

    /// Get the underlying OrtEnv pointer
    pub fn getPtr(self: Self) *c_api.OrtEnv {
        return self.ptr;
    }

    /// Enable telemetry collection
    pub fn enableTelemetry(self: Self) OrtError!void {
        const status = self.api.EnableTelemetryEvents.?(self.ptr);
        try errors.checkStatus(self.api, status);
    }

    /// Disable telemetry collection
    pub fn disableTelemetry(self: Self) OrtError!void {
        const status = self.api.DisableTelemetryEvents.?(self.ptr);
        try errors.checkStatus(self.api, status);
    }
};

test "create and destroy environment" {
    var env = try Environment.init(.{});
    defer env.deinit();

    // Environment pointer should not be null
    try std.testing.expect(@intFromPtr(env.ptr) != 0);
}

test "environment with custom log level" {
    var env = try Environment.init(.{
        .log_level = .info,
        .log_id = "test-env",
    });
    defer env.deinit();

    // API pointer should not be null
    try std.testing.expect(@intFromPtr(env.api) != 0);
}
