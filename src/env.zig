//! ONNX Runtime Environment
//!
//! The Environment holds the global state for ONNX Runtime.
//! It should be created once and used for all sessions.
//! Supports generic c_api modules via the Environment(CApi) factory.

const std = @import("std");
const default_c_api = @import("c_api.zig");
const errors_mod = @import("errors.zig");

const OrtError = errors_mod.OrtError;

/// Generic Environment factory for any c_api module
pub fn Environment(comptime CApi: type) type {
    const Errs = errors_mod.Errors(CApi);

    return struct {
        ptr: *CApi.OrtEnv,
        api: *const CApi.OrtApi,

        const Self = @This();

        /// Initialize options for creating an environment
        pub const InitOptions = struct {
            /// Logging severity level
            log_level: CApi.LoggingLevel = .warning,
            /// Logger identifier (appears in log messages)
            log_id: [:0]const u8 = "onnxruntime-zig",
        };

        /// Create a new ONNX Runtime environment
        pub fn init(options: InitOptions) (OrtError || error{ApiNotAvailable})!Self {
            const api = CApi.getApi() orelse return error.ApiNotAvailable;

            var env: ?*CApi.OrtEnv = null;
            const status = api.CreateEnv.?(
                options.log_level.toC(),
                options.log_id.ptr,
                &env,
            );

            try Errs.checkStatus(api, status);

            return Self{
                .ptr = env.?,
                .api = api,
            };
        }

        /// Create environment with pre-obtained API pointer
        /// Use this when you already have the API from another source
        pub fn initWithApi(api: *const CApi.OrtApi, options: InitOptions) OrtError!Self {
            var env: ?*CApi.OrtEnv = null;
            const status = api.CreateEnv.?(
                options.log_level.toC(),
                options.log_id.ptr,
                &env,
            );

            try Errs.checkStatus(api, status);

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
        pub fn getApi(self: Self) *const CApi.OrtApi {
            return self.api;
        }

        /// Get the underlying OrtEnv pointer
        pub fn getPtr(self: Self) *CApi.OrtEnv {
            return self.ptr;
        }

        /// Enable telemetry collection
        pub fn enableTelemetry(self: Self) OrtError!void {
            const status = self.api.EnableTelemetryEvents.?(self.ptr);
            try Errs.checkStatus(self.api, status);
        }

        /// Disable telemetry collection
        pub fn disableTelemetry(self: Self) OrtError!void {
            const status = self.api.DisableTelemetryEvents.?(self.ptr);
            try Errs.checkStatus(self.api, status);
        }
    };
}

// =============================================================================
// Backward-compatible exports using default c_api
// =============================================================================

/// Default Environment type using built-in c_api (backward compatible)
pub const DefaultEnvironment = Environment(default_c_api);

test "create and destroy environment" {
    var env = try DefaultEnvironment.init(.{});
    defer env.deinit();

    // Environment pointer should not be null
    try std.testing.expect(@intFromPtr(env.ptr) != 0);
}

test "environment with custom log level" {
    var env = try DefaultEnvironment.init(.{
        .log_level = .info,
        .log_id = "test-env",
    });
    defer env.deinit();

    // API pointer should not be null
    try std.testing.expect(@intFromPtr(env.api) != 0);
}
