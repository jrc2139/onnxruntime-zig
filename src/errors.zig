//! Error handling for ONNX Runtime
//!
//! Maps OrtStatus to Zig errors with detailed error messages.

const std = @import("std");
const c_api = @import("c_api.zig");

/// Errors that can occur when using ONNX Runtime
pub const OrtError = error{
    /// General failure
    Fail,
    /// Invalid argument passed to API
    InvalidArgument,
    /// File not found
    NoSuchFile,
    /// Model not found or invalid
    NoModel,
    /// Execution engine error
    EngineError,
    /// Runtime exception occurred
    RuntimeException,
    /// Invalid protobuf format
    InvalidProtobuf,
    /// Model already loaded
    ModelLoaded,
    /// Feature not implemented
    NotImplemented,
    /// Invalid computation graph
    InvalidGraph,
    /// Execution provider failure
    ExecutionProviderFail,
    /// API not available
    ApiNotAvailable,
    /// Unknown error
    Unknown,
};

/// Convert OrtStatus to Zig error
pub fn checkStatus(api: *const c_api.OrtApi, status: ?*c_api.OrtStatus) OrtError!void {
    if (status) |s| {
        defer api.ReleaseStatus.?(s);

        const code = api.GetErrorCode.?(s);
        return switch (code) {
            c_api.c.ORT_OK => {},
            c_api.c.ORT_FAIL => OrtError.Fail,
            c_api.c.ORT_INVALID_ARGUMENT => OrtError.InvalidArgument,
            c_api.c.ORT_NO_SUCHFILE => OrtError.NoSuchFile,
            c_api.c.ORT_NO_MODEL => OrtError.NoModel,
            c_api.c.ORT_ENGINE_ERROR => OrtError.EngineError,
            c_api.c.ORT_RUNTIME_EXCEPTION => OrtError.RuntimeException,
            c_api.c.ORT_INVALID_PROTOBUF => OrtError.InvalidProtobuf,
            c_api.c.ORT_MODEL_LOADED => OrtError.ModelLoaded,
            c_api.c.ORT_NOT_IMPLEMENTED => OrtError.NotImplemented,
            c_api.c.ORT_INVALID_GRAPH => OrtError.InvalidGraph,
            c_api.c.ORT_EP_FAIL => OrtError.ExecutionProviderFail,
            else => OrtError.Unknown,
        };
    }
    // null status means success
}

/// Get error message from OrtStatus
pub fn getErrorMessage(api: *const c_api.OrtApi, status: *c_api.OrtStatus) []const u8 {
    const msg = api.GetErrorMessage.?(status);
    if (msg) |m| {
        return std.mem.span(m);
    }
    return "Unknown error";
}

/// Helper to wrap API calls that return OrtStatus
pub fn call(api: *const c_api.OrtApi, status: ?*c_api.OrtStatus) OrtError!void {
    return checkStatus(api, status);
}
