//! Error handling for ONNX Runtime
//!
//! Maps OrtStatus to Zig errors with detailed error messages.
//! Supports generic c_api modules via the Errors(CApi) factory.

const std = @import("std");
const default_c_api = @import("c_api.zig");

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

/// Generic error handling factory for any c_api module
pub fn Errors(comptime CApi: type) type {
    return struct {
        /// Convert OrtStatus to Zig error
        pub fn checkStatus(api: *const CApi.OrtApi, status: ?*CApi.OrtStatus) OrtError!void {
            if (status) |s| {
                defer api.ReleaseStatus.?(s);

                const code = api.GetErrorCode.?(s);
                return switch (code) {
                    CApi.ORT_OK => {},
                    CApi.ORT_FAIL => OrtError.Fail,
                    CApi.ORT_INVALID_ARGUMENT => OrtError.InvalidArgument,
                    CApi.ORT_NO_SUCHFILE => OrtError.NoSuchFile,
                    CApi.ORT_NO_MODEL => OrtError.NoModel,
                    CApi.ORT_ENGINE_ERROR => OrtError.EngineError,
                    CApi.ORT_RUNTIME_EXCEPTION => OrtError.RuntimeException,
                    CApi.ORT_INVALID_PROTOBUF => OrtError.InvalidProtobuf,
                    CApi.ORT_MODEL_LOADED => OrtError.ModelLoaded,
                    CApi.ORT_NOT_IMPLEMENTED => OrtError.NotImplemented,
                    CApi.ORT_INVALID_GRAPH => OrtError.InvalidGraph,
                    CApi.ORT_EP_FAIL => OrtError.ExecutionProviderFail,
                    else => OrtError.Unknown,
                };
            }
            // null status means success
        }

        /// Get error message from OrtStatus
        pub fn getErrorMessage(api: *const CApi.OrtApi, status_ptr: *CApi.OrtStatus) []const u8 {
            const msg = api.GetErrorMessage.?(status_ptr);
            if (msg) |m| {
                return std.mem.span(m);
            }
            return "Unknown error";
        }

        /// Helper to wrap API calls that return OrtStatus
        pub fn call(api: *const CApi.OrtApi, status: ?*CApi.OrtStatus) OrtError!void {
            return @This().checkStatus(api, status);
        }
    };
}

// =============================================================================
// Backward-compatible exports using default c_api
// =============================================================================

const DefaultErrors = Errors(default_c_api);

/// Convert OrtStatus to Zig error (backward compatible)
pub const checkStatus = DefaultErrors.checkStatus;

/// Get error message from OrtStatus (backward compatible)
pub const getErrorMessage = DefaultErrors.getErrorMessage;

/// Helper to wrap API calls that return OrtStatus (backward compatible)
pub const call = DefaultErrors.call;
