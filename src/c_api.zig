//! Raw C bindings for ONNX Runtime C API
//!
//! This module provides direct access to the ONNX Runtime C API via @cImport.
//! For a higher-level idiomatic Zig API, use the main `onnxruntime` module.

pub const c = @cImport({
    @cInclude("onnxruntime_c_api.h");
});

// Re-export commonly used types for convenience
pub const OrtApi = c.OrtApi;
pub const OrtApiBase = c.OrtApiBase;
pub const OrtEnv = c.OrtEnv;
pub const OrtSession = c.OrtSession;
pub const OrtSessionOptions = c.OrtSessionOptions;
pub const OrtValue = c.OrtValue;
pub const OrtMemoryInfo = c.OrtMemoryInfo;
pub const OrtAllocator = c.OrtAllocator;
pub const OrtStatus = c.OrtStatus;
pub const OrtRunOptions = c.OrtRunOptions;
pub const OrtTypeInfo = c.OrtTypeInfo;
pub const OrtTensorTypeAndShapeInfo = c.OrtTensorTypeAndShapeInfo;
pub const OrtIoBinding = c.OrtIoBinding;

/// Callback function type for RunAsync
pub const RunAsyncCallbackFn = *const fn (
    user_data: ?*anyopaque,
    outputs: [*c]?*OrtValue,
    num_outputs: usize,
    status: ?*OrtStatus,
) callconv(.c) void;

// Enums
pub const OrtErrorCode = c.OrtErrorCode;
pub const OrtLoggingLevel = c.OrtLoggingLevel;
pub const ONNXTensorElementDataType = c.ONNXTensorElementDataType;
pub const ONNXType = c.ONNXType;
pub const OrtMemType = c.OrtMemType;
pub const OrtAllocatorType = c.OrtAllocatorType;
pub const GraphOptimizationLevel = c.GraphOptimizationLevel;
pub const ExecutionMode = c.ExecutionMode;

// Constants
pub const ORT_API_VERSION = c.ORT_API_VERSION;

/// Get the ONNX Runtime API base
pub fn getApiBase() *const OrtApiBase {
    return c.OrtGetApiBase();
}

/// Get the ONNX Runtime API for the current version
pub fn getApi() ?*const OrtApi {
    const base = getApiBase();
    return base.GetApi.?(ORT_API_VERSION);
}

// Common tensor element types for convenience
pub const TensorElementType = enum(c_int) {
    undefined = c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,
    float32 = c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    uint8 = c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
    int8 = c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
    uint16 = c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
    int16 = c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
    int32 = c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
    int64 = c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    string = c.ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
    bool_ = c.ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
    float16 = c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
    float64 = c.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
    uint32 = c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
    uint64 = c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
    bfloat16 = c.ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,

    pub fn toC(self: TensorElementType) ONNXTensorElementDataType {
        return @intCast(@intFromEnum(self));
    }
};

pub const LoggingLevel = enum(c_uint) {
    verbose = c.ORT_LOGGING_LEVEL_VERBOSE,
    info = c.ORT_LOGGING_LEVEL_INFO,
    warning = c.ORT_LOGGING_LEVEL_WARNING,
    err = c.ORT_LOGGING_LEVEL_ERROR,
    fatal = c.ORT_LOGGING_LEVEL_FATAL,

    pub fn toC(self: LoggingLevel) c_uint {
        return @intFromEnum(self);
    }
};

pub const ErrorCode = enum(c_int) {
    ok = c.ORT_OK,
    fail = c.ORT_FAIL,
    invalid_argument = c.ORT_INVALID_ARGUMENT,
    no_such_file = c.ORT_NO_SUCHFILE,
    no_model = c.ORT_NO_MODEL,
    engine_error = c.ORT_ENGINE_ERROR,
    runtime_exception = c.ORT_RUNTIME_EXCEPTION,
    invalid_protobuf = c.ORT_INVALID_PROTOBUF,
    model_loaded = c.ORT_MODEL_LOADED,
    not_implemented = c.ORT_NOT_IMPLEMENTED,
    invalid_graph = c.ORT_INVALID_GRAPH,
    ep_fail = c.ORT_EP_FAIL,
};

test "can get API" {
    const api = getApi();
    try @import("std").testing.expect(api != null);
}

test "API version is 23" {
    try @import("std").testing.expectEqual(@as(u32, 23), ORT_API_VERSION);
}
