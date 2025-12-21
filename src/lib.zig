//! ONNX Runtime Zig Bindings
//!
//! A modern, idiomatic Zig interface to ONNX Runtime 1.23+.
//!
//! ## Quick Start
//!
//! ```zig
//! const ort = @import("onnxruntime");
//!
//! pub fn main() !void {
//!     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//!     defer _ = gpa.deinit();
//!     const allocator = gpa.allocator();
//!
//!     // Create environment (one per application)
//!     var env = try ort.Environment.init(.{});
//!     defer env.deinit();
//!
//!     // Load a model
//!     var session = try ort.Session.init(env, "model.onnx", null, allocator);
//!     defer session.deinit();
//!
//!     // Create input tensor
//!     var input_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
//!     const shape = [_]i64{ 1, 4 };
//!     var input = try ort.Tensor.fromSlice(f32, env.getApi(), &input_data, &shape);
//!     defer input.deinit();
//!
//!     // Run inference
//!     const outputs = try session.run(&[_]ort.Tensor{input});
//!     defer {
//!         for (outputs) |*out| out.deinit();
//!         allocator.free(outputs);
//!     }
//!
//!     // Get results
//!     const result = try outputs[0].getData(f32);
//!     std.debug.print("Output: {any}\n", .{result});
//! }
//! ```

const std = @import("std");

// Re-export public types
pub const c_api = @import("c_api.zig");
pub const errors = @import("errors.zig");

// Default types (backward compatible)
const env_mod = @import("env.zig");
const session_mod = @import("session.zig");
const tensor_mod = @import("tensor.zig");
const io_binding_mod = @import("io_binding.zig");
const run_options_mod = @import("run_options.zig");
const memory_info_mod = @import("memory_info.zig");

pub const Environment = env_mod.DefaultEnvironment;
pub const Session = session_mod.DefaultSession;
pub const SessionOptions = session_mod.DefaultSessionOptions;
pub const GraphOptLevel = session_mod.GraphOptLevel;
pub const AsyncCallback = Session.AsyncCallback;
pub const AsyncResult = Session.AsyncResult;
pub const Tensor = tensor_mod.DefaultTensor;

// Zero-allocation inference API
pub const IoBinding = io_binding_mod.DefaultIoBinding;
pub const RunOptions = run_options_mod.DefaultRunOptions;
pub const MemoryInfo = memory_info_mod.DefaultMemoryInfo;

// Generic factories for custom c_api modules
pub const GenericEnvironment = env_mod.Environment;
pub const GenericSession = session_mod.Session;
pub const GenericSessionOptions = session_mod.SessionOptions;
pub const GenericTensor = tensor_mod.Tensor;
pub const GenericIoBinding = io_binding_mod.IoBinding;
pub const GenericRunOptions = run_options_mod.RunOptions;
pub const GenericMemoryInfo = memory_info_mod.MemoryInfo;
pub const GenericErrors = errors.Errors;

// Execution providers
const provider = @import("provider.zig");
pub const ExecutionProvider = provider.DefaultExecutionProvider;
pub const GenericExecutionProvider = provider.ExecutionProvider;
pub const CoreMLOptions = provider.CoreMLOptions;
pub const CoreMLComputeUnits = provider.CoreMLComputeUnits;
pub const CoreMLModelFormat = provider.CoreMLModelFormat;
pub const CUDAOptions = provider.CUDAOptions;

// Re-export commonly used types from c_api
pub const OrtError = errors.OrtError;
pub const TensorElementType = c_api.TensorElementType;
pub const LoggingLevel = c_api.LoggingLevel;

/// Convenience helper: create all types for a custom c_api module.
///
/// This enables fastembed-zig (or any consumer with a custom c_api) to use
/// onnxruntime-zig's implementation with their own c_api module:
///
/// ```zig
/// const ort = @import("onnxruntime");
/// const my_c_api = @import("my_c_api.zig");  // with dynamic loading
///
/// // Use onnxruntime-zig's implementation with custom c_api
/// const ORT = ort.OnnxRuntime(my_c_api);
/// var env = try ORT.Environment.init(.{});
/// var session = try ORT.Session.init(env, "model.onnx", null, allocator);
/// ```
pub fn OnnxRuntime(comptime CApi: type) type {
    return struct {
        /// The c_api module this was instantiated with
        pub const c_api_module = CApi;

        /// Environment holds global state for ONNX Runtime
        pub const Environment = GenericEnvironment(CApi);

        /// Session represents a loaded model
        pub const Session = GenericSession(CApi);

        /// SessionOptions configures session behavior
        pub const SessionOptions = GenericSessionOptions(CApi);

        /// Tensor wraps OrtValue for input/output data
        pub const Tensor = GenericTensor(CApi);

        /// IoBinding enables zero-allocation inference
        pub const IoBinding = GenericIoBinding(CApi);

        /// RunOptions configures individual inference runs
        pub const RunOptions = GenericRunOptions(CApi);

        /// MemoryInfo describes memory allocation location
        pub const MemoryInfo = GenericMemoryInfo(CApi);

        /// ExecutionProvider selects hardware acceleration
        pub const ExecutionProvider = GenericExecutionProvider(CApi);

        /// Error handling utilities
        pub const Errors = GenericErrors(CApi);

        /// Async callback type for async inference
        pub const AsyncCallback = @This().Session.AsyncCallback;

        /// Async result type for async inference
        pub const AsyncResult = @This().Session.AsyncResult;
    };
}

/// Get the ONNX Runtime API version
pub fn getApiVersion() u32 {
    return c_api.ORT_API_VERSION;
}

/// Check if ONNX Runtime is available
pub fn isAvailable() bool {
    return c_api.getApi() != null;
}

test {
    // Run all module tests
    std.testing.refAllDecls(@This());
}

test "API is available" {
    try std.testing.expect(isAvailable());
}

test "API version is 23" {
    try std.testing.expectEqual(@as(u32, 23), getApiVersion());
}
