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

pub const Environment = @import("env.zig").Environment;
pub const Session = @import("session.zig").Session;
pub const SessionOptions = @import("session.zig").SessionOptions;
pub const GraphOptLevel = @import("session.zig").GraphOptLevel;
pub const AsyncCallback = Session.AsyncCallback;
pub const AsyncResult = Session.AsyncResult;
pub const Tensor = @import("tensor.zig").Tensor;

// Zero-allocation inference API
pub const IoBinding = @import("io_binding.zig").IoBinding;
pub const RunOptions = @import("run_options.zig").RunOptions;
pub const MemoryInfo = @import("memory_info.zig").MemoryInfo;

// Execution providers
const provider = @import("provider.zig");
pub const ExecutionProvider = provider.ExecutionProvider;
pub const CoreMLOptions = provider.CoreMLOptions;
pub const CoreMLComputeUnits = provider.CoreMLComputeUnits;
pub const CoreMLModelFormat = provider.CoreMLModelFormat;
pub const CUDAOptions = provider.CUDAOptions;

// Re-export commonly used types from c_api
pub const OrtError = errors.OrtError;
pub const TensorElementType = c_api.TensorElementType;
pub const LoggingLevel = c_api.LoggingLevel;

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
