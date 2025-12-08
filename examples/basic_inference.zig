//! Basic ONNX Runtime inference example
//!
//! This example demonstrates how to:
//! 1. Create an ONNX Runtime environment
//! 2. Load an ONNX model
//! 3. Create input tensors
//! 4. Run inference
//! 5. Read output tensors
//!
//! Usage: zig build run -- path/to/model.onnx

const std = @import("std");
const ort = @import("onnxruntime");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Get model path from command line
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: {s} <model.onnx>\n", .{args[0]});
        std.debug.print("\nThis example loads an ONNX model and displays its input/output info.\n", .{});
        std.debug.print("For a simple test, try creating an 'add' model with two float inputs.\n", .{});
        return;
    }

    const model_path = args[1];

    // Check if ONNX Runtime is available
    if (!ort.isAvailable()) {
        std.debug.print("Error: ONNX Runtime not available\n", .{});
        return;
    }

    std.debug.print("ONNX Runtime version: {d}\n", .{ort.getApiVersion()});

    // Create environment
    std.debug.print("Creating environment...\n", .{});
    var env = ort.Environment.init(.{
        .log_level = .warning,
        .log_id = "basic-inference",
    }) catch |err| {
        std.debug.print("Failed to create environment: {}\n", .{err});
        return;
    };
    defer env.deinit();

    // Create session options with optimizations
    std.debug.print("Creating session options...\n", .{});
    var opts = ort.SessionOptions.init(env.getApi()) catch |err| {
        std.debug.print("Failed to create session options: {}\n", .{err});
        return;
    };
    defer opts.deinit();

    opts.setGraphOptimizationLevel(.all) catch {};
    opts.setIntraOpNumThreads(4) catch {};

    // Load the model - need null-terminated string
    std.debug.print("Loading model: {s}\n", .{model_path});

    const model_path_z = allocator.allocSentinel(u8, model_path.len, 0) catch {
        std.debug.print("Failed to allocate memory for model path\n", .{});
        return;
    };
    defer allocator.free(model_path_z);
    @memcpy(model_path_z, model_path);

    var session = ort.Session.init(env, model_path_z, opts, allocator) catch |err| {
        std.debug.print("Failed to load model: {}\n", .{err});
        return;
    };
    defer session.deinit();

    // Print model info
    std.debug.print("\nModel loaded successfully!\n", .{});
    std.debug.print("  Inputs:  {d}\n", .{session.getInputCount()});
    std.debug.print("  Outputs: {d}\n", .{session.getOutputCount()});

    // Print input names
    std.debug.print("\nInput names:\n", .{});
    for (session.input_names, 0..) |name, i| {
        std.debug.print("  [{d}] {s}\n", .{ i, name });
    }

    // Print output names
    std.debug.print("\nOutput names:\n", .{});
    for (session.output_names, 0..) |name, i| {
        std.debug.print("  [{d}] {s}\n", .{ i, name });
    }

    std.debug.print("\nTo run inference, you would:\n", .{});
    std.debug.print("  1. Create input tensors with Tensor.fromSlice()\n", .{});
    std.debug.print("  2. Call session.run() with the inputs\n", .{});
    std.debug.print("  3. Read outputs with tensor.getData()\n", .{});
}
