//! Zero-Allocation Inference Example
//!
//! This example demonstrates the IoBinding API for zero-allocation inference:
//! - Pre-allocate all tensors once
//! - Bind inputs/outputs once
//! - Run inference with zero allocations on the hot path
//!
//! Compare with basic_inference.zig which allocates on every run() call.
//!
//! Usage: zig build run-zero-alloc -- path/to/model.onnx

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
        std.debug.print("\nZero-allocation inference example using IoBinding.\n", .{});
        std.debug.print("Requires a model with known input/output shapes.\n", .{});
        return;
    }

    const model_path = args[1];

    if (!ort.isAvailable()) {
        std.debug.print("Error: ONNX Runtime not available\n", .{});
        return;
    }

    std.debug.print("=== Zero-Allocation Inference Demo ===\n\n", .{});
    std.debug.print("ONNX Runtime version: {d}\n", .{ort.getApiVersion()});

    // === SETUP PHASE (allocations allowed) ===

    // Create environment
    var env = ort.Environment.init(.{
        .log_level = .warning,
        .log_id = "zero-alloc",
    }) catch |err| {
        std.debug.print("Failed to create environment: {}\n", .{err});
        return;
    };
    defer env.deinit();

    const api = env.getApi();

    // Load model
    const model_path_z = allocator.allocSentinel(u8, model_path.len, 0) catch return;
    defer allocator.free(model_path_z);
    @memcpy(model_path_z, model_path);

    var session = ort.Session.init(env, model_path_z, null, allocator) catch |err| {
        std.debug.print("Failed to load model: {}\n", .{err});
        return;
    };
    defer session.deinit();

    std.debug.print("Model loaded: {s}\n", .{model_path});
    std.debug.print("  Inputs:  {d}\n", .{session.getInputCount()});
    std.debug.print("  Outputs: {d}\n", .{session.getOutputCount()});

    // Create IoBinding for zero-alloc inference
    var binding = ort.IoBinding.init(session) catch |err| {
        std.debug.print("Failed to create IoBinding: {}\n", .{err});
        return;
    };
    defer binding.deinit();

    std.debug.print("\nIoBinding created - ready for zero-alloc inference\n", .{});

    // Pre-allocate tensors (example with 4-element float tensors)
    // In real use, match these to your model's expected shapes
    var input_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var output_data = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const shape = [_]i64{ 1, 4 };

    var input_tensor = ort.Tensor.fromSlice(f32, api, &input_data, &shape) catch |err| {
        std.debug.print("Failed to create input tensor: {}\n", .{err});
        return;
    };
    defer input_tensor.deinit();

    var output_tensor = ort.Tensor.fromSlice(f32, api, &output_data, &shape) catch |err| {
        std.debug.print("Failed to create output tensor: {}\n", .{err});
        return;
    };
    defer output_tensor.deinit();

    // Bind tensors to names (requires knowing your model's input/output names)
    if (session.input_names.len > 0 and session.output_names.len > 0) {
        binding.bindInput(session.input_names[0], input_tensor) catch |err| {
            std.debug.print("Failed to bind input '{s}': {}\n", .{ session.input_names[0], err });
            std.debug.print("(Shape mismatch? This example uses shape [1,4] float32)\n", .{});
            return;
        };

        // For output, we can either:
        // 1. bindOutput() - pre-allocate output buffer (zero-copy)
        // 2. bindOutputToDevice() - let ORT allocate (for dynamic shapes)

        // Using pre-allocated output for true zero-alloc:
        binding.bindOutput(session.output_names[0], output_tensor) catch |err| {
            std.debug.print("Failed to bind output '{s}': {}\n", .{ session.output_names[0], err });
            std.debug.print("(Shape mismatch? This example uses shape [1,4] float32)\n", .{});
            return;
        };

        std.debug.print("\nBindings established:\n", .{});
        std.debug.print("  Input:  '{s}' -> pre-allocated f32[1,4]\n", .{session.input_names[0]});
        std.debug.print("  Output: '{s}' -> pre-allocated f32[1,4]\n", .{session.output_names[0]});

        // === HOT PATH (zero allocations) ===

        std.debug.print("\n=== Running inference (zero allocations) ===\n", .{});

        const iterations: usize = 10;
        var timer = std.time.Timer.start() catch {
            std.debug.print("Timer unavailable\n", .{});
            return;
        };

        for (0..iterations) |i| {
            // Update input in place (simulating new data each frame)
            input_data[0] = @floatFromInt(i);

            // Run inference - ZERO ALLOCATIONS
            binding.run(null) catch |err| {
                std.debug.print("Inference failed: {}\n", .{err});
                return;
            };

            // Output is already in output_data - ZERO COPY
        }

        const elapsed_ns = timer.read();
        const elapsed_us = @as(f64, @floatFromInt(elapsed_ns)) / 1000.0;
        const per_iter_us = elapsed_us / @as(f64, @floatFromInt(iterations));

        std.debug.print("\nCompleted {d} iterations\n", .{iterations});
        std.debug.print("  Total time:    {d:.2} us\n", .{elapsed_us});
        std.debug.print("  Per iteration: {d:.2} us\n", .{per_iter_us});
        std.debug.print("\nFinal output: {any}\n", .{output_data});
    } else {
        std.debug.print("\nModel has no inputs/outputs to bind\n", .{});
    }

    std.debug.print("\n=== Comparison ===\n", .{});
    std.debug.print("session.run():  5 allocations per call\n", .{});
    std.debug.print("binding.run():  0 allocations per call\n", .{});
}
