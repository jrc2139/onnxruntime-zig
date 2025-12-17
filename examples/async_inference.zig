//! Async Inference Example
//!
//! Demonstrates non-blocking inference using session.runAsync().
//! The main thread can do other work while inference runs.
//!
//! Usage: zig build run-async -- path/to/model.onnx

const std = @import("std");
const ort = @import("onnxruntime");

/// Context for async callback
const InferenceContext = struct {
    completed: std.atomic.Value(bool),
    result: ?ort.AsyncResult,
    allocator: std.mem.Allocator,

    fn init(allocator: std.mem.Allocator) InferenceContext {
        return .{
            .completed = std.atomic.Value(bool).init(false),
            .result = null,
            .allocator = allocator,
        };
    }
};

/// Callback invoked when inference completes
fn onInferenceComplete(user_data: ?*anyopaque, result: ort.AsyncResult) void {
    const ctx: *InferenceContext = @ptrCast(@alignCast(user_data));
    ctx.result = result;
    ctx.completed.store(true, .release);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: {s} <model.onnx>\n", .{args[0]});
        std.debug.print("\nDemonstrates non-blocking async inference.\n", .{});
        return;
    }

    const model_path = args[1];

    if (!ort.isAvailable()) {
        std.debug.print("ONNX Runtime not available\n", .{});
        return;
    }

    std.debug.print("=== Async Inference Demo ===\n\n", .{});

    // Setup
    var env = ort.Environment.init(.{ .log_level = .warning }) catch |e| {
        std.debug.print("Failed to create env: {}\n", .{e});
        return;
    };
    defer env.deinit();

    const api = env.getApi();

    const model_path_z = allocator.allocSentinel(u8, model_path.len, 0) catch return;
    defer allocator.free(model_path_z);
    @memcpy(model_path_z, model_path);

    var session = ort.Session.init(env, model_path_z, null, allocator) catch |e| {
        std.debug.print("Failed to load model: {}\n", .{e});
        return;
    };
    defer session.deinit();

    std.debug.print("Model loaded: {s}\n", .{model_path});
    std.debug.print("  Input:  '{s}'\n", .{session.input_names[0]});
    std.debug.print("  Output: '{s}'\n\n", .{session.output_names[0]});

    // Create input tensor
    var input_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const shape = [_]i64{ 1, 4 };

    var input = ort.Tensor.fromSlice(f32, api, &input_data, &shape) catch |e| {
        std.debug.print("Failed to create tensor: {}\n", .{e});
        return;
    };
    defer input.deinit();

    // Create context for callback
    var ctx = InferenceContext.init(allocator);

    std.debug.print("Starting async inference...\n", .{});
    const start_time = std.time.milliTimestamp();

    // Start async inference
    session.runAsync(
        &[_]ort.Tensor{input},
        null,
        onInferenceComplete,
        &ctx,
    ) catch |e| {
        std.debug.print("Failed to start async inference: {}\n", .{e});
        return;
    };

    std.debug.print("Inference started (non-blocking)\n", .{});

    // Do other work while inference runs
    var work_iterations: usize = 0;
    while (!ctx.completed.load(.acquire)) {
        // Simulate other work
        work_iterations += 1;
        std.Thread.sleep(100); // 100ns
    }

    const end_time = std.time.milliTimestamp();
    const elapsed = end_time - start_time;

    std.debug.print("Inference completed!\n", .{});
    std.debug.print("  Time: {d} ms\n", .{elapsed});
    std.debug.print("  Work iterations during inference: {d}\n\n", .{work_iterations});

    // Process result
    if (ctx.result) |result| {
        switch (result) {
            .success => |outputs| {
                defer {
                    for (outputs) |*out| out.deinit();
                    allocator.free(outputs);
                }

                const data = outputs[0].getData(f32) catch {
                    std.debug.print("Failed to get output data\n", .{});
                    return;
                };
                std.debug.print("Input:  {any}\n", .{input_data});
                std.debug.print("Output: {any}\n", .{data});
            },
            .err => |e| {
                std.debug.print("Inference failed: {}\n", .{e});
            },
        }
    }

    std.debug.print("\n=== Async inference complete ===\n", .{});
}
