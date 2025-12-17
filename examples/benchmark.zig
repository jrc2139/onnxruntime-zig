//! Performance Benchmark: session.run() vs IoBinding.run()
//!
//! Compares allocation overhead between convenience API and zero-alloc API.
//!
//! Usage: zig build run-benchmark -- path/to/model.onnx [iterations]

const std = @import("std");
const ort = @import("onnxruntime");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: {s} <model.onnx> [iterations]\n", .{args[0]});
        std.debug.print("\nBenchmarks session.run() vs IoBinding.run()\n", .{});
        return;
    }

    const model_path = args[1];
    const iterations: usize = if (args.len >= 3)
        std.fmt.parseInt(usize, args[2], 10) catch 1000
    else
        1000;

    if (!ort.isAvailable()) {
        std.debug.print("ONNX Runtime not available\n", .{});
        return;
    }

    std.debug.print("╔══════════════════════════════════════════════════════╗\n", .{});
    std.debug.print("║     ONNX Runtime Inference Benchmark                 ║\n", .{});
    std.debug.print("╚══════════════════════════════════════════════════════╝\n\n", .{});

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

    std.debug.print("Model: {s}\n", .{model_path});
    std.debug.print("Iterations: {d}\n\n", .{iterations});

    // Pre-allocate data
    var input_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var output_data = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const shape = [_]i64{ 1, 4 };

    // ═══════════════════════════════════════════════════════════════════
    // Benchmark 1: session.run() - allocates per call
    // ═══════════════════════════════════════════════════════════════════
    {
        std.debug.print("─── session.run() (5 allocs/call) ───\n", .{});

        var input = ort.Tensor.fromSlice(f32, api, &input_data, &shape) catch |e| {
            std.debug.print("Failed to create tensor: {}\n", .{e});
            return;
        };
        defer input.deinit();

        // Warmup
        for (0..10) |_| {
            const outputs = session.run(&[_]ort.Tensor{input}) catch continue;
            for (outputs) |*out| out.deinit();
            allocator.free(outputs);
        }

        var timer = std.time.Timer.start() catch return;

        for (0..iterations) |i| {
            input_data[0] = @floatFromInt(i % 100);

            const outputs = session.run(&[_]ort.Tensor{input}) catch continue;

            // Must free outputs each call
            for (outputs) |*out| out.deinit();
            allocator.free(outputs);
        }

        const elapsed_ns = timer.read();
        const elapsed_us = @as(f64, @floatFromInt(elapsed_ns)) / 1000.0;
        const per_iter_us = elapsed_us / @as(f64, @floatFromInt(iterations));

        std.debug.print("  Total:     {d:.2} ms\n", .{elapsed_us / 1000.0});
        std.debug.print("  Per call:  {d:.2} us\n", .{per_iter_us});
        std.debug.print("  Allocs:    {d} (5 x {d})\n\n", .{ iterations * 5, iterations });
    }

    // ═══════════════════════════════════════════════════════════════════
    // Benchmark 2: IoBinding.run() - zero allocations
    // ═══════════════════════════════════════════════════════════════════
    {
        std.debug.print("─── binding.run() (0 allocs/call) ───\n", .{});

        var binding = ort.IoBinding.init(session) catch |e| {
            std.debug.print("Failed to create binding: {}\n", .{e});
            return;
        };
        defer binding.deinit();

        var input = ort.Tensor.fromSlice(f32, api, &input_data, &shape) catch return;
        defer input.deinit();

        var output = ort.Tensor.fromSlice(f32, api, &output_data, &shape) catch return;
        defer output.deinit();

        binding.bindInput(session.input_names[0], input) catch |e| {
            std.debug.print("Bind input failed: {}\n", .{e});
            return;
        };
        binding.bindOutput(session.output_names[0], output) catch |e| {
            std.debug.print("Bind output failed: {}\n", .{e});
            return;
        };

        // Warmup
        for (0..10) |_| {
            binding.run(null) catch continue;
        }

        var timer = std.time.Timer.start() catch return;

        for (0..iterations) |i| {
            input_data[0] = @floatFromInt(i % 100);
            binding.run(null) catch continue;
            // Output already in output_data - no cleanup needed!
        }

        const elapsed_ns = timer.read();
        const elapsed_us = @as(f64, @floatFromInt(elapsed_ns)) / 1000.0;
        const per_iter_us = elapsed_us / @as(f64, @floatFromInt(iterations));

        std.debug.print("  Total:     {d:.2} ms\n", .{elapsed_us / 1000.0});
        std.debug.print("  Per call:  {d:.2} us\n", .{per_iter_us});
        std.debug.print("  Allocs:    0\n\n", .{});
    }

    std.debug.print("═══════════════════════════════════════════════════════\n", .{});
}
