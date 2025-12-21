const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Build options (can be passed from parent project)
    const use_static = b.option(bool, "static", "Link against static ONNX Runtime libraries") orelse false;
    const cuda_enabled = b.option(bool, "cuda_enabled", "Enable CUDA support") orelse false;
    const coreml_enabled = b.option(bool, "coreml_enabled", "Enable CoreML support") orelse true;
    const dynamic_ort = b.option(bool, "dynamic_ort", "Load ONNX Runtime dynamically at runtime") orelse false;

    // Create build_options module
    const options = b.addOptions();
    options.addOption(bool, "cuda_enabled", cuda_enabled);
    options.addOption(bool, "coreml_enabled", coreml_enabled);
    options.addOption(bool, "dynamic_ort", dynamic_ort);
    const build_options_mod = options.createModule();

    // -------------------------------------------------------------------------
    // ONNX Runtime paths
    // -------------------------------------------------------------------------
    const ort_include_path = if (use_static)
        b.path("deps/onnxruntime-static/include")
    else
        b.path("deps/onnxruntime/include");

    const ort_lib_path = if (use_static)
        b.path("deps/onnxruntime-static/lib")
    else
        b.path("deps/onnxruntime/lib");

    const ort_abseil_path = b.path("deps/onnxruntime-static/lib/abseil");

    // -------------------------------------------------------------------------
    // Library Module (internal use for examples/tests)
    // -------------------------------------------------------------------------
    const ort_mod = b.createModule(.{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "build_options", .module = build_options_mod },
        },
    });

    // Add ONNX Runtime include path for @cImport
    ort_mod.addIncludePath(ort_include_path);

    // Link against ONNX Runtime
    ort_mod.addLibraryPath(ort_lib_path);

    if (use_static) {
        linkStaticOnnxRuntime(ort_mod, ort_abseil_path, target.result.os.tag == .macos);
    } else if (!dynamic_ort) {
        ort_mod.linkSystemLibrary("onnxruntime", .{});
    }

    ort_mod.link_libc = true;

    // For macOS: link required frameworks
    if (target.result.os.tag == .macos) {
        ort_mod.linkFramework("Foundation", .{});
        if (coreml_enabled) {
            ort_mod.linkFramework("CoreML", .{});
        }
    }

    // -------------------------------------------------------------------------
    // Export module for other packages (with build_options)
    // -------------------------------------------------------------------------
    const exported_mod = b.addModule("onnxruntime", .{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "build_options", .module = build_options_mod },
        },
    });
    exported_mod.addIncludePath(ort_include_path);

    // -------------------------------------------------------------------------
    // Tests
    // -------------------------------------------------------------------------
    const lib_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/lib.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "build_options", .module = build_options_mod },
            },
        }),
    });
    lib_tests.root_module.addIncludePath(ort_include_path);
    lib_tests.root_module.addLibraryPath(ort_lib_path);

    if (use_static) {
        linkStaticOnnxRuntime(lib_tests.root_module, ort_abseil_path, target.result.os.tag == .macos);
    } else if (!dynamic_ort) {
        lib_tests.root_module.linkSystemLibrary("onnxruntime", .{});
        lib_tests.root_module.addRPath(ort_lib_path);
    }
    lib_tests.root_module.link_libc = true;

    if (target.result.os.tag == .macos) {
        lib_tests.root_module.linkFramework("Foundation", .{});
    }

    const run_lib_tests = b.addRunArtifact(lib_tests);

    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(&run_lib_tests.step);

    // -------------------------------------------------------------------------
    // Example: Basic Inference
    // -------------------------------------------------------------------------
    const example = b.addExecutable(.{
        .name = "basic_inference",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/basic_inference.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "onnxruntime", .module = ort_mod },
            },
        }),
    });
    example.root_module.addIncludePath(ort_include_path);
    example.root_module.addLibraryPath(ort_lib_path);

    if (use_static) {
        linkStaticOnnxRuntime(example.root_module, ort_abseil_path, target.result.os.tag == .macos);
    } else {
        example.root_module.linkSystemLibrary("onnxruntime", .{});
        example.root_module.addRPath(ort_lib_path);
    }
    example.root_module.link_libc = true;

    if (target.result.os.tag == .macos) {
        example.root_module.linkFramework("Foundation", .{});
    }

    b.installArtifact(example);

    const run_example = b.addRunArtifact(example);
    run_example.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_example.addArgs(args);
    }

    const run_step = b.step("run", "Run the basic inference example");
    run_step.dependOn(&run_example.step);

    // -------------------------------------------------------------------------
    // Example: Zero-Alloc Inference (IoBinding)
    // -------------------------------------------------------------------------
    const zero_alloc_example = b.addExecutable(.{
        .name = "zero_alloc_inference",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/zero_alloc_inference.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "onnxruntime", .module = ort_mod },
            },
        }),
    });
    zero_alloc_example.root_module.addIncludePath(ort_include_path);
    zero_alloc_example.root_module.addLibraryPath(ort_lib_path);

    if (use_static) {
        linkStaticOnnxRuntime(zero_alloc_example.root_module, ort_abseil_path, target.result.os.tag == .macos);
    } else {
        zero_alloc_example.root_module.linkSystemLibrary("onnxruntime", .{});
        zero_alloc_example.root_module.addRPath(ort_lib_path);
    }
    zero_alloc_example.root_module.link_libc = true;

    if (target.result.os.tag == .macos) {
        zero_alloc_example.root_module.linkFramework("Foundation", .{});
    }

    b.installArtifact(zero_alloc_example);

    const run_zero_alloc = b.addRunArtifact(zero_alloc_example);
    run_zero_alloc.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_zero_alloc.addArgs(args);
    }

    const run_zero_alloc_step = b.step("run-zero-alloc", "Run the zero-allocation inference example");
    run_zero_alloc_step.dependOn(&run_zero_alloc.step);

    // -------------------------------------------------------------------------
    // Example: Benchmark (compares both approaches)
    // -------------------------------------------------------------------------
    const benchmark = b.addExecutable(.{
        .name = "benchmark",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/benchmark.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "onnxruntime", .module = ort_mod },
            },
        }),
    });
    benchmark.root_module.addIncludePath(ort_include_path);
    benchmark.root_module.addLibraryPath(ort_lib_path);

    if (use_static) {
        linkStaticOnnxRuntime(benchmark.root_module, ort_abseil_path, target.result.os.tag == .macos);
    } else {
        benchmark.root_module.linkSystemLibrary("onnxruntime", .{});
        benchmark.root_module.addRPath(ort_lib_path);
    }
    benchmark.root_module.link_libc = true;

    if (target.result.os.tag == .macos) {
        benchmark.root_module.linkFramework("Foundation", .{});
    }

    b.installArtifact(benchmark);

    const run_benchmark = b.addRunArtifact(benchmark);
    run_benchmark.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_benchmark.addArgs(args);
    }

    const run_benchmark_step = b.step("run-benchmark", "Run performance benchmark");
    run_benchmark_step.dependOn(&run_benchmark.step);

    // -------------------------------------------------------------------------
    // Example: Async Inference
    // -------------------------------------------------------------------------
    const async_example = b.addExecutable(.{
        .name = "async_inference",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/async_inference.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "onnxruntime", .module = ort_mod },
            },
        }),
    });
    async_example.root_module.addIncludePath(ort_include_path);
    async_example.root_module.addLibraryPath(ort_lib_path);

    if (use_static) {
        linkStaticOnnxRuntime(async_example.root_module, ort_abseil_path, target.result.os.tag == .macos);
    } else {
        async_example.root_module.linkSystemLibrary("onnxruntime", .{});
        async_example.root_module.addRPath(ort_lib_path);
    }
    async_example.root_module.link_libc = true;

    if (target.result.os.tag == .macos) {
        async_example.root_module.linkFramework("Foundation", .{});
    }

    b.installArtifact(async_example);

    const run_async = b.addRunArtifact(async_example);
    run_async.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_async.addArgs(args);
    }

    const run_async_step = b.step("run-async", "Run the async inference example");
    run_async_step.dependOn(&run_async.step);

    // -------------------------------------------------------------------------
    // Check (for ZLS)
    // -------------------------------------------------------------------------
    const check = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/lib.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "build_options", .module = build_options_mod },
            },
        }),
    });
    check.root_module.addIncludePath(ort_include_path);

    const check_step = b.step("check", "Check for compilation errors");
    check_step.dependOn(&check.step);
}

/// Link all static ONNX Runtime libraries
fn linkStaticOnnxRuntime(mod: *std.Build.Module, abseil_path: std.Build.LazyPath, is_macos: bool) void {
    _ = abseil_path; // Combined library includes abseil

    // Link the combined static library (includes ORT, abseil, protobuf, RE2, libc++, libc++abi, etc.)
    // Created by: ar -M with MRI script combining all .a files
    mod.linkSystemLibrary("onnxruntime_all", .{ .preferred_link_mode = .static });

    // On macOS, link system libc++ (not bundled in static lib)
    // On Linux, libc++ and libc++abi are bundled in the static library
    if (is_macos) {
        mod.linkSystemLibrary("c++", .{});
    }
}
