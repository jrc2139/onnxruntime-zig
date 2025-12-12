const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Build option for static linking
    const use_static = b.option(bool, "static", "Link against static ONNX Runtime libraries") orelse false;

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
    // Library Module
    // -------------------------------------------------------------------------
    const ort_mod = b.createModule(.{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add ONNX Runtime include path for @cImport
    ort_mod.addIncludePath(ort_include_path);

    // Link against ONNX Runtime
    ort_mod.addLibraryPath(ort_lib_path);

    if (use_static) {
        linkStaticOnnxRuntime(ort_mod, ort_abseil_path, target.result.os.tag == .macos);
    } else {
        ort_mod.linkSystemLibrary("onnxruntime", .{});
    }

    ort_mod.link_libc = true;

    // For macOS: link required frameworks
    if (target.result.os.tag == .macos) {
        ort_mod.linkFramework("Foundation", .{});
        ort_mod.linkFramework("CoreML", .{});
    }

    // Export the module for other packages
    _ = b.addModule("onnxruntime", .{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    // -------------------------------------------------------------------------
    // Tests
    // -------------------------------------------------------------------------
    const lib_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/lib.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    lib_tests.root_module.addIncludePath(ort_include_path);
    lib_tests.root_module.addLibraryPath(ort_lib_path);

    if (use_static) {
        linkStaticOnnxRuntime(lib_tests.root_module, ort_abseil_path, target.result.os.tag == .macos);
    } else {
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
    // Check (for ZLS)
    // -------------------------------------------------------------------------
    const check = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/lib.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    check.root_module.addIncludePath(ort_include_path);

    const check_step = b.step("check", "Check for compilation errors");
    check_step.dependOn(&check.step);
}

/// Link all static ONNX Runtime libraries
fn linkStaticOnnxRuntime(mod: *std.Build.Module, abseil_path: std.Build.LazyPath, is_macos: bool) void {
    _ = abseil_path; // Combined library includes abseil

    // Link the combined static library (includes ORT, abseil, protobuf, RE2, etc.)
    // Created by: libtool -static -o libonnxruntime_all.a *.a
    mod.linkSystemLibrary("onnxruntime_all", .{ .preferred_link_mode = .static });

    // Link C++ standard library (libc++ - must match ONNX Runtime build)
    // ONNX Runtime is built with clang -stdlib=libc++, so we must link libc++ not libstdc++
    mod.linkSystemLibrary("c++", .{});
    if (!is_macos) {
        // On Linux, libc++ also needs libc++abi
        mod.linkSystemLibrary("c++abi", .{});
    }
}
