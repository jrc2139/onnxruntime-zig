//! Execution Provider abstraction for ONNX Runtime
//!
//! Provides configurable execution providers (CPU, CoreML, CUDA) with
//! automatic platform detection.
//! Supports generic c_api modules via the ExecutionProvider(CApi) factory.

const std = @import("std");
const builtin = @import("builtin");
const default_c_api = @import("c_api.zig");
const errors_mod = @import("errors.zig");

const OrtError = errors_mod.OrtError;

/// CoreML compute units for hardware selection
pub const CoreMLComputeUnits = enum {
    /// Enable all available compute units (CPU, GPU, Neural Engine)
    all,
    /// Restrict execution to CPU only
    cpu_only,
    /// Enable CPU and GPU acceleration
    cpu_and_gpu,
    /// Enable CPU and Neural Engine (recommended for precision)
    cpu_and_neural_engine,
};

/// CoreML model format
pub const CoreMLModelFormat = enum {
    /// Legacy NeuralNetwork format (Core ML 3+)
    neural_network,
    /// Modern MLProgram format (Core ML 5+, macOS 12+) - better precision
    ml_program,
};

/// CoreML execution provider options
pub const CoreMLOptions = struct {
    /// Model format - MLProgram has better precision
    model_format: CoreMLModelFormat = .ml_program,
    /// Which hardware units to use
    compute_units: CoreMLComputeUnits = .cpu_and_neural_engine,
    /// Allow low precision accumulation on GPU (may cause NaN issues)
    allow_low_precision_accumulation_on_gpu: bool = false,
    /// Require static input shapes (may improve performance)
    require_static_input_shapes: bool = false,
};

/// CUDA execution provider options
pub const CUDAOptions = struct {
    /// CUDA device ID (0 = first GPU)
    device_id: u32 = 0,
};

/// Generic ExecutionProvider factory for any c_api module
pub fn ExecutionProvider(comptime CApi: type) type {
    return union(enum) {
        /// CPU execution (default, works everywhere)
        cpu: void,
        /// CoreML execution (macOS only, uses Neural Engine + Metal GPU)
        coreml: CoreMLOptions,
        /// CUDA execution (NVIDIA GPU)
        cuda: CUDAOptions,
        /// Auto-detect best provider for platform
        auto: void,

        const Self = @This();

        /// Default CPU provider
        pub fn cpuProvider() Self {
            return .{ .cpu = {} };
        }

        /// CoreML provider with safe defaults (precision-focused)
        pub fn coremlProvider() Self {
            return .{ .coreml = .{} };
        }

        /// CoreML provider with custom options
        pub fn coremlWithOptions(opts: CoreMLOptions) Self {
            return .{ .coreml = opts };
        }

        /// CoreML with CPU only (for debugging precision issues)
        pub fn coremlSafe() Self {
            return .{ .coreml = .{ .compute_units = .cpu_only } };
        }

        /// CoreML with all compute units (maximum performance)
        pub fn coremlPerformance() Self {
            return .{ .coreml = .{ .compute_units = .all } };
        }

        /// CUDA provider with device ID
        pub fn cudaWithDevice(device_id: u32) Self {
            return .{ .cuda = .{ .device_id = device_id } };
        }

        /// CUDA provider with default device (0)
        pub fn cudaProvider() Self {
            return .{ .cuda = .{} };
        }

        /// Auto-detect best provider
        pub fn autoProvider() Self {
            return .{ .auto = {} };
        }

        /// Get the display name for this provider
        pub fn getName(self: Self) []const u8 {
            return switch (self) {
                .cpu => "CPU",
                .coreml => |opts| switch (opts.compute_units) {
                    .all => "CoreML:All",
                    .cpu_only => "CoreML:CPUOnly",
                    .cpu_and_gpu => "CoreML:CPU+GPU",
                    .cpu_and_neural_engine => "CoreML:CPU+ANE",
                },
                .cuda => "CUDA",
                .auto => "Auto",
            };
        }

        /// Apply this provider to session options
        pub fn apply(self: Self, session_options: *CApi.OrtSessionOptions) OrtError!void {
            const api = CApi.getApi() orelse return OrtError.ApiNotAvailable;
            const resolved = self.resolve();

            switch (resolved) {
                .cpu => {
                    // CPU is the default, nothing to configure
                },
                .coreml => |coreml_opts| {
                    // CoreML is only available on macOS
                    if (comptime builtin.os.tag != .macos) {
                        return OrtError.InvalidArgument;
                    }

                    // Build CoreML flags
                    var flags: u32 = 0;

                    // Model format
                    if (coreml_opts.model_format == .ml_program) {
                        flags |= CApi.CoreMLFlags.CREATE_MLPROGRAM;
                    }

                    // Compute units
                    switch (coreml_opts.compute_units) {
                        .cpu_only => flags |= CApi.CoreMLFlags.USE_CPU_ONLY,
                        .cpu_and_gpu => flags |= CApi.CoreMLFlags.USE_CPU_AND_GPU,
                        .cpu_and_neural_engine => {
                            // Default behavior uses CPU + ANE when available
                        },
                        .all => {
                            // Use everything available
                        },
                    }

                    // Static shapes can improve performance
                    if (coreml_opts.require_static_input_shapes) {
                        flags |= CApi.CoreMLFlags.ONLY_ALLOW_STATIC_INPUT_SHAPES;
                    }

                    const status = CApi.OrtSessionOptionsAppendExecutionProvider_CoreML(session_options, flags);
                    if (status) |s| {
                        api.ReleaseStatus.?(s);
                        return OrtError.EngineError;
                    }
                },
                .cuda => |cuda_opts| {
                    // CUDA provider requires the function to be available
                    // The function may be:
                    // 1. An extern fn (directly callable)
                    // 2. An optional fn pointer (null when CUDA not enabled)
                    // 3. Not declared at all
                    if (comptime @hasDecl(CApi, "OrtSessionOptionsAppendExecutionProvider_CUDA")) {
                        const cuda_decl = CApi.OrtSessionOptionsAppendExecutionProvider_CUDA;
                        const CudaDeclType = @TypeOf(cuda_decl);

                        // Check if it's an optional type (pointer that could be null)
                        if (comptime @typeInfo(CudaDeclType) == .optional) {
                            if (cuda_decl) |cuda_fn| {
                                const status = cuda_fn(
                                    session_options,
                                    @intCast(cuda_opts.device_id),
                                );
                                if (status) |s| {
                                    api.ReleaseStatus.?(s);
                                    return OrtError.EngineError;
                                }
                            } else {
                                // Function is null - CUDA not available
                                return OrtError.InvalidArgument;
                            }
                        } else {
                            // Direct extern function
                            const status = cuda_decl(
                                session_options,
                                @intCast(cuda_opts.device_id),
                            );
                            if (status) |s| {
                                api.ReleaseStatus.?(s);
                                return OrtError.EngineError;
                            }
                        }
                    } else {
                        // CUDA not available in this c_api
                        return OrtError.InvalidArgument;
                    }
                },
                .auto => unreachable, // resolve() handles auto
            }
        }

        /// Resolve auto provider to concrete provider
        pub fn resolve(self: Self) Self {
            if (self != .auto) return self;

            // Platform-specific auto-detection
            if (comptime builtin.os.tag == .macos) {
                // On macOS, use CoreML
                return Self.coremlProvider();
            }

            // Fall back to CPU
            return Self.cpuProvider();
        }
    };
}

// =============================================================================
// Backward-compatible exports using default c_api
// =============================================================================

/// Default ExecutionProvider type using built-in c_api (backward compatible)
pub const DefaultExecutionProvider = ExecutionProvider(default_c_api);

test "provider names" {
    const testing = std.testing;
    try testing.expectEqualStrings("CPU", DefaultExecutionProvider.cpuProvider().getName());
    try testing.expectEqualStrings("CoreML:CPU+ANE", DefaultExecutionProvider.coremlProvider().getName());
    try testing.expectEqualStrings("CoreML:CPUOnly", DefaultExecutionProvider.coremlSafe().getName());
    try testing.expectEqualStrings("CoreML:All", DefaultExecutionProvider.coremlPerformance().getName());
    try testing.expectEqualStrings("CUDA", DefaultExecutionProvider.cudaProvider().getName());
}

test "auto resolve" {
    const resolved = DefaultExecutionProvider.autoProvider().resolve();
    if (comptime builtin.os.tag == .macos) {
        try std.testing.expect(resolved == .coreml);
    } else {
        try std.testing.expect(resolved == .cpu);
    }
}

test "resolve non-auto returns self" {
    const cpu_provider = DefaultExecutionProvider.cpuProvider();
    try std.testing.expectEqual(cpu_provider, cpu_provider.resolve());

    const coreml_provider = DefaultExecutionProvider.coremlProvider();
    try std.testing.expectEqual(coreml_provider, coreml_provider.resolve());
}
