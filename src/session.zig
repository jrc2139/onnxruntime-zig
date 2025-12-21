//! ONNX Runtime Session management
//!
//! Sessions load and run ONNX models. Each session holds a loaded model
//! and can perform inference with it.
//! Supports generic c_api modules via the Session(CApi) factory.

const std = @import("std");
const default_c_api = @import("c_api.zig");
const errors_mod = @import("errors.zig");
const env_mod = @import("env.zig");
const tensor_mod = @import("tensor.zig");
const run_options_mod = @import("run_options.zig");

const OrtError = errors_mod.OrtError;

/// Graph optimization levels
pub const GraphOptLevel = enum(c_uint) {
    disable = 0,
    basic = 1,
    extended = 2,
    layout = 3,
    all = 99,

    pub fn toC(self: GraphOptLevel) c_uint {
        return @intFromEnum(self);
    }
};

/// Generic SessionOptions factory for any c_api module
pub fn SessionOptions(comptime CApi: type) type {
    const Errs = errors_mod.Errors(CApi);

    return struct {
        ptr: *CApi.OrtSessionOptions,
        api: *const CApi.OrtApi,

        const Self = @This();

        /// Create new session options with default settings
        pub fn init(api: *const CApi.OrtApi) OrtError!Self {
            var options: ?*CApi.OrtSessionOptions = null;
            const status = api.CreateSessionOptions.?(&options);
            try Errs.checkStatus(api, status);

            return Self{
                .ptr = options.?,
                .api = api,
            };
        }

        /// Release the session options
        pub fn deinit(self: *Self) void {
            self.api.ReleaseSessionOptions.?(self.ptr);
            self.ptr = undefined;
        }

        /// Set the number of intra-op threads (parallelism within operators)
        pub fn setIntraOpNumThreads(self: Self, num_threads: i32) OrtError!void {
            const status = self.api.SetIntraOpNumThreads.?(self.ptr, @intCast(num_threads));
            try Errs.checkStatus(self.api, status);
        }

        /// Set the number of inter-op threads (parallelism between operators)
        pub fn setInterOpNumThreads(self: Self, num_threads: i32) OrtError!void {
            const status = self.api.SetInterOpNumThreads.?(self.ptr, @intCast(num_threads));
            try Errs.checkStatus(self.api, status);
        }

        /// Set graph optimization level
        pub fn setGraphOptimizationLevel(self: Self, level: GraphOptLevel) OrtError!void {
            const status = self.api.SetSessionGraphOptimizationLevel.?(self.ptr, level.toC());
            try Errs.checkStatus(self.api, status);
        }

        /// Enable/disable memory pattern optimization
        pub fn enableMemPattern(self: Self, enable: bool) OrtError!void {
            if (enable) {
                const status = self.api.EnableMemPattern.?(self.ptr);
                try Errs.checkStatus(self.api, status);
            } else {
                const status = self.api.DisableMemPattern.?(self.ptr);
                try Errs.checkStatus(self.api, status);
            }
        }

        /// Get the underlying pointer
        pub fn getPtr(self: Self) *CApi.OrtSessionOptions {
            return self.ptr;
        }
    };
}

/// Generic Session factory for any c_api module
pub fn Session(comptime CApi: type) type {
    const Errs = errors_mod.Errors(CApi);
    const EnvironmentType = env_mod.Environment(CApi);
    const TensorType = tensor_mod.Tensor(CApi);
    const SessionOptionsType = SessionOptions(CApi);
    const RunOptionsType = run_options_mod.RunOptions(CApi);

    return struct {
        ptr: *CApi.OrtSession,
        api: *const CApi.OrtApi,
        allocator_ptr: *CApi.OrtAllocator,
        input_names: [][:0]const u8,
        output_names: [][:0]const u8,
        zig_allocator: std.mem.Allocator,

        const Self = @This();

        /// Load a model from a file path
        pub fn init(
            env: EnvironmentType,
            model_path: [:0]const u8,
            options: ?SessionOptionsType,
            allocator: std.mem.Allocator,
        ) OrtError!Self {
            const api = env.getApi();

            // Create default options if none provided
            var owned_options: ?SessionOptionsType = null;
            const opts = if (options) |o| o else blk: {
                owned_options = try SessionOptionsType.init(api);
                break :blk owned_options.?;
            };
            defer if (owned_options) |*o| o.deinit();

            // Create session
            var session: ?*CApi.OrtSession = null;
            const status = api.CreateSession.?(
                env.getPtr(),
                model_path.ptr,
                opts.getPtr(),
                &session,
            );
            try Errs.checkStatus(api, status);

            // Get allocator
            var ort_allocator: ?*CApi.OrtAllocator = null;
            const alloc_status = api.GetAllocatorWithDefaultOptions.?(&ort_allocator);
            try Errs.checkStatus(api, alloc_status);

            // Get input/output names
            const input_names = try getInputNamesGeneric(CApi, api, session.?, ort_allocator.?, allocator);
            errdefer freeNames(allocator, input_names);

            const output_names = try getOutputNamesGeneric(CApi, api, session.?, ort_allocator.?, allocator);
            errdefer freeNames(allocator, output_names);

            return Self{
                .ptr = session.?,
                .api = api,
                .allocator_ptr = ort_allocator.?,
                .input_names = input_names,
                .output_names = output_names,
                .zig_allocator = allocator,
            };
        }

        /// Release the session
        pub fn deinit(self: *Self) void {
            freeNames(self.zig_allocator, self.input_names);
            freeNames(self.zig_allocator, self.output_names);
            self.api.ReleaseSession.?(self.ptr);
            self.ptr = undefined;
        }

        /// Run inference with the given inputs
        ///
        /// Returns an array of output tensors. Caller is responsible for calling
        /// deinit() on each output tensor.
        pub fn run(self: Self, inputs: []const TensorType) OrtError![]TensorType {
            if (inputs.len != self.input_names.len) {
                return OrtError.InvalidArgument;
            }

            // Prepare input pointers
            const input_ptrs = self.zig_allocator.alloc(*CApi.OrtValue, inputs.len) catch return OrtError.Fail;
            defer self.zig_allocator.free(input_ptrs);

            for (inputs, 0..) |input, i| {
                input_ptrs[i] = input.getPtr();
            }

            // Prepare input names as C strings
            const input_name_ptrs = self.zig_allocator.alloc([*c]const u8, self.input_names.len) catch return OrtError.Fail;
            defer self.zig_allocator.free(input_name_ptrs);

            for (self.input_names, 0..) |name, i| {
                input_name_ptrs[i] = name.ptr;
            }

            // Prepare output names as C strings
            const output_name_ptrs = self.zig_allocator.alloc([*c]const u8, self.output_names.len) catch return OrtError.Fail;
            defer self.zig_allocator.free(output_name_ptrs);

            for (self.output_names, 0..) |name, i| {
                output_name_ptrs[i] = name.ptr;
            }

            // Prepare output array
            const output_ptrs = self.zig_allocator.alloc(?*CApi.OrtValue, self.output_names.len) catch return OrtError.Fail;
            defer self.zig_allocator.free(output_ptrs);

            @memset(output_ptrs, null);

            // Run inference
            const status = self.api.Run.?(
                self.ptr,
                null, // run options
                input_name_ptrs.ptr,
                @ptrCast(input_ptrs.ptr),
                inputs.len,
                output_name_ptrs.ptr,
                self.output_names.len,
                @ptrCast(output_ptrs.ptr),
            );
            try Errs.checkStatus(self.api, status);

            // Wrap outputs in Tensor structs
            const outputs = self.zig_allocator.alloc(TensorType, self.output_names.len) catch return OrtError.Fail;
            errdefer self.zig_allocator.free(outputs);

            for (output_ptrs, 0..) |ptr, i| {
                outputs[i] = TensorType{
                    .ptr = ptr.?,
                    .api = self.api,
                    .owns_data = true,
                };
            }

            return outputs;
        }

        /// Callback type for async inference
        pub const AsyncCallback = *const fn (
            user_data: ?*anyopaque,
            result: AsyncResult,
        ) void;

        /// Result of async inference
        pub const AsyncResult = union(enum) {
            /// Successful inference with output tensors
            success: []TensorType,
            /// Inference failed with error
            err: OrtError,
        };

        /// Context passed through C callback
        const AsyncContext = struct {
            callback: AsyncCallback,
            user_data: ?*anyopaque,
            api: *const CApi.OrtApi,
            allocator: std.mem.Allocator,
            num_outputs: usize,
            // Arrays that must live until callback completes
            input_ptrs: []*CApi.OrtValue,
            input_name_ptrs: [][*c]const u8,
            output_name_ptrs: [][*c]const u8,
            output_ptrs: []?*CApi.OrtValue,
        };

        /// C-compatible trampoline that converts to Zig callback
        fn asyncTrampoline(
            ctx_ptr: ?*anyopaque,
            outputs: [*c]?*CApi.OrtValue,
            num_outputs: usize,
            status: ?*CApi.OrtStatus,
        ) callconv(.c) void {
            const ctx: *AsyncContext = @ptrCast(@alignCast(ctx_ptr));
            defer {
                // Free all allocated arrays
                ctx.allocator.free(ctx.input_ptrs);
                ctx.allocator.free(ctx.input_name_ptrs);
                ctx.allocator.free(ctx.output_name_ptrs);
                ctx.allocator.free(ctx.output_ptrs);
                ctx.allocator.destroy(ctx);
            }

            // Check for errors
            if (status) |s| {
                const code = ctx.api.GetErrorCode.?(s);
                ctx.api.ReleaseStatus.?(s);

                const ort_err: OrtError = switch (code) {
                    CApi.ORT_FAIL => OrtError.Fail,
                    CApi.ORT_INVALID_ARGUMENT => OrtError.InvalidArgument,
                    CApi.ORT_NO_SUCHFILE => OrtError.NoSuchFile,
                    CApi.ORT_NO_MODEL => OrtError.NoModel,
                    CApi.ORT_ENGINE_ERROR => OrtError.EngineError,
                    CApi.ORT_RUNTIME_EXCEPTION => OrtError.RuntimeException,
                    CApi.ORT_INVALID_PROTOBUF => OrtError.InvalidProtobuf,
                    CApi.ORT_MODEL_LOADED => OrtError.ModelLoaded,
                    CApi.ORT_NOT_IMPLEMENTED => OrtError.NotImplemented,
                    CApi.ORT_INVALID_GRAPH => OrtError.InvalidGraph,
                    CApi.ORT_EP_FAIL => OrtError.ExecutionProviderFail,
                    else => OrtError.Unknown,
                };
                ctx.callback(ctx.user_data, .{ .err = ort_err });
                return;
            }

            // Wrap outputs in Tensor structs
            const tensor_outputs = ctx.allocator.alloc(TensorType, num_outputs) catch {
                ctx.callback(ctx.user_data, .{ .err = OrtError.Fail });
                return;
            };

            for (0..num_outputs) |i| {
                tensor_outputs[i] = TensorType{
                    .ptr = outputs[i].?,
                    .api = ctx.api,
                    .owns_data = true,
                };
            }

            ctx.callback(ctx.user_data, .{ .success = tensor_outputs });
        }

        /// Run inference asynchronously
        ///
        /// The callback will be invoked when inference completes (or fails).
        /// On success, caller is responsible for freeing outputs (deinit each, then free slice).
        pub fn runAsync(
            self: Self,
            inputs: []const TensorType,
            run_options: ?*RunOptionsType,
            callback: AsyncCallback,
            user_data: ?*anyopaque,
        ) OrtError!void {
            if (inputs.len != self.input_names.len) {
                return OrtError.InvalidArgument;
            }

            // Allocate context (freed in trampoline)
            const ctx = self.zig_allocator.create(AsyncContext) catch return OrtError.Fail;
            errdefer self.zig_allocator.destroy(ctx);

            // Allocate arrays that must persist until callback
            const input_ptrs = self.zig_allocator.alloc(*CApi.OrtValue, inputs.len) catch return OrtError.Fail;
            errdefer self.zig_allocator.free(input_ptrs);

            const input_name_ptrs = self.zig_allocator.alloc([*c]const u8, self.input_names.len) catch return OrtError.Fail;
            errdefer self.zig_allocator.free(input_name_ptrs);

            const output_name_ptrs = self.zig_allocator.alloc([*c]const u8, self.output_names.len) catch return OrtError.Fail;
            errdefer self.zig_allocator.free(output_name_ptrs);

            const output_ptrs = self.zig_allocator.alloc(?*CApi.OrtValue, self.output_names.len) catch return OrtError.Fail;
            errdefer self.zig_allocator.free(output_ptrs);

            // Fill arrays
            for (inputs, 0..) |input, i| {
                input_ptrs[i] = input.getPtr();
            }
            for (self.input_names, 0..) |name, i| {
                input_name_ptrs[i] = name.ptr;
            }
            for (self.output_names, 0..) |name, i| {
                output_name_ptrs[i] = name.ptr;
            }
            @memset(output_ptrs, null);

            // Fill context
            ctx.* = .{
                .callback = callback,
                .user_data = user_data,
                .api = self.api,
                .allocator = self.zig_allocator,
                .num_outputs = self.output_names.len,
                .input_ptrs = input_ptrs,
                .input_name_ptrs = input_name_ptrs,
                .output_name_ptrs = output_name_ptrs,
                .output_ptrs = output_ptrs,
            };

            const opts_ptr: ?*CApi.OrtRunOptions = if (run_options) |ro| ro.ptr else null;

            // Call RunAsync
            const status = self.api.RunAsync.?(
                self.ptr,
                opts_ptr,
                input_name_ptrs.ptr,
                @ptrCast(input_ptrs.ptr),
                inputs.len,
                output_name_ptrs.ptr,
                self.output_names.len,
                @ptrCast(output_ptrs.ptr),
                asyncTrampoline,
                ctx,
            );
            try Errs.checkStatus(self.api, status);
        }

        /// Get the number of inputs
        pub fn getInputCount(self: Self) usize {
            return self.input_names.len;
        }

        /// Get the number of outputs
        pub fn getOutputCount(self: Self) usize {
            return self.output_names.len;
        }

        /// Get the ORT allocator
        pub fn getAllocator(self: Self) *CApi.OrtAllocator {
            return self.allocator_ptr;
        }

        /// Get the underlying session pointer
        pub fn getPtr(self: Self) *CApi.OrtSession {
            return self.ptr;
        }

        fn freeNames(allocator: std.mem.Allocator, names: [][:0]const u8) void {
            for (names) |name| {
                allocator.free(name);
            }
            allocator.free(names);
        }
    };
}

/// Get input names (generic version)
fn getInputNamesGeneric(
    comptime CApi: type,
    api: *const CApi.OrtApi,
    session: *CApi.OrtSession,
    ort_allocator: *CApi.OrtAllocator,
    allocator: std.mem.Allocator,
) OrtError![][:0]const u8 {
    const Errs = errors_mod.Errors(CApi);

    var count: usize = 0;
    var status = api.SessionGetInputCount.?(session, &count);
    try Errs.checkStatus(api, status);

    const names = allocator.alloc([:0]const u8, count) catch return OrtError.Fail;
    errdefer allocator.free(names);

    for (0..count) |i| {
        var name_ptr: [*c]u8 = undefined;
        status = api.SessionGetInputName.?(session, i, ort_allocator, &name_ptr);
        try Errs.checkStatus(api, status);

        // Copy the name to our allocator and free the ORT copy
        const name_len = std.mem.len(name_ptr);
        const name_copy = allocator.allocSentinel(u8, name_len, 0) catch {
            _ = api.AllocatorFree.?(ort_allocator, name_ptr);
            return OrtError.Fail;
        };
        @memcpy(name_copy, name_ptr[0..name_len]);

        _ = api.AllocatorFree.?(ort_allocator, name_ptr);
        names[i] = name_copy;
    }

    return names;
}

/// Get output names (generic version)
fn getOutputNamesGeneric(
    comptime CApi: type,
    api: *const CApi.OrtApi,
    session: *CApi.OrtSession,
    ort_allocator: *CApi.OrtAllocator,
    allocator: std.mem.Allocator,
) OrtError![][:0]const u8 {
    const Errs = errors_mod.Errors(CApi);

    var count: usize = 0;
    var status = api.SessionGetOutputCount.?(session, &count);
    try Errs.checkStatus(api, status);

    const names = allocator.alloc([:0]const u8, count) catch return OrtError.Fail;
    errdefer allocator.free(names);

    for (0..count) |i| {
        var name_ptr: [*c]u8 = undefined;
        status = api.SessionGetOutputName.?(session, i, ort_allocator, &name_ptr);
        try Errs.checkStatus(api, status);

        // Copy the name to our allocator and free the ORT copy
        const name_len = std.mem.len(name_ptr);
        const name_copy = allocator.allocSentinel(u8, name_len, 0) catch {
            _ = api.AllocatorFree.?(ort_allocator, name_ptr);
            return OrtError.Fail;
        };
        @memcpy(name_copy, name_ptr[0..name_len]);

        _ = api.AllocatorFree.?(ort_allocator, name_ptr);
        names[i] = name_copy;
    }

    return names;
}

// =============================================================================
// Backward-compatible exports using default c_api
// =============================================================================

/// Default SessionOptions type using built-in c_api (backward compatible)
pub const DefaultSessionOptions = SessionOptions(default_c_api);

/// Default Session type using built-in c_api (backward compatible)
pub const DefaultSession = Session(default_c_api);

test "SessionOptions creation" {
    const api = default_c_api.getApi() orelse return error.SkipZigTest;
    var opts = try DefaultSessionOptions.init(api);
    defer opts.deinit();

    try opts.setIntraOpNumThreads(4);
    try opts.setGraphOptimizationLevel(.all);
}
