//! ONNX Runtime IoBinding for Zero-Allocation Inference
//!
//! IoBinding pre-binds tensors to input/output names, eliminating
//! all per-call allocations on the hot path. Use this for real-time
//! inference where every allocation counts.
//! Supports generic c_api modules via the IoBinding(CApi) factory.
//!
//! ## Example
//!
//! ```zig
//! var binding = try IoBinding.init(session);
//! defer binding.deinit();
//!
//! // Bind once
//! try binding.bindInput("x", input_tensor);
//! try binding.bindOutput("y", output_tensor);
//!
//! // Run many times - zero allocations
//! while (running) {
//!     try binding.run(null);
//! }
//! ```

const std = @import("std");
const default_c_api = @import("c_api.zig");
const errors_mod = @import("errors.zig");
const session_mod = @import("session.zig");
const tensor_mod = @import("tensor.zig");
const memory_info_mod = @import("memory_info.zig");
const run_options_mod = @import("run_options.zig");

const OrtError = errors_mod.OrtError;

/// Generic IoBinding factory for any c_api module
pub fn IoBinding(comptime CApi: type) type {
    const Errs = errors_mod.Errors(CApi);
    const SessionType = session_mod.Session(CApi);
    const TensorType = tensor_mod.Tensor(CApi);
    const MemoryInfoType = memory_info_mod.MemoryInfo(CApi);
    const RunOptionsType = run_options_mod.RunOptions(CApi);

    return struct {
        ptr: *CApi.OrtIoBinding,
        api: *const CApi.OrtApi,
        session_ptr: *CApi.OrtSession,

        const Self = @This();

        /// Create an IoBinding for the given session
        pub fn init(session: SessionType) OrtError!Self {
            const api = session.api;
            var binding: ?*CApi.OrtIoBinding = null;
            const status = api.CreateIoBinding.?(session.ptr, &binding);
            try Errs.checkStatus(api, status);

            return Self{
                .ptr = binding.?,
                .api = api,
                .session_ptr = session.ptr,
            };
        }

        /// Release the binding
        pub fn deinit(self: *Self) void {
            self.api.ReleaseIoBinding.?(self.ptr);
            self.ptr = undefined;
        }

        /// Bind an input tensor to a name
        ///
        /// The tensor must remain valid for the lifetime of the binding
        /// or until clearInputs() is called.
        pub fn bindInput(self: Self, name: [:0]const u8, tensor: TensorType) OrtError!void {
            const status = self.api.BindInput.?(self.ptr, name.ptr, tensor.ptr);
            try Errs.checkStatus(self.api, status);
        }

        /// Bind a pre-allocated output tensor to a name
        ///
        /// The tensor must remain valid for the lifetime of the binding
        /// or until clearOutputs() is called. ORT writes results directly
        /// into this tensor's memory - zero copy.
        pub fn bindOutput(self: Self, name: [:0]const u8, tensor: TensorType) OrtError!void {
            const status = self.api.BindOutput.?(self.ptr, name.ptr, tensor.ptr);
            try Errs.checkStatus(self.api, status);
        }

        /// Bind output to a device, letting ORT allocate the output
        ///
        /// Use this when output shape is dynamic or unknown at bind time.
        /// After run(), use getOutputs() to retrieve the allocated tensors.
        pub fn bindOutputToDevice(self: Self, name: [:0]const u8, mem_info: MemoryInfoType) OrtError!void {
            const status = self.api.BindOutputToDevice.?(self.ptr, name.ptr, mem_info.ptr);
            try Errs.checkStatus(self.api, status);
        }

        /// Run inference with bound inputs/outputs - zero allocations
        pub fn run(self: Self, run_options: ?*RunOptionsType) OrtError!void {
            const opts_ptr: ?*CApi.OrtRunOptions = if (run_options) |ro| ro.ptr else null;
            const status = self.api.RunWithBinding.?(self.session_ptr, opts_ptr, self.ptr);
            try Errs.checkStatus(self.api, status);
        }

        /// Get outputs after run (for bindOutputToDevice case)
        ///
        /// Only needed when using bindOutputToDevice. The returned tensors
        /// are owned by the binding and become invalid after clearOutputs()
        /// or deinit().
        pub fn getOutputs(self: Self, allocator: std.mem.Allocator) OrtError![]TensorType {
            var output_count: usize = 0;
            var output_values: [*c]?*CApi.OrtValue = undefined;

            // Get bound output values
            var ort_allocator: ?*CApi.OrtAllocator = null;
            const alloc_status = self.api.GetAllocatorWithDefaultOptions.?(&ort_allocator);
            try Errs.checkStatus(self.api, alloc_status);

            const status = self.api.GetBoundOutputValues.?(
                self.ptr,
                ort_allocator.?,
                &output_values,
                &output_count,
            );
            try Errs.checkStatus(self.api, status);
            defer _ = self.api.AllocatorFree.?(ort_allocator.?, output_values);

            // Wrap in Tensor structs
            const outputs = allocator.alloc(TensorType, output_count) catch return OrtError.Fail;
            errdefer allocator.free(outputs);

            for (0..output_count) |i| {
                outputs[i] = TensorType{
                    .ptr = output_values[i].?,
                    .api = self.api,
                    .owns_data = false, // Owned by binding
                };
            }

            return outputs;
        }

        /// Clear all input bindings
        pub fn clearInputs(self: Self) void {
            self.api.ClearBoundInputs.?(self.ptr);
        }

        /// Clear all output bindings
        pub fn clearOutputs(self: Self) void {
            self.api.ClearBoundOutputs.?(self.ptr);
        }

        /// Get the underlying pointer
        pub fn getPtr(self: Self) *CApi.OrtIoBinding {
            return self.ptr;
        }
    };
}

// =============================================================================
// Backward-compatible exports using default c_api
// =============================================================================

/// Default IoBinding type using built-in c_api (backward compatible)
pub const DefaultIoBinding = IoBinding(default_c_api);

test "IoBinding creation" {
    const api = default_c_api.getApi() orelse return error.SkipZigTest;

    // We can't test much without a real model, but we can verify the API compiles
    _ = api;
}
