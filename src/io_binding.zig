//! ONNX Runtime IoBinding for Zero-Allocation Inference
//!
//! IoBinding pre-binds tensors to input/output names, eliminating
//! all per-call allocations on the hot path. Use this for real-time
//! inference where every allocation counts.
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
const c_api = @import("c_api.zig");
const errors = @import("errors.zig");
const session_mod = @import("session.zig");
const tensor_mod = @import("tensor.zig");
const memory_info_mod = @import("memory_info.zig");
const run_options_mod = @import("run_options.zig");

const OrtError = errors.OrtError;
const Session = session_mod.Session;
const Tensor = tensor_mod.Tensor;
const MemoryInfo = memory_info_mod.MemoryInfo;
const RunOptions = run_options_mod.RunOptions;

/// IoBinding for zero-allocation inference
pub const IoBinding = struct {
    ptr: *c_api.OrtIoBinding,
    api: *const c_api.OrtApi,
    session_ptr: *c_api.OrtSession,

    const Self = @This();

    /// Create an IoBinding for the given session
    pub fn init(session: Session) OrtError!Self {
        const api = session.api;
        var binding: ?*c_api.OrtIoBinding = null;
        const status = api.CreateIoBinding.?(session.ptr, &binding);
        try errors.checkStatus(api, status);

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
    pub fn bindInput(self: Self, name: [:0]const u8, tensor: Tensor) OrtError!void {
        const status = self.api.BindInput.?(self.ptr, name.ptr, tensor.ptr);
        try errors.checkStatus(self.api, status);
    }

    /// Bind a pre-allocated output tensor to a name
    ///
    /// The tensor must remain valid for the lifetime of the binding
    /// or until clearOutputs() is called. ORT writes results directly
    /// into this tensor's memory - zero copy.
    pub fn bindOutput(self: Self, name: [:0]const u8, tensor: Tensor) OrtError!void {
        const status = self.api.BindOutput.?(self.ptr, name.ptr, tensor.ptr);
        try errors.checkStatus(self.api, status);
    }

    /// Bind output to a device, letting ORT allocate the output
    ///
    /// Use this when output shape is dynamic or unknown at bind time.
    /// After run(), use getOutputs() to retrieve the allocated tensors.
    pub fn bindOutputToDevice(self: Self, name: [:0]const u8, mem_info: MemoryInfo) OrtError!void {
        const status = self.api.BindOutputToDevice.?(self.ptr, name.ptr, mem_info.ptr);
        try errors.checkStatus(self.api, status);
    }

    /// Run inference with bound inputs/outputs - zero allocations
    pub fn run(self: Self, run_options: ?*RunOptions) OrtError!void {
        const opts_ptr: ?*c_api.OrtRunOptions = if (run_options) |ro| ro.ptr else null;
        const status = self.api.RunWithBinding.?(self.session_ptr, opts_ptr, self.ptr);
        try errors.checkStatus(self.api, status);
    }

    /// Get outputs after run (for bindOutputToDevice case)
    ///
    /// Only needed when using bindOutputToDevice. The returned tensors
    /// are owned by the binding and become invalid after clearOutputs()
    /// or deinit().
    pub fn getOutputs(self: Self, allocator: std.mem.Allocator) OrtError![]Tensor {
        var output_count: usize = 0;
        var output_values: [*c]?*c_api.OrtValue = undefined;

        // Get bound output values
        var ort_allocator: ?*c_api.OrtAllocator = null;
        const alloc_status = self.api.GetAllocatorWithDefaultOptions.?(&ort_allocator);
        try errors.checkStatus(self.api, alloc_status);

        const status = self.api.GetBoundOutputValues.?(
            self.ptr,
            ort_allocator.?,
            &output_values,
            &output_count,
        );
        try errors.checkStatus(self.api, status);
        defer _ = self.api.AllocatorFree.?(ort_allocator.?, output_values);

        // Wrap in Tensor structs
        const outputs = allocator.alloc(Tensor, output_count) catch return OrtError.Fail;
        errdefer allocator.free(outputs);

        for (0..output_count) |i| {
            outputs[i] = Tensor{
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
    pub fn getPtr(self: Self) *c_api.OrtIoBinding {
        return self.ptr;
    }
};

test "IoBinding creation" {
    const api = c_api.getApi() orelse return error.SkipZigTest;

    // We can't test much without a real model, but we can verify the API compiles
    _ = api;
}
