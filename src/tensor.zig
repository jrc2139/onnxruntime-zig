//! ONNX Runtime Tensor (Value) operations
//!
//! Tensors are the primary data type for inputs and outputs in ONNX models.

const std = @import("std");
const c_api = @import("c_api.zig");
const errors = @import("errors.zig");
const MemoryInfo = @import("memory_info.zig").MemoryInfo;

const OrtError = errors.OrtError;

/// A tensor value that can be used as input or output
pub const Tensor = struct {
    ptr: *c_api.OrtValue,
    api: *const c_api.OrtApi,
    /// Whether this tensor owns the underlying memory
    owns_data: bool,

    const Self = @This();

    /// Create a tensor from a slice of data
    ///
    /// The data must remain valid for the lifetime of the tensor.
    pub fn fromSlice(
        comptime T: type,
        api: *const c_api.OrtApi,
        data: []T,
        shape: []const i64,
    ) OrtError!Self {
        const element_type = comptime zigTypeToOnnx(T);

        // Create memory info for CPU
        var memory_info: ?*c_api.OrtMemoryInfo = null;
        var status = api.CreateCpuMemoryInfo.?(
            c_api.c.OrtArenaAllocator,
            c_api.c.OrtMemTypeDefault,
            &memory_info,
        );
        try errors.checkStatus(api, status);
        defer api.ReleaseMemoryInfo.?(memory_info.?);

        // Create the tensor
        var value: ?*c_api.OrtValue = null;
        status = api.CreateTensorWithDataAsOrtValue.?(
            memory_info.?,
            @ptrCast(data.ptr),
            data.len * @sizeOf(T),
            shape.ptr,
            shape.len,
            element_type.toC(),
            &value,
        );
        try errors.checkStatus(api, status);

        return Self{
            .ptr = value.?,
            .api = api,
            .owns_data = false,
        };
    }

    /// Create a tensor from a slice using existing MemoryInfo
    ///
    /// This is more efficient when creating many tensors, as it reuses the MemoryInfo.
    /// The data and memory_info must remain valid for the lifetime of the tensor.
    pub fn fromSliceWithMemoryInfo(
        comptime T: type,
        api: *const c_api.OrtApi,
        data: []T,
        shape: []const i64,
        memory_info: MemoryInfo,
    ) OrtError!Self {
        const element_type = comptime zigTypeToOnnx(T);

        var value: ?*c_api.OrtValue = null;
        const status = api.CreateTensorWithDataAsOrtValue.?(
            memory_info.getPtr(),
            @ptrCast(data.ptr),
            data.len * @sizeOf(T),
            shape.ptr,
            shape.len,
            element_type.toC(),
            &value,
        );
        try errors.checkStatus(api, status);

        return Self{
            .ptr = value.?,
            .api = api,
            .owns_data = false,
        };
    }

    /// Create an empty tensor with allocated memory
    pub fn allocate(
        comptime T: type,
        api: *const c_api.OrtApi,
        allocator_ptr: *c_api.OrtAllocator,
        shape: []const i64,
    ) OrtError!Self {
        const element_type = comptime zigTypeToOnnx(T);

        var value: ?*c_api.OrtValue = null;
        const status = api.CreateTensorAsOrtValue.?(
            allocator_ptr,
            shape.ptr,
            shape.len,
            element_type.toC(),
            &value,
        );
        try errors.checkStatus(api, status);

        return Self{
            .ptr = value.?,
            .api = api,
            .owns_data = true,
        };
    }

    /// Get a pointer to the tensor's data
    pub fn getData(self: Self, comptime T: type) OrtError![]T {
        var data_ptr: ?*anyopaque = null;
        const status = self.api.GetTensorMutableData.?(self.ptr, &data_ptr);
        try errors.checkStatus(self.api, status);

        const count = try self.getElementCount();
        const typed_ptr: [*]T = @ptrCast(@alignCast(data_ptr.?));
        return typed_ptr[0..count];
    }

    /// Get the shape of the tensor
    pub fn getShape(self: Self, allocator: std.mem.Allocator) OrtError![]i64 {
        var type_info: ?*c_api.OrtTensorTypeAndShapeInfo = null;
        var status = self.api.GetTensorTypeAndShape.?(self.ptr, &type_info);
        try errors.checkStatus(self.api, status);
        defer self.api.ReleaseTensorTypeAndShapeInfo.?(type_info.?);

        var dims_count: usize = 0;
        status = self.api.GetDimensionsCount.?(type_info.?, &dims_count);
        try errors.checkStatus(self.api, status);

        const shape = allocator.alloc(i64, dims_count) catch return OrtError.Fail;
        status = self.api.GetDimensions.?(type_info.?, shape.ptr, dims_count);
        try errors.checkStatus(self.api, status);

        return shape;
    }

    /// Get the total number of elements in the tensor
    pub fn getElementCount(self: Self) OrtError!usize {
        var type_info: ?*c_api.OrtTensorTypeAndShapeInfo = null;
        var status = self.api.GetTensorTypeAndShape.?(self.ptr, &type_info);
        try errors.checkStatus(self.api, status);
        defer self.api.ReleaseTensorTypeAndShapeInfo.?(type_info.?);

        var count: usize = 0;
        status = self.api.GetTensorShapeElementCount.?(type_info.?, &count);
        try errors.checkStatus(self.api, status);

        return count;
    }

    /// Get the element type of the tensor
    pub fn getElementType(self: Self) OrtError!c_api.TensorElementType {
        var type_info: ?*c_api.OrtTensorTypeAndShapeInfo = null;
        var status = self.api.GetTensorTypeAndShape.?(self.ptr, &type_info);
        try errors.checkStatus(self.api, status);
        defer self.api.ReleaseTensorTypeAndShapeInfo.?(type_info.?);

        var element_type: c_api.ONNXTensorElementDataType = undefined;
        status = self.api.GetTensorElementType.?(type_info.?, &element_type);
        try errors.checkStatus(self.api, status);

        return @enumFromInt(@intFromEnum(element_type));
    }

    /// Release the tensor
    pub fn deinit(self: *Self) void {
        self.api.ReleaseValue.?(self.ptr);
        self.ptr = undefined;
    }

    /// Get the underlying OrtValue pointer
    pub fn getPtr(self: Self) *c_api.OrtValue {
        return self.ptr;
    }
};

/// Map Zig types to ONNX tensor element types
fn zigTypeToOnnx(comptime T: type) c_api.TensorElementType {
    return switch (T) {
        f32 => .float32,
        f64 => .float64,
        i8 => .int8,
        i16 => .int16,
        i32 => .int32,
        i64 => .int64,
        u8 => .uint8,
        u16 => .uint16,
        u32 => .uint32,
        u64 => .uint64,
        bool => .bool_,
        else => @compileError("Unsupported tensor element type: " ++ @typeName(T)),
    };
}

test "zigTypeToOnnx returns correct types" {
    try std.testing.expectEqual(c_api.TensorElementType.float32, zigTypeToOnnx(f32));
    try std.testing.expectEqual(c_api.TensorElementType.int64, zigTypeToOnnx(i64));
    try std.testing.expectEqual(c_api.TensorElementType.uint8, zigTypeToOnnx(u8));
}
