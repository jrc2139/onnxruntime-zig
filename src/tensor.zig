//! ONNX Runtime Tensor (Value) operations
//!
//! Tensors are the primary data type for inputs and outputs in ONNX models.
//! Supports generic c_api modules via the Tensor(CApi) factory.

const std = @import("std");
const default_c_api = @import("c_api.zig");
const errors_mod = @import("errors.zig");
const memory_info_mod = @import("memory_info.zig");

const OrtError = errors_mod.OrtError;

/// Generic Tensor factory for any c_api module
pub fn Tensor(comptime CApi: type) type {
    const Errs = errors_mod.Errors(CApi);
    const MemoryInfoType = memory_info_mod.MemoryInfo(CApi);

    return struct {
        ptr: *CApi.OrtValue,
        api: *const CApi.OrtApi,
        /// Whether this tensor owns the underlying memory
        owns_data: bool,

        const Self = @This();

        /// Create a tensor from a slice of data
        ///
        /// The data must remain valid for the lifetime of the tensor.
        pub fn fromSlice(
            comptime T: type,
            api: *const CApi.OrtApi,
            data: []T,
            shape: []const i64,
        ) OrtError!Self {
            const element_type = comptime zigTypeToOnnx(T);

            // Create memory info for CPU
            var memory_info: ?*CApi.OrtMemoryInfo = null;
            var status = api.CreateCpuMemoryInfo.?(
                CApi.OrtArenaAllocator,
                CApi.OrtMemTypeDefault,
                &memory_info,
            );
            try Errs.checkStatus(api, status);
            defer api.ReleaseMemoryInfo.?(memory_info.?);

            // Create the tensor
            var value: ?*CApi.OrtValue = null;
            status = api.CreateTensorWithDataAsOrtValue.?(
                memory_info.?,
                @ptrCast(data.ptr),
                data.len * @sizeOf(T),
                shape.ptr,
                shape.len,
                element_type.toC(),
                &value,
            );
            try Errs.checkStatus(api, status);

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
            api: *const CApi.OrtApi,
            data: []T,
            shape: []const i64,
            memory_info: MemoryInfoType,
        ) OrtError!Self {
            const element_type = comptime zigTypeToOnnx(T);

            var value: ?*CApi.OrtValue = null;
            const status = api.CreateTensorWithDataAsOrtValue.?(
                memory_info.getPtr(),
                @ptrCast(data.ptr),
                data.len * @sizeOf(T),
                shape.ptr,
                shape.len,
                element_type.toC(),
                &value,
            );
            try Errs.checkStatus(api, status);

            return Self{
                .ptr = value.?,
                .api = api,
                .owns_data = false,
            };
        }

        /// Create an empty tensor with allocated memory
        pub fn allocate(
            comptime T: type,
            api: *const CApi.OrtApi,
            allocator_ptr: *CApi.OrtAllocator,
            shape: []const i64,
        ) OrtError!Self {
            const element_type = comptime zigTypeToOnnx(T);

            var value: ?*CApi.OrtValue = null;
            const status = api.CreateTensorAsOrtValue.?(
                allocator_ptr,
                shape.ptr,
                shape.len,
                element_type.toC(),
                &value,
            );
            try Errs.checkStatus(api, status);

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
            try Errs.checkStatus(self.api, status);

            const count = try self.getElementCount();
            const typed_ptr: [*]T = @ptrCast(@alignCast(data_ptr.?));
            return typed_ptr[0..count];
        }

        /// Get the shape of the tensor
        pub fn getShape(self: Self, allocator: std.mem.Allocator) OrtError![]i64 {
            var type_info: ?*CApi.OrtTensorTypeAndShapeInfo = null;
            var status = self.api.GetTensorTypeAndShape.?(self.ptr, &type_info);
            try Errs.checkStatus(self.api, status);
            defer self.api.ReleaseTensorTypeAndShapeInfo.?(type_info.?);

            var dims_count: usize = 0;
            status = self.api.GetDimensionsCount.?(type_info.?, &dims_count);
            try Errs.checkStatus(self.api, status);

            const shape = allocator.alloc(i64, dims_count) catch return OrtError.Fail;
            status = self.api.GetDimensions.?(type_info.?, shape.ptr, dims_count);
            try Errs.checkStatus(self.api, status);

            return shape;
        }

        /// Get the total number of elements in the tensor
        pub fn getElementCount(self: Self) OrtError!usize {
            var type_info: ?*CApi.OrtTensorTypeAndShapeInfo = null;
            var status = self.api.GetTensorTypeAndShape.?(self.ptr, &type_info);
            try Errs.checkStatus(self.api, status);
            defer self.api.ReleaseTensorTypeAndShapeInfo.?(type_info.?);

            var count: usize = 0;
            status = self.api.GetTensorShapeElementCount.?(type_info.?, &count);
            try Errs.checkStatus(self.api, status);

            return count;
        }

        /// Get the element type of the tensor
        pub fn getElementType(self: Self) OrtError!default_c_api.TensorElementType {
            var type_info: ?*CApi.OrtTensorTypeAndShapeInfo = null;
            var status = self.api.GetTensorTypeAndShape.?(self.ptr, &type_info);
            try Errs.checkStatus(self.api, status);
            defer self.api.ReleaseTensorTypeAndShapeInfo.?(type_info.?);

            var element_type: CApi.ONNXTensorElementDataType = undefined;
            status = self.api.GetTensorElementType.?(type_info.?, &element_type);
            try Errs.checkStatus(self.api, status);

            return @enumFromInt(@intFromEnum(element_type));
        }

        /// Release the tensor
        pub fn deinit(self: *Self) void {
            self.api.ReleaseValue.?(self.ptr);
            self.ptr = undefined;
        }

        /// Get the underlying OrtValue pointer
        pub fn getPtr(self: Self) *CApi.OrtValue {
            return self.ptr;
        }
    };
}

/// Map Zig types to ONNX tensor element types
/// This is standalone and works with any c_api that has compatible TensorElementType
fn zigTypeToOnnx(comptime T: type) default_c_api.TensorElementType {
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

// =============================================================================
// Backward-compatible exports using default c_api
// =============================================================================

/// Default Tensor type using built-in c_api (backward compatible)
pub const DefaultTensor = Tensor(default_c_api);

test "zigTypeToOnnx returns correct types" {
    try std.testing.expectEqual(default_c_api.TensorElementType.float32, zigTypeToOnnx(f32));
    try std.testing.expectEqual(default_c_api.TensorElementType.int64, zigTypeToOnnx(i64));
    try std.testing.expectEqual(default_c_api.TensorElementType.uint8, zigTypeToOnnx(u8));
}
