//! The MLX array accessors: scalar constructors and setters, per-dtype
//! element and storage readers, contiguity queries, and managed buffers.

use crate::error::check_status;
use crate::raw;
use crate::{MlxArray, MlxCError, MlxDtype};

impl MlxArray {
    /// Creates a boolean scalar array.
    pub fn from_bool(value: bool) -> Self {
        // SAFETY: The scalar constructor takes a plain value and the returned
        // handle enters RAII ownership.
        let raw_array = unsafe { raw::mlx_array_new_bool(value) };
        Self::own(raw_array)
    }

    /// Creates an integer scalar array.
    pub fn from_scalar_i32(value: i32) -> Self {
        // SAFETY: The scalar constructor takes a plain value and the returned
        // handle enters RAII ownership.
        let raw_array = unsafe { raw::mlx_array_new_int(value) };
        Self::own(raw_array)
    }

    /// Creates a float32 scalar array.
    pub fn from_scalar_f32(value: f32) -> Self {
        // SAFETY: The scalar constructor takes a plain value and the returned
        // handle enters RAII ownership.
        let raw_array = unsafe { raw::mlx_array_new_float32(value) };
        Self::own(raw_array)
    }

    /// Creates a float64 scalar array.
    pub fn from_scalar_f64(value: f64) -> Self {
        // SAFETY: The scalar constructor takes a plain value and the returned
        // handle enters RAII ownership.
        let raw_array = unsafe { raw::mlx_array_new_float64(value) };
        Self::own(raw_array)
    }

    /// Creates a complex scalar array from real and imaginary parts.
    pub fn from_scalar_complex(real: f32, imag: f32) -> Self {
        // SAFETY: The scalar constructor takes plain values and the returned
        // handle enters RAII ownership.
        let raw_array = unsafe { raw::mlx_array_new_complex(real, imag) };
        Self::own(raw_array)
    }

    fn own(raw_array: raw::mlx_array) -> Self {
        Self { raw_array }
    }

    /// Creates an array that borrows host memory without copying.
    ///
    /// # Safety
    /// The buffer must stay valid and unmodified until the destructor runs,
    /// its contents must match `dtype`'s element layout, and its element
    /// count must match the shape product.
    pub unsafe fn from_managed_buffer(
        buffer: Vec<u8>,
        shape: &[i32],
        dtype: MlxDtype,
    ) -> Result<Self, MlxCError> {
        let operation: &'static str = "create an MLX array from a managed buffer";
        let element_count = shape.iter().try_fold(1_usize, |product, dimension| {
            product.checked_mul(usize::try_from(*dimension).ok()?)
        });
        if element_count != Some(buffer.len() / Self::byte_size_of_dtype(dtype)) {
            return Err(MlxCError {
                operation,
                description: "buffer byte count does not match the shape and dtype".to_owned(),
            });
        }
        let managed = Box::new(ManagedBuffer { storage: buffer });
        let data_pointer = managed.storage.as_ptr();
        let payload = Box::into_raw(managed);
        let rank = i32::try_from(shape.len()).map_err(|_| MlxCError {
            operation,
            description: "array rank exceeds the C API integer range".to_owned(),
        })?;
        // SAFETY: The boxed buffer stays alive until the payload destructor
        // runs and the destructor reclaims exactly this allocation; the shape
        // slice remains valid for the copying constructor.
        let raw_array = unsafe {
            raw::mlx_array_new_data_managed_payload(
                data_pointer.cast_mut().cast(),
                shape.as_ptr(),
                rank,
                dtype.to_raw(),
                payload.cast(),
                Some(drop_managed_payload),
            )
        };
        let array = Self::own(raw_array);
        array.require_populated(operation)?;
        Ok(array)
    }

    /// Replaces this handle's value with a boolean scalar.
    pub fn set_bool(&mut self, value: bool) -> Result<(), MlxCError> {
        // SAFETY: The handle is live and the value is plain.
        let status = unsafe { raw::mlx_array_set_bool(self.raw_mut(), value) };
        check_status(status, "replace an MLX array value")
    }

    /// Replaces this handle's value with an integer scalar.
    pub fn set_i32(&mut self, value: i32) -> Result<(), MlxCError> {
        // SAFETY: The handle is live and the value is plain.
        let status = unsafe { raw::mlx_array_set_int(self.raw_mut(), value) };
        check_status(status, "replace an MLX array value")
    }

    /// Replaces this handle's value with a float32 scalar.
    pub fn set_f32(&mut self, value: f32) -> Result<(), MlxCError> {
        // SAFETY: The handle is live and the value is plain.
        let status = unsafe { raw::mlx_array_set_float32(self.raw_mut(), value) };
        check_status(status, "replace an MLX array value")
    }

    /// Replaces this handle's value with a float64 scalar.
    pub fn set_f64(&mut self, value: f64) -> Result<(), MlxCError> {
        // SAFETY: The handle is live and the value is plain.
        let status = unsafe { raw::mlx_array_set_float64(self.raw_mut(), value) };
        check_status(status, "replace an MLX array value")
    }

    /// Replaces this handle's value with a complex scalar.
    pub fn set_complex(&mut self, real: f32, imag: f32) -> Result<(), MlxCError> {
        // SAFETY: The handle is live and the values are plain.
        let status = unsafe { raw::mlx_array_set_complex(self.raw_mut(), real, imag) };
        check_status(status, "replace an MLX array value")
    }

    /// The per-element byte size of the array's dtype.
    #[must_use]
    pub fn item_size(&self) -> usize {
        // SAFETY: `self` owns a live MLX array handle.
        unsafe { raw::mlx_array_itemsize(self.raw_array) }
    }

    /// The element strides in elements per dimension.
    #[must_use]
    pub fn strides(&self) -> Vec<usize> {
        // SAFETY: `self` owns a live MLX array handle.
        let dimension_count = unsafe { raw::mlx_array_ndim(self.raw_array) };
        if dimension_count == 0 {
            return Vec::new();
        }
        // SAFETY: MLX keeps this stride storage alive with the array and it
        // addresses exactly `dimension_count` entries.
        let stride_pointer = unsafe { raw::mlx_array_strides(self.raw_array) };
        if stride_pointer.is_null() {
            return Vec::new();
        }
        // SAFETY: The preceding MLX contract establishes the pointer length;
        // the values are copied before returning.
        unsafe { std::slice::from_raw_parts(stride_pointer, dimension_count) }.to_vec()
    }

    /// The size of one dimension.
    #[must_use]
    pub fn dimension(&self, axis: i32) -> i32 {
        // SAFETY: `self` owns a live MLX array handle.
        unsafe { raw::mlx_array_dim(self.raw_array, axis) }
    }

    /// The dtype's element byte size.
    #[must_use]
    pub fn byte_size_of_dtype(dtype: MlxDtype) -> usize {
        // SAFETY: The dtype query is a pure function of the tag.
        unsafe { raw::mlx_dtype_size(dtype.to_raw()) }
    }

    /// Copies the boolean scalar value.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the read fails.
    pub fn item_bool(&self) -> Result<bool, MlxCError> {
        let mut value = false;
        // SAFETY: The output pointer is valid writable storage and `self`
        // owns a live evaluated array.
        let status = unsafe { raw::mlx_array_item_bool(&mut value, self.raw_array) };
        check_status(status, "read an MLX boolean scalar")?;
        Ok(value)
    }

    /// Copies the uint8 scalar value.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the read fails.
    pub fn item_u8(&self) -> Result<u8, MlxCError> {
        read_scalar(self, raw::mlx_array_item_uint8, "read an MLX uint8 scalar")
    }

    /// Copies the uint16 scalar value.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the read fails.
    pub fn item_u16(&self) -> Result<u16, MlxCError> {
        read_scalar(
            self,
            raw::mlx_array_item_uint16,
            "read an MLX uint16 scalar",
        )
    }

    /// Copies the uint64 scalar value.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the read fails.
    pub fn item_u64(&self) -> Result<u64, MlxCError> {
        read_scalar(
            self,
            raw::mlx_array_item_uint64,
            "read an MLX uint64 scalar",
        )
    }

    /// Copies the int8 scalar value.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the read fails.
    pub fn item_i8(&self) -> Result<i8, MlxCError> {
        read_scalar(self, raw::mlx_array_item_int8, "read an MLX int8 scalar")
    }

    /// Copies the int16 scalar value.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the read fails.
    pub fn item_i16(&self) -> Result<i16, MlxCError> {
        read_scalar(self, raw::mlx_array_item_int16, "read an MLX int16 scalar")
    }

    /// Copies the int32 scalar value.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the read fails.
    pub fn item_i32(&self) -> Result<i32, MlxCError> {
        read_scalar(self, raw::mlx_array_item_int32, "read an MLX int32 scalar")
    }

    /// Copies the int64 scalar value.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the read fails.
    pub fn item_i64(&self) -> Result<i64, MlxCError> {
        read_scalar(self, raw::mlx_array_item_int64, "read an MLX int64 scalar")
    }

    /// Copies the float64 scalar value.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the read fails.
    pub fn item_f64(&self) -> Result<f64, MlxCError> {
        read_scalar(
            self,
            raw::mlx_array_item_float64,
            "read an MLX float64 scalar",
        )
    }

    /// Copies the complex scalar value as real and imaginary parts.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the read fails.
    pub fn item_complex(&self) -> Result<(f32, f32), MlxCError> {
        let mut value = raw::mlx_complex64_t { re: 0.0, im: 0.0 };
        // SAFETY: The output pointer is valid writable storage and `self`
        // owns a live evaluated array.
        let status = unsafe { raw::mlx_array_item_complex64(&mut value, self.raw_array) };
        check_status(status, "read an MLX complex scalar")?;
        Ok((value.re, value.im))
    }

    /// Copies the float16 scalar value as raw bits.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the read fails.
    pub fn item_f16_bits(&self) -> Result<u16, MlxCError> {
        let mut value = raw::__BindgenFloat16(0);
        // SAFETY: The output pointer is valid writable storage and `self`
        // owns a live evaluated array.
        let status = unsafe { raw::mlx_array_item_float16(&mut value, self.raw_array) };
        check_status(status, "read an MLX float16 scalar")?;
        Ok(value.0)
    }

    /// Copies the bfloat16 scalar value as raw bits.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the read fails.
    pub fn item_bf16_bits(&self) -> Result<u16, MlxCError> {
        let mut value: u16 = 0;
        // SAFETY: The output pointer is valid writable storage and `self`
        // owns a live evaluated array.
        let status = unsafe { raw::mlx_array_item_bfloat16(&mut value, self.raw_array) };
        check_status(status, "read an MLX bfloat16 scalar")?;
        Ok(value)
    }

    /// Copies the evaluated boolean storage into Rust memory.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the copy fails.
    pub fn to_vec_bool(&self) -> Result<Vec<bool>, MlxCError> {
        self.copy_evaluated_storage(
            |array| unsafe { raw::mlx_array_data_bool(array) },
            "copy an MLX boolean array",
        )
    }

    /// Copies the evaluated uint16 storage into Rust memory.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the copy fails.
    pub fn to_vec_u16(&self) -> Result<Vec<u16>, MlxCError> {
        self.copy_evaluated_storage(
            |array| unsafe { raw::mlx_array_data_uint16(array) },
            "copy an MLX uint16 array",
        )
    }

    /// Copies the evaluated uint64 storage into Rust memory.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the copy fails.
    pub fn to_vec_u64(&self) -> Result<Vec<u64>, MlxCError> {
        self.copy_evaluated_storage(
            |array| unsafe { raw::mlx_array_data_uint64(array) },
            "copy an MLX uint64 array",
        )
    }

    /// Copies the evaluated int8 storage into Rust memory.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the copy fails.
    pub fn to_vec_i8(&self) -> Result<Vec<i8>, MlxCError> {
        self.copy_evaluated_storage(
            |array| unsafe { raw::mlx_array_data_int8(array) },
            "copy an MLX int8 array",
        )
    }

    /// Copies the evaluated int16 storage into Rust memory.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the copy fails.
    pub fn to_vec_i16(&self) -> Result<Vec<i16>, MlxCError> {
        self.copy_evaluated_storage(
            |array| unsafe { raw::mlx_array_data_int16(array) },
            "copy an MLX int16 array",
        )
    }

    /// Copies the evaluated int32 storage into Rust memory.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the copy fails.
    pub fn to_vec_i32(&self) -> Result<Vec<i32>, MlxCError> {
        self.copy_evaluated_storage(
            |array| unsafe { raw::mlx_array_data_int32(array) },
            "copy an MLX int32 array",
        )
    }

    /// Copies the evaluated int64 storage into Rust memory.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the copy fails.
    pub fn to_vec_i64(&self) -> Result<Vec<i64>, MlxCError> {
        self.copy_evaluated_storage(
            |array| unsafe { raw::mlx_array_data_int64(array) },
            "copy an MLX int64 array",
        )
    }

    /// Copies the evaluated float64 storage into Rust memory.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the copy fails.
    pub fn to_vec_f64(&self) -> Result<Vec<f64>, MlxCError> {
        self.copy_evaluated_storage(
            |array| unsafe { raw::mlx_array_data_float64(array) },
            "copy an MLX float64 array",
        )
    }

    /// Copies the evaluated float16 storage into Rust memory as raw bits.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the copy fails.
    pub fn to_vec_f16_bits(&self) -> Result<Vec<u16>, MlxCError> {
        self.copy_evaluated_storage(
            |array| unsafe { raw::mlx_array_data_float16(array) }.cast(),
            "copy an MLX float16 array",
        )
    }

    /// Copies the evaluated bfloat16 storage into Rust memory as raw bits.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when the copy fails.
    pub fn to_vec_bf16_bits(&self) -> Result<Vec<u16>, MlxCError> {
        self.copy_evaluated_storage(
            |array| unsafe { raw::mlx_array_data_bfloat16(array) },
            "copy an MLX bfloat16 array",
        )
    }

    fn copy_evaluated_storage<T: Clone>(
        &self,
        data_pointer: impl Fn(raw::mlx_array) -> *const T,
        operation: &'static str,
    ) -> Result<Vec<T>, MlxCError> {
        self.evaluate()?;
        let element_count = self.element_count();
        if element_count == 0 {
            return Ok(Vec::new());
        }
        // SAFETY: Evaluation materializes contiguous readable storage owned
        // by the live array for at least `element_count` values.
        let values_pointer = data_pointer(self.raw_array);
        if values_pointer.is_null() {
            return Err(MlxCError {
                operation,
                description: "MLX returned a null data pointer after evaluation".to_owned(),
            });
        }
        // SAFETY: The evaluated array establishes the pointer's exact element
        // count; values are copied before the borrow ends.
        let copied = unsafe { std::slice::from_raw_parts(values_pointer, element_count) }.to_vec();
        Ok(copied)
    }
}

fn read_scalar<T>(
    array: &MlxArray,
    reader: unsafe extern "C" fn(*mut T, raw::mlx_array) -> i32,
    operation: &'static str,
) -> Result<T, MlxCError> {
    let mut value: T = unsafe { std::mem::zeroed() };
    // SAFETY: The output pointer is valid writable storage and `array` owns
    // a live evaluated array.
    let status = unsafe { reader(&mut value, array.raw()) };
    check_status(status, operation)?;
    Ok(value)
}

/// A host buffer whose lifetime MLX owns through the managed-array
/// destructor.
struct ManagedBuffer {
    storage: Vec<u8>,
}

/// The destructor trampoline for payload-managed buffers.
unsafe extern "C" fn drop_managed_payload(payload: *mut std::os::raw::c_void) {
    if payload.is_null() {
        return;
    }
    // SAFETY: The payload was created by `Box::into_raw` of a `ManagedBuffer`
    // and this destructor runs exactly once per managed array.
    drop(unsafe { Box::from_raw(payload.cast::<ManagedBuffer>()) });
}
