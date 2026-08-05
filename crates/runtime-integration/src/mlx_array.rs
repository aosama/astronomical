use crate::{
    MlxRuntimeError,
    mlx_runtime::{
        check_status, classify_mlx_error, clear_captured_mlx_error, take_captured_mlx_error,
    },
    raw,
};

/// MLX array element types exposed without leaking generated C declarations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlxDtype {
    Bool,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float16,
    Float32,
    Float64,
    BFloat16,
    Complex64,
}

impl MlxDtype {
    fn from_raw(raw_dtype: raw::mlx_dtype) -> Self {
        match raw_dtype {
            raw::mlx_dtype__MLX_BOOL => Self::Bool,
            raw::mlx_dtype__MLX_UINT8 => Self::UInt8,
            raw::mlx_dtype__MLX_UINT16 => Self::UInt16,
            raw::mlx_dtype__MLX_UINT32 => Self::UInt32,
            raw::mlx_dtype__MLX_UINT64 => Self::UInt64,
            raw::mlx_dtype__MLX_INT8 => Self::Int8,
            raw::mlx_dtype__MLX_INT16 => Self::Int16,
            raw::mlx_dtype__MLX_INT32 => Self::Int32,
            raw::mlx_dtype__MLX_INT64 => Self::Int64,
            raw::mlx_dtype__MLX_FLOAT16 => Self::Float16,
            raw::mlx_dtype__MLX_FLOAT32 => Self::Float32,
            raw::mlx_dtype__MLX_FLOAT64 => Self::Float64,
            raw::mlx_dtype__MLX_BFLOAT16 => Self::BFloat16,
            raw::mlx_dtype__MLX_COMPLEX64 => Self::Complex64,
            _ => Self::Bool,
        }
    }

    pub(crate) const fn to_raw(self) -> raw::mlx_dtype {
        match self {
            Self::Bool => raw::mlx_dtype__MLX_BOOL,
            Self::UInt8 => raw::mlx_dtype__MLX_UINT8,
            Self::UInt16 => raw::mlx_dtype__MLX_UINT16,
            Self::UInt32 => raw::mlx_dtype__MLX_UINT32,
            Self::UInt64 => raw::mlx_dtype__MLX_UINT64,
            Self::Int8 => raw::mlx_dtype__MLX_INT8,
            Self::Int16 => raw::mlx_dtype__MLX_INT16,
            Self::Int32 => raw::mlx_dtype__MLX_INT32,
            Self::Int64 => raw::mlx_dtype__MLX_INT64,
            Self::Float16 => raw::mlx_dtype__MLX_FLOAT16,
            Self::Float32 => raw::mlx_dtype__MLX_FLOAT32,
            Self::Float64 => raw::mlx_dtype__MLX_FLOAT64,
            Self::BFloat16 => raw::mlx_dtype__MLX_BFLOAT16,
            Self::Complex64 => raw::mlx_dtype__MLX_COMPLEX64,
        }
    }
}

/// Owned MLX array handle released exactly once through the official C API.
#[derive(Debug)]
pub struct MlxArray {
    raw_array: raw::mlx_array,
}

impl MlxArray {
    pub(crate) fn from_f32(values: &[f32], shape: &[i32]) -> Result<Self, MlxRuntimeError> {
        validate_shape(values.len(), shape)?;
        let dimension_count =
            i32::try_from(shape.len()).map_err(|_| MlxRuntimeError::RuntimeOperation {
                operation: "create an MLX float32 array",
                description: "array rank exceeds the C API integer range".to_owned(),
            })?;
        // SAFETY: The slices remain valid for this copying constructor, their
        // lengths were validated, and the returned handle enters RAII ownership.
        let raw_array = unsafe {
            raw::mlx_array_new_data(
                values.as_ptr().cast(),
                shape.as_ptr(),
                dimension_count,
                raw::mlx_dtype__MLX_FLOAT32,
            )
        };
        let array = Self { raw_array };
        array.require_populated("create an MLX float32 array")?;
        Ok(array)
    }

    pub(crate) fn from_i32(values: &[i32], shape: &[i32]) -> Result<Self, MlxRuntimeError> {
        validate_shape(values.len(), shape)?;
        let dimension_count =
            i32::try_from(shape.len()).map_err(|_| MlxRuntimeError::RuntimeOperation {
                operation: "create an MLX int32 array",
                description: "array rank exceeds the C API integer range".to_owned(),
            })?;
        // SAFETY: The slices remain valid for this copying constructor, their
        // lengths were validated, and the returned handle enters RAII ownership.
        let raw_array = unsafe {
            raw::mlx_array_new_data(
                values.as_ptr().cast(),
                shape.as_ptr(),
                dimension_count,
                raw::mlx_dtype__MLX_INT32,
            )
        };
        let array = Self { raw_array };
        array.require_populated("create an MLX int32 array")?;
        Ok(array)
    }

    pub(crate) fn from_u32(values: &[u32], shape: &[i32]) -> Result<Self, MlxRuntimeError> {
        validate_shape(values.len(), shape)?;
        let dimension_count =
            i32::try_from(shape.len()).map_err(|_| MlxRuntimeError::RuntimeOperation {
                operation: "create an MLX uint32 array",
                description: "array rank exceeds the C API integer range".to_owned(),
            })?;
        // SAFETY: The slices remain valid for this copying constructor, their
        // lengths were validated, and the returned handle enters RAII ownership.
        let raw_array = unsafe {
            raw::mlx_array_new_data(
                values.as_ptr().cast(),
                shape.as_ptr(),
                dimension_count,
                raw::mlx_dtype__MLX_UINT32,
            )
        };
        let array = Self { raw_array };
        array.require_populated("create an MLX uint32 array")?;
        Ok(array)
    }

    pub(crate) fn empty() -> Self {
        // SAFETY: The runtime error handler is installed before model loading,
        // and the official API's empty placeholder is immediately placed under
        // RAII ownership before a fallible operation populates it.
        let raw_array = unsafe { raw::mlx_array_new() };
        Self { raw_array }
    }

    #[cfg(feature = "experimental-aligned-expert-packs")]
    pub(crate) fn from_populated_owned_raw(raw_array: raw::mlx_array) -> Self {
        debug_assert!(!raw_array.ctx.is_null());
        Self { raw_array }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.raw_array.ctx.is_null()
    }

    pub(crate) const fn raw(&self) -> raw::mlx_array {
        self.raw_array
    }

    pub(crate) const fn empty_raw() -> raw::mlx_array {
        raw::mlx_array {
            ctx: std::ptr::null_mut(),
        }
    }

    pub(crate) fn raw_mut(&mut self) -> *mut raw::mlx_array {
        &mut self.raw_array
    }

    pub(crate) fn require_populated(&self, operation: &'static str) -> Result<(), MlxRuntimeError> {
        if !self.is_empty() {
            clear_captured_mlx_error();
            return Ok(());
        }
        let description = take_captured_mlx_error()
            .unwrap_or_else(|| "MLX returned an empty array handle".to_owned());
        Err(classify_mlx_error(operation, description))
    }

    /// Returns the dimensions of this live array.
    #[must_use]
    pub fn shape(&self) -> Vec<i32> {
        // SAFETY: `self` exclusively owns a live MLX array handle.
        let dimension_count = unsafe { raw::mlx_array_ndim(self.raw_array) };
        if dimension_count == 0 {
            return Vec::new();
        }
        // SAFETY: MLX keeps this shape storage alive with the array. The
        // pointer addresses exactly `dimension_count` dimensions.
        let shape_pointer = unsafe { raw::mlx_array_shape(self.raw_array) };
        if shape_pointer.is_null() {
            return Vec::new();
        }
        // SAFETY: The preceding MLX contract establishes the pointer length;
        // this method copies the values before returning.
        unsafe { std::slice::from_raw_parts(shape_pointer, dimension_count) }.to_vec()
    }

    /// Returns the array's element type.
    #[must_use]
    pub fn dtype(&self) -> MlxDtype {
        // SAFETY: `self` exclusively owns a live MLX array handle.
        MlxDtype::from_raw(unsafe { raw::mlx_array_dtype(self.raw_array) })
    }

    /// Returns the number of logical elements.
    #[must_use]
    pub fn element_count(&self) -> usize {
        // SAFETY: `self` exclusively owns a live MLX array handle.
        unsafe { raw::mlx_array_size(self.raw_array) }
    }

    /// Returns the logical tensor payload size in bytes.
    #[must_use]
    pub fn byte_count(&self) -> usize {
        // SAFETY: `self` exclusively owns a live MLX array handle.
        unsafe { raw::mlx_array_nbytes(self.raw_array) }
    }

    /// Materializes this lazy array and reports runtime failures without exit.
    pub fn evaluate(&self) -> Result<(), MlxRuntimeError> {
        // SAFETY: `self` owns a live handle and MLX evaluation does not retain
        // any Rust borrow beyond this call.
        let status = unsafe { raw::mlx_array_eval(self.raw_array) };
        check_status(status, "evaluate an MLX array")
    }

    /// Evaluates and copies a float32 array into Rust-owned memory.
    pub fn to_vec_f32(&self) -> Result<Vec<f32>, MlxRuntimeError> {
        if self.dtype() != MlxDtype::Float32 {
            return Err(MlxRuntimeError::RuntimeOperation {
                operation: "copy an MLX float32 array",
                description: "array dtype is not float32".to_owned(),
            });
        }
        self.evaluate()?;
        let element_count = self.element_count();
        if element_count == 0 {
            return Ok(Vec::new());
        }
        // SAFETY: Evaluation materializes contiguous readable storage owned by
        // the live array for at least `element_count` float32 values.
        let values_pointer = unsafe { raw::mlx_array_data_float32(self.raw_array) };
        if values_pointer.is_null() {
            return Err(MlxRuntimeError::RuntimeOperation {
                operation: "copy an MLX float32 array",
                description: "MLX returned a null data pointer after evaluation".to_owned(),
            });
        }
        // SAFETY: The evaluated MLX array establishes the pointer's exact
        // element count; values are copied before the borrow ends.
        Ok(unsafe { std::slice::from_raw_parts(values_pointer, element_count) }.to_vec())
    }

    pub fn to_vec_u32(&self) -> Result<Vec<u32>, MlxRuntimeError> {
        if self.dtype() != MlxDtype::UInt32 {
            return Err(MlxRuntimeError::RuntimeOperation {
                operation: "copy an MLX uint32 array",
                description: "array dtype is not uint32".to_owned(),
            });
        }
        self.evaluate()?;
        self.copy_evaluated_u32_values()
    }

    /// Copies an already evaluated uint32 array into Rust-owned memory.
    ///
    /// Callers must evaluate the array first. This split exists so performance
    /// attribution can distinguish the synchronization wait from the small host copy.
    pub fn copy_evaluated_u32_values(&self) -> Result<Vec<u32>, MlxRuntimeError> {
        if self.dtype() != MlxDtype::UInt32 {
            return Err(MlxRuntimeError::RuntimeOperation {
                operation: "copy an evaluated MLX uint32 array",
                description: "array dtype is not uint32".to_owned(),
            });
        }
        let element_count = self.element_count();
        if element_count == 0 {
            return Ok(Vec::new());
        }
        // SAFETY: Evaluation materializes contiguous readable storage owned by
        // the live array for at least `element_count` uint32 values.
        let values_pointer = unsafe { raw::mlx_array_data_uint32(self.raw_array) };
        if values_pointer.is_null() {
            return Err(MlxRuntimeError::RuntimeOperation {
                operation: "copy an evaluated MLX uint32 array",
                description: "MLX returned a null data pointer for an evaluated array".to_owned(),
            });
        }
        // SAFETY: The evaluated MLX array establishes the pointer's exact
        // element count; values are copied before the borrow ends.
        Ok(unsafe { std::slice::from_raw_parts(values_pointer, element_count) }.to_vec())
    }

    /// Evaluates a scalar array and copies its value as an unsigned token ID.
    pub fn item_u32(&self) -> Result<u32, MlxRuntimeError> {
        self.evaluate()?;
        let mut scalar_value = 0_u32;
        // SAFETY: The output pointer is valid writable storage and `self` owns
        // a live evaluated array. MLX validates scalar shape and conversion.
        let status = unsafe { raw::mlx_array_item_uint32(&mut scalar_value, self.raw_array) };
        check_status(status, "read an MLX uint32 scalar")?;
        Ok(scalar_value)
    }

    /// Creates a shared reference to the same underlying MLX data.
    ///
    /// MLX uses reference counting, so this is cheap and does not copy the
    /// tensor data. The returned array shares the same lazy graph and
    /// evaluated buffers as the source.
    pub fn retain(&self) -> Result<Self, MlxRuntimeError> {
        let mut retained_array = Self::empty();
        // SAFETY: `retained_array` owns a live (empty) handle and `self` owns a
        // live source handle. `mlx_array_set` copies the source reference into
        // the destination.
        let status = unsafe { raw::mlx_array_set(retained_array.raw_mut(), self.raw_array) };
        check_status(status, "retain an MLX array")?;
        if retained_array.is_empty() {
            return Err(MlxRuntimeError::RuntimeOperation {
                operation: "retain an MLX array",
                description: "MLX returned an empty handle after set".to_owned(),
            });
        }
        Ok(retained_array)
    }
}

fn validate_shape(element_count: usize, shape: &[i32]) -> Result<(), MlxRuntimeError> {
    let shaped_element_count = shape.iter().try_fold(1_usize, |product, dimension| {
        let dimension = usize::try_from(*dimension).ok()?;
        product.checked_mul(dimension)
    });
    if shaped_element_count != Some(element_count) {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: "create an MLX array",
            description: "shape element count does not match the provided values".to_owned(),
        });
    }
    Ok(())
}

impl Drop for MlxArray {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live array exactly once and never
        // accesses the handle afterward.
        unsafe {
            raw::mlx_array_free(self.raw_array);
        }
    }
}
