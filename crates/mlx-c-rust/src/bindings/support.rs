//! Shared helpers for the generated MLX-C operation wrappers.
//!
//! Every wrapper funnels through the same conventions: borrowed arrays pass
//! their live raw handles, absent optional arrays use MLX's official empty
//! placeholder, host string arguments are copied into C strings up front, and
//! results land in uniquely writable output handles that enter RAII
//! ownership before fallible checks run.

use std::ffi::CString;

use crate::error::check_status;
use crate::raw;
use crate::{MlxArray, MlxArrayVector, MlxBindingsContext, MlxCError, MlxDtype};

/// Copies a host string argument for a C API call that requires UTF-8 text
/// without interior null bytes.
pub(crate) fn c_string_argument(
    value: &str,
    operation: &'static str,
) -> Result<CString, MlxCError> {
    CString::new(value).map_err(|_| MlxCError {
        operation,
        description: "string argument contains an interior null byte".to_owned(),
    })
}

/// The C pointer for an optional `const int*` list argument; an empty slice
/// passes the null pointer with a zero count.
pub(crate) fn optional_i32_slice(values: &[i32]) -> *const i32 {
    if values.is_empty() {
        std::ptr::null()
    } else {
        values.as_ptr()
    }
}

/// The C pointer for an optional `const int64_t*` list argument.
pub(crate) fn optional_i64_slice(values: &[i64]) -> *const i64 {
    if values.is_empty() {
        std::ptr::null()
    } else {
        values.as_ptr()
    }
}

/// Builds the raw optional-integer value for MLX-C optional parameters.
pub(crate) fn raw_optional_int(value: Option<i32>) -> raw::mlx_optional_int {
    match value {
        Some(present_value) => raw::mlx_optional_int {
            value: present_value,
            has_value: true,
        },
        None => raw::mlx_optional_int {
            value: 0,
            has_value: false,
        },
    }
}

/// Builds the raw optional-float value for MLX-C optional parameters.
pub(crate) fn raw_optional_float(value: Option<f32>) -> raw::mlx_optional_float {
    match value {
        Some(present_value) => raw::mlx_optional_float {
            value: present_value,
            has_value: true,
        },
        None => raw::mlx_optional_float {
            value: 0.0,
            has_value: false,
        },
    }
}

/// Builds the raw optional-dtype value for MLX-C optional parameters.
pub(crate) fn raw_optional_dtype(value: Option<MlxDtype>) -> raw::mlx_optional_dtype {
    match value {
        Some(present_value) => raw::mlx_optional_dtype {
            value: present_value.to_raw(),
            has_value: true,
        },
        None => raw::mlx_optional_dtype {
            value: raw::mlx_dtype__MLX_BOOL,
            has_value: false,
        },
    }
}

impl MlxBindingsContext {
    /// Builds a vector-of-arrays result on the runtime GPU stream.
    pub(crate) fn output_vector_array(
        &self,
        operation: &'static str,
        build_graph: impl FnOnce(*mut raw::mlx_vector_array, raw::mlx_stream) -> i32,
    ) -> Result<MlxArrayVector, MlxCError> {
        let mut output = MlxArrayVector::empty(operation)?;
        let status = build_graph(output.raw_mut(), self.gpu_stream().raw());
        check_status(status, operation)?;
        Ok(output)
    }

    /// Builds two array results on the runtime GPU stream.
    pub(crate) fn output_array_pair(
        &self,
        operation: &'static str,
        build_graph: impl FnOnce(*mut raw::mlx_array, *mut raw::mlx_array, raw::mlx_stream) -> i32,
    ) -> Result<(MlxArray, MlxArray), MlxCError> {
        let mut first_output = MlxArray::empty();
        let mut second_output = MlxArray::empty();
        let status = build_graph(
            first_output.raw_mut(),
            second_output.raw_mut(),
            self.gpu_stream().raw(),
        );
        check_status(status, operation)?;
        first_output.require_populated(operation)?;
        second_output.require_populated(operation)?;
        Ok((first_output, second_output))
    }
}
