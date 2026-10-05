//! The custom MLX closure domain: the custom-transform (VJP), custom-JVP,
//! and custom-VMAP closure variants built from Rust callables through
//! payload trampolines.
//!
//! Every payload is a `RefCell` of the Rust callable boxed into C memory; the
//! matching destructor reclaims it. Callback panics are caught at the FFI
//! boundary, converted to typed failures, and parked thread-locally for the
//! caller's status translation so nothing unwinds across the C ABI. The
//! shared trampoline machinery lives in `crate::types::closures`.

use std::cell::RefCell;
use std::os::raw::c_void;

use crate::error::check_status;
use crate::raw;
use crate::types::closures::{
    drop_payload, owned_input_arrays, run_payload_closure, write_output_arrays,
};
use crate::types::vectors::MlxVectorInt;
use crate::{MlxArray, MlxArrayVector, MlxCError};

/// Owned MLX custom-transform (VJP) closure, released exactly once.
#[derive(Debug)]
pub struct MlxClosureCustom(raw::mlx_closure_custom);

impl MlxClosureCustom {
    /// Builds an owned custom-transform closure from a Rust callable that
    /// receives inputs, outputs, and upstream gradients.
    pub fn from_fn<F>(callback: F) -> Result<Self, MlxCError>
    where
        F: FnMut(&[MlxArray], &[MlxArray], &[MlxArray]) -> Result<Vec<MlxArray>, MlxCError>
            + 'static,
    {
        let operation: &'static str = "create an MLX custom-transform closure";
        let payload = Box::into_raw(Box::new(RefCell::new(callback)));
        // SAFETY: The trampoline and destructor follow the exact C ABI and
        // the returned handle enters RAII ownership.
        let raw_closure = unsafe {
            raw::mlx_closure_custom_new_func_payload(
                Some(custom_callback_trampoline::<F>),
                payload.cast(),
                Some(drop_payload::<F>),
            )
        };
        let closure = Self(raw_closure);
        if closure.0.ctx.is_null() {
            return Err(MlxCError {
                operation,
                description: "MLX returned an empty closure handle".to_owned(),
            });
        }
        Ok(closure)
    }

    /// Creates an empty custom-transform closure handle.
    pub fn empty() -> Self {
        // SAFETY: The returned handle enters RAII ownership immediately.
        let raw_closure = unsafe { raw::mlx_closure_custom_new() };
        Self(raw_closure)
    }

    /// Copies the source closure handle into this handle.
    pub fn set(&mut self, source: &Self) -> Result<(), MlxCError> {
        // SAFETY: Both handles are live and this call only replaces the
        // owned closure value.
        let status = unsafe { raw::mlx_closure_custom_set(&mut self.0, source.0) };
        check_status(status, "copy an MLX custom-transform closure")
    }

    /// Applies the custom transform.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when application fails.
    pub fn apply(
        &self,
        inputs: &MlxArrayVector,
        outputs: &MlxArrayVector,
        upstream_gradients: &MlxArrayVector,
    ) -> Result<MlxArrayVector, MlxCError> {
        let operation: &'static str = "apply an MLX custom-transform closure";
        let mut output = MlxArrayVector::empty(operation)?;
        // SAFETY: All handles are live and the output pointer is valid
        // writable storage.
        let status = unsafe {
            raw::mlx_closure_custom_apply(
                output.raw_mut(),
                self.0,
                inputs.raw(),
                outputs.raw(),
                upstream_gradients.raw(),
            )
        };
        check_status(status, operation)?;
        Ok(output)
    }
}

impl MlxClosureCustom {
    pub(crate) const fn raw(&self) -> raw::mlx_closure_custom {
        self.0
    }
}

impl Drop for MlxClosureCustom {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live closure exactly once and
        // never accesses the handle afterward.
        unsafe {
            raw::mlx_closure_custom_free(self.0);
        }
    }
}

unsafe extern "C" fn custom_callback_trampoline<F>(
    output: *mut raw::mlx_vector_array,
    inputs: raw::mlx_vector_array,
    outputs: raw::mlx_vector_array,
    upstream_gradients: raw::mlx_vector_array,
    payload: *mut c_void,
) -> i32
where
    F: FnMut(&[MlxArray], &[MlxArray], &[MlxArray]) -> Result<Vec<MlxArray>, MlxCError> + 'static,
{
    let Some(result) = run_payload_closure::<F, Result<Vec<MlxArray>, MlxCError>>(
        payload,
        "apply a Rust custom-transform closure",
        |callback| {
            let owned_inputs = owned_input_arrays(inputs)?;
            let owned_outputs = owned_input_arrays(outputs)?;
            let owned_gradients = owned_input_arrays(upstream_gradients)?;
            callback(&owned_inputs, &owned_outputs, &owned_gradients)
        },
    ) else {
        return 1;
    };
    match result.and_then(|owned| write_output_arrays(output, owned)) {
        Ok(()) => 0,
        Err(failure) => {
            crate::error::set_closure_error(failure);
            1
        }
    }
}

/// Owned MLX custom JVP closure, released exactly once.
#[derive(Debug)]
pub struct MlxClosureCustomJvp(raw::mlx_closure_custom_jvp);

impl MlxClosureCustomJvp {
    /// Builds an owned custom-JVP closure from a Rust callable that receives
    /// primals, tangents, and the requested output indices.
    pub fn from_fn<F>(callback: F) -> Result<Self, MlxCError>
    where
        F: FnMut(&[MlxArray], &[MlxArray], &[i32]) -> Result<Vec<MlxArray>, MlxCError> + 'static,
    {
        let operation: &'static str = "create an MLX custom-JVP closure";
        let payload = Box::into_raw(Box::new(RefCell::new(callback)));
        // SAFETY: The trampoline and destructor follow the exact C ABI and
        // the returned handle enters RAII ownership.
        let raw_closure = unsafe {
            raw::mlx_closure_custom_jvp_new_func_payload(
                Some(custom_jvp_callback_trampoline::<F>),
                payload.cast(),
                Some(drop_payload::<F>),
            )
        };
        let closure = Self(raw_closure);
        if closure.0.ctx.is_null() {
            return Err(MlxCError {
                operation,
                description: "MLX returned an empty closure handle".to_owned(),
            });
        }
        Ok(closure)
    }

    /// Creates an empty custom-JVP closure handle.
    pub fn empty() -> Self {
        // SAFETY: The returned handle enters RAII ownership immediately.
        let raw_closure = unsafe { raw::mlx_closure_custom_jvp_new() };
        Self(raw_closure)
    }

    /// Copies the source closure handle into this handle.
    pub fn set(&mut self, source: &Self) -> Result<(), MlxCError> {
        // SAFETY: Both handles are live and this call only replaces the
        // owned closure value.
        let status = unsafe { raw::mlx_closure_custom_jvp_set(&mut self.0, source.0) };
        check_status(status, "copy an MLX custom-JVP closure")
    }

    /// Applies the custom JVP transform.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when application fails.
    pub fn apply(
        &self,
        primals: &MlxArrayVector,
        tangents: &MlxArrayVector,
        output_indices: &[i32],
    ) -> Result<MlxArrayVector, MlxCError> {
        let operation: &'static str = "apply an MLX custom-JVP closure";
        let mut output = MlxArrayVector::empty(operation)?;
        // SAFETY: All handles are live, the slice remains valid for the
        // call, and the output pointer is valid writable storage.
        let status = unsafe {
            raw::mlx_closure_custom_jvp_apply(
                output.raw_mut(),
                self.0,
                primals.raw(),
                tangents.raw(),
                output_indices.as_ptr(),
                output_indices.len(),
            )
        };
        check_status(status, operation)?;
        Ok(output)
    }
}

impl MlxClosureCustomJvp {
    pub(crate) const fn raw(&self) -> raw::mlx_closure_custom_jvp {
        self.0
    }
}

impl Drop for MlxClosureCustomJvp {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live closure exactly once and
        // never accesses the handle afterward.
        unsafe {
            raw::mlx_closure_custom_jvp_free(self.0);
        }
    }
}

unsafe extern "C" fn custom_jvp_callback_trampoline<F>(
    output: *mut raw::mlx_vector_array,
    primals: raw::mlx_vector_array,
    tangents: raw::mlx_vector_array,
    output_indices: *const i32,
    output_index_count: usize,
    payload: *mut c_void,
) -> i32
where
    F: FnMut(&[MlxArray], &[MlxArray], &[i32]) -> Result<Vec<MlxArray>, MlxCError> + 'static,
{
    let Some(result) = run_payload_closure::<F, Result<Vec<MlxArray>, MlxCError>>(
        payload,
        "apply a Rust custom-JVP closure",
        |callback| {
            let owned_primals = owned_input_arrays(primals)?;
            let owned_tangents = owned_input_arrays(tangents)?;
            // SAFETY: MLX loans the index storage owned by the live call
            // arguments for the trampoline's duration.
            let borrowed_indices =
                unsafe { std::slice::from_raw_parts(output_indices, output_index_count) };
            callback(&owned_primals, &owned_tangents, borrowed_indices)
        },
    ) else {
        return 1;
    };
    match result.and_then(|owned| write_output_arrays(output, owned)) {
        Ok(()) => 0,
        Err(failure) => {
            crate::error::set_closure_error(failure);
            1
        }
    }
}

/// Owned MLX custom VMAP closure, released exactly once.
#[derive(Debug)]
pub struct MlxClosureCustomVmap(raw::mlx_closure_custom_vmap);

impl MlxClosureCustomVmap {
    /// Builds an owned custom-VMAP closure from a Rust callable that
    /// receives inputs and input axes and returns outputs with output axes.
    pub fn from_fn<F>(callback: F) -> Result<Self, MlxCError>
    where
        F: FnMut(&[MlxArray], &[i32]) -> Result<(Vec<MlxArray>, Vec<i32>), MlxCError> + 'static,
    {
        let operation: &'static str = "create an MLX custom-VMAP closure";
        let payload = Box::into_raw(Box::new(RefCell::new(callback)));
        // SAFETY: The trampoline and destructor follow the exact C ABI and
        // the returned handle enters RAII ownership.
        let raw_closure = unsafe {
            raw::mlx_closure_custom_vmap_new_func_payload(
                Some(custom_vmap_callback_trampoline::<F>),
                payload.cast(),
                Some(drop_payload::<F>),
            )
        };
        let closure = Self(raw_closure);
        if closure.0.ctx.is_null() {
            return Err(MlxCError {
                operation,
                description: "MLX returned an empty closure handle".to_owned(),
            });
        }
        Ok(closure)
    }

    /// Creates an empty custom-VMAP closure handle.
    pub fn empty() -> Self {
        // SAFETY: The returned handle enters RAII ownership immediately.
        let raw_closure = unsafe { raw::mlx_closure_custom_vmap_new() };
        Self(raw_closure)
    }

    /// Copies the source closure handle into this handle.
    pub fn set(&mut self, source: &Self) -> Result<(), MlxCError> {
        // SAFETY: Both handles are live and this call only replaces the
        // owned closure value.
        let status = unsafe { raw::mlx_closure_custom_vmap_set(&mut self.0, source.0) };
        check_status(status, "copy an MLX custom-VMAP closure")
    }

    /// Applies the custom VMAP transform.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when application fails.
    pub fn apply(
        &self,
        inputs: &MlxArrayVector,
        input_axes: &[i32],
    ) -> Result<(MlxArrayVector, MlxVectorInt), MlxCError> {
        let operation: &'static str = "apply an MLX custom-VMAP closure";
        let mut outputs = MlxArrayVector::empty(operation)?;
        let mut output_axes = MlxVectorInt::empty();
        // SAFETY: All handles are live, the slice remains valid for the
        // call, and both output pointers are valid writable storage.
        let status = unsafe {
            raw::mlx_closure_custom_vmap_apply(
                outputs.raw_mut(),
                output_axes.raw_mut(),
                self.0,
                inputs.raw(),
                input_axes.as_ptr(),
                input_axes.len(),
            )
        };
        check_status(status, operation)?;
        Ok((outputs, output_axes))
    }
}

impl MlxClosureCustomVmap {
    pub(crate) const fn raw(&self) -> raw::mlx_closure_custom_vmap {
        self.0
    }
}

impl Drop for MlxClosureCustomVmap {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live closure exactly once and
        // never accesses the handle afterward.
        unsafe {
            raw::mlx_closure_custom_vmap_free(self.0);
        }
    }
}

unsafe extern "C" fn custom_vmap_callback_trampoline<F>(
    outputs: *mut raw::mlx_vector_array,
    output_axes: *mut raw::mlx_vector_int,
    inputs: raw::mlx_vector_array,
    input_axes: *const i32,
    input_axis_count: usize,
    payload: *mut c_void,
) -> i32
where
    F: FnMut(&[MlxArray], &[i32]) -> Result<(Vec<MlxArray>, Vec<i32>), MlxCError> + 'static,
{
    let Some(result) = run_payload_closure::<F, Result<(Vec<MlxArray>, Vec<i32>), MlxCError>>(
        payload,
        "apply a Rust custom-VMAP closure",
        |callback| {
            let owned_inputs = owned_input_arrays(inputs)?;
            // SAFETY: MLX loans the axis storage owned by the live call
            // arguments for the trampoline's duration.
            let borrowed_axes = unsafe { std::slice::from_raw_parts(input_axes, input_axis_count) };
            callback(&owned_inputs, borrowed_axes)
        },
    ) else {
        return 1;
    };
    let outcome = match result {
        Ok((owned_outputs, owned_axes)) => {
            if let Err(failure) = write_output_arrays(outputs, owned_outputs) {
                Err(failure)
            } else {
                // SAFETY: The output pointer is valid writable storage and
                // the value storage remains valid for this copying call.
                let status = unsafe {
                    raw::mlx_vector_int_set_data(
                        output_axes,
                        owned_axes.as_ptr().cast_mut(),
                        owned_axes.len(),
                    )
                };
                check_status(status, "write custom-VMAP closure output axes")
            }
        }
        Err(failure) => Err(failure),
    };
    match outcome {
        Ok(()) => 0,
        Err(failure) => {
            crate::error::set_closure_error(failure);
            1
        }
    }
}
