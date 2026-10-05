//! The MLX closure domain: owned closures built from Rust callables through
//! payload trampolines, plus the kwargs and value-and-gradient variants.
//! The custom-transform family lives in `crate::types::closure_custom` and
//! shares this module's trampoline machinery.
//!
//! Every payload is a `RefCell` of the Rust callable boxed into C memory; the
//! matching destructor reclaims it. Callback panics are caught at the FFI
//! boundary, converted to typed failures, and parked thread-locally for the
//! caller's status translation so nothing unwinds across the C ABI.

use std::cell::RefCell;
use std::os::raw::c_void;

use crate::error::check_status;
use crate::raw;
use crate::types::maps::MlxMapStringToArray;
use crate::{MlxArray, MlxArrayVector, MlxCError};

/// Reads every array of a live input vector into owned handles.
///
/// # Errors
/// Returns the captured MLX-C description when an element read fails.
pub(crate) fn owned_input_arrays(input: raw::mlx_vector_array) -> Result<Vec<MlxArray>, MlxCError> {
    let element_count = unsafe { raw::mlx_vector_array_size(input) };
    let mut elements = Vec::with_capacity(element_count);
    for element_index in 0..element_count {
        let mut element = MlxArray::empty();
        // SAFETY: The input vector is live for this trampoline's duration and
        // the output pointer is valid writable storage.
        let status = unsafe { raw::mlx_vector_array_get(element.raw_mut(), input, element_index) };
        check_status(status, "read a closure input array")?;
        elements.push(element);
    }
    Ok(elements)
}

/// Writes owned output arrays into the caller's output vector handle.
///
/// # Errors
/// Returns the captured MLX-C description when the write fails.
pub(crate) fn write_output_arrays(
    output: *mut raw::mlx_vector_array,
    outputs: Vec<MlxArray>,
) -> Result<(), MlxCError> {
    let raw_outputs: Vec<raw::mlx_array> = outputs.iter().map(MlxArray::raw).collect();
    // SAFETY: The output pointer is valid writable storage and the handle
    // storage remains valid for this copying call.
    let status =
        unsafe { raw::mlx_vector_array_set_data(output, raw_outputs.as_ptr(), raw_outputs.len()) };
    check_status(status, "write closure output arrays")
}

/// Runs the payload callable, catching panics at the FFI boundary.
pub(crate) fn run_payload_closure<F, R>(
    payload: *mut c_void,
    operation: &'static str,
    invoke: impl FnOnce(&mut F) -> R,
) -> Option<R> {
    let payload_cell = unsafe { &*(payload.cast::<RefCell<F>>()) };
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut callback = payload_cell
            .try_borrow_mut()
            .expect("closure payload is not reentrant");
        invoke(&mut callback)
    }));
    match outcome {
        Ok(value) => Some(value),
        Err(panic_payload) => {
            crate::error::set_closure_error(MlxCError {
                operation,
                description: panic_message(panic_payload),
            });
            None
        }
    }
}

fn panic_message(panic_payload: Box<dyn std::any::Any + Send>) -> String {
    panic_payload
        .downcast_ref::<&str>()
        .map(|message| (*message).to_owned())
        .or_else(|| panic_payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "closure panicked".to_owned())
}

/// Owned MLX closure handle released exactly once through the C API.
#[derive(Debug)]
pub struct MlxClosure(pub(crate) raw::mlx_closure);

impl MlxClosure {
    /// Builds an owned closure from a Rust callable mapping input arrays to
    /// output arrays.
    pub fn from_fn<F>(callback: F) -> Result<Self, MlxCError>
    where
        F: FnMut(&[MlxArray]) -> Result<Vec<MlxArray>, MlxCError> + 'static,
    {
        let operation: &'static str = "create an MLX closure from a Rust callable";
        let payload = Box::into_raw(Box::new(RefCell::new(callback)));
        // SAFETY: The trampoline and destructor follow the exact C ABI, the
        // payload stays alive until the destructor runs, and the returned
        // handle enters RAII ownership.
        let raw_closure = unsafe {
            raw::mlx_closure_new_func_payload(
                Some(vector_callback_trampoline::<F>),
                payload.cast(),
                Some(drop_payload::<F>),
            )
        };
        let closure = Self(raw_closure);
        closure.require_populated(operation)?;
        Ok(closure)
    }

    /// Creates an empty closure handle.
    pub fn empty() -> Self {
        // SAFETY: The returned handle enters RAII ownership immediately.
        let raw_closure = unsafe { raw::mlx_closure_new() };
        Self(raw_closure)
    }

    /// Copies the source closure handle into this handle.
    pub fn set(&mut self, source: &Self) -> Result<(), MlxCError> {
        // SAFETY: Both handles are live and this call only replaces the
        // owned closure value.
        let status = unsafe { raw::mlx_closure_set(&mut self.0, source.0) };
        check_status(status, "copy an MLX closure")
    }

    /// Applies the closure to the input arrays on the caller's behalf.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when application fails.
    pub fn apply(&self, input: &MlxArrayVector) -> Result<MlxArrayVector, MlxCError> {
        let operation: &'static str = "apply an MLX closure";
        let mut output = MlxArrayVector::empty(operation)?;
        // SAFETY: Both handles are live and the output pointer is valid
        // writable storage.
        let status = unsafe { raw::mlx_closure_apply(output.raw_mut(), self.0, input.raw()) };
        check_status(status, operation)?;
        Ok(output)
    }

    pub(crate) const fn raw(&self) -> raw::mlx_closure {
        self.0
    }

    pub(crate) fn raw_mut(&mut self) -> *mut raw::mlx_closure {
        &mut self.0
    }

    pub(crate) fn require_populated(&self, operation: &'static str) -> Result<(), MlxCError> {
        if self.0.ctx.is_null() {
            return Err(MlxCError {
                operation,
                description: "MLX returned an empty closure handle".to_owned(),
            });
        }
        Ok(())
    }
}

impl Drop for MlxClosure {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live closure exactly once and
        // never accesses the handle afterward.
        unsafe {
            raw::mlx_closure_free(self.0);
        }
    }
}

unsafe extern "C" fn vector_callback_trampoline<F>(
    output: *mut raw::mlx_vector_array,
    input: raw::mlx_vector_array,
    payload: *mut c_void,
) -> i32
where
    F: FnMut(&[MlxArray]) -> Result<Vec<MlxArray>, MlxCError> + 'static,
{
    let Some(outputs) = run_payload_closure::<F, Result<Vec<MlxArray>, MlxCError>>(
        payload,
        "apply a Rust closure",
        |callback| owned_input_arrays(input).and_then(|inputs| callback(&inputs)),
    ) else {
        return 1;
    };
    match outputs.and_then(|owned_outputs| write_output_arrays(output, owned_outputs)) {
        Ok(()) => 0,
        Err(failure) => {
            crate::error::set_closure_error(failure);
            1
        }
    }
}

pub(crate) unsafe extern "C" fn drop_payload<F>(payload: *mut c_void) {
    if payload.is_null() {
        return;
    }
    // SAFETY: The payload was created by `Box::into_raw` of the matching
    // type and the destructor runs exactly once per closure.
    drop(unsafe { Box::from_raw(payload.cast::<RefCell<F>>()) });
}

/// Owned MLX closure with keyword arguments, released exactly once.
#[derive(Debug)]
pub struct MlxClosureKwargs(raw::mlx_closure_kwargs);

impl MlxClosureKwargs {
    /// Builds an owned kwargs closure from a Rust callable mapping input
    /// arrays and a keyword map to output arrays.
    pub fn from_fn<F>(callback: F) -> Result<Self, MlxCError>
    where
        F: FnMut(&[MlxArray], &MlxMapStringToArray) -> Result<Vec<MlxArray>, MlxCError> + 'static,
    {
        let operation: &'static str = "create an MLX kwargs closure from a Rust callable";
        let payload = Box::into_raw(Box::new(RefCell::new(callback)));
        // SAFETY: The trampoline and destructor follow the exact C ABI and
        // the returned handle enters RAII ownership.
        let raw_closure = unsafe {
            raw::mlx_closure_kwargs_new_func_payload(
                Some(kwargs_callback_trampoline::<F>),
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

    /// Creates an empty kwargs closure handle.
    pub fn empty() -> Self {
        // SAFETY: The returned handle enters RAII ownership immediately.
        let raw_closure = unsafe { raw::mlx_closure_kwargs_new() };
        Self(raw_closure)
    }

    /// Copies the source closure handle into this handle.
    pub fn set(&mut self, source: &Self) -> Result<(), MlxCError> {
        // SAFETY: Both handles are live and this call only replaces the
        // owned closure value.
        let status = unsafe { raw::mlx_closure_kwargs_set(&mut self.0, source.0) };
        check_status(status, "copy an MLX kwargs closure")
    }

    /// Applies the closure to the inputs and keyword map.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when application fails.
    pub fn apply(
        &self,
        input: &MlxArrayVector,
        keyword_arguments: &MlxMapStringToArray,
    ) -> Result<MlxArrayVector, MlxCError> {
        let operation: &'static str = "apply an MLX kwargs closure";
        let mut output = MlxArrayVector::empty(operation)?;
        // SAFETY: All handles are live and the output pointer is valid
        // writable storage.
        let status = unsafe {
            raw::mlx_closure_kwargs_apply(
                output.raw_mut(),
                self.0,
                input.raw(),
                keyword_arguments.raw(),
            )
        };
        check_status(status, operation)?;
        Ok(output)
    }
}

impl MlxClosureKwargs {
    pub(crate) const fn raw(&self) -> raw::mlx_closure_kwargs {
        self.0
    }
}

impl Drop for MlxClosureKwargs {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live closure exactly once and
        // never accesses the handle afterward.
        unsafe {
            raw::mlx_closure_kwargs_free(self.0);
        }
    }
}

unsafe extern "C" fn kwargs_callback_trampoline<F>(
    output: *mut raw::mlx_vector_array,
    input: raw::mlx_vector_array,
    keyword_arguments: raw::mlx_map_string_to_array,
    payload: *mut c_void,
) -> i32
where
    F: FnMut(&[MlxArray], &MlxMapStringToArray) -> Result<Vec<MlxArray>, MlxCError> + 'static,
{
    let Some(outputs) = run_payload_closure::<F, Result<Vec<MlxArray>, MlxCError>>(
        payload,
        "apply a Rust kwargs closure",
        |callback| {
            owned_input_arrays(input).and_then(|inputs| {
                let borrowed_keywords = MlxMapStringToArray::from_borrowed_raw(keyword_arguments);
                callback(&inputs, &borrowed_keywords)
            })
        },
    ) else {
        return 1;
    };
    match outputs.and_then(|owned_outputs| write_output_arrays(output, owned_outputs)) {
        Ok(()) => 0,
        Err(failure) => {
            crate::error::set_closure_error(failure);
            1
        }
    }
}

/// Owned MLX value-and-gradient closure, released exactly once.
#[derive(Debug)]
pub struct MlxClosureValueAndGrad(raw::mlx_closure_value_and_grad);

impl MlxClosureValueAndGrad {
    /// Builds an owned value-and-gradient closure from a Rust callable
    /// returning both outputs and gradients.
    pub fn from_fn<F>(callback: F) -> Result<Self, MlxCError>
    where
        F: FnMut(&[MlxArray]) -> Result<(Vec<MlxArray>, Vec<MlxArray>), MlxCError> + 'static,
    {
        let operation: &'static str = "create an MLX value-and-gradient closure";
        let payload = Box::into_raw(Box::new(RefCell::new(callback)));
        // SAFETY: The trampoline and destructor follow the exact C ABI and
        // the returned handle enters RAII ownership.
        let raw_closure = unsafe {
            raw::mlx_closure_value_and_grad_new_func_payload(
                Some(value_and_grad_callback_trampoline::<F>),
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

    /// Creates an empty value-and-gradient closure handle.
    pub fn empty() -> Self {
        // SAFETY: The returned handle enters RAII ownership immediately.
        let raw_closure = unsafe { raw::mlx_closure_value_and_grad_new() };
        Self(raw_closure)
    }

    /// Copies the source closure handle into this handle.
    pub fn set(&mut self, source: &Self) -> Result<(), MlxCError> {
        // SAFETY: Both handles are live and this call only replaces the
        // owned closure value.
        let status = unsafe { raw::mlx_closure_value_and_grad_set(&mut self.0, source.0) };
        check_status(status, "copy an MLX value-and-gradient closure")
    }

    /// Applies the closure, producing both the outputs and the gradients.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when application fails.
    pub fn apply(
        &self,
        input: &MlxArrayVector,
    ) -> Result<(MlxArrayVector, MlxArrayVector), MlxCError> {
        let operation: &'static str = "apply an MLX value-and-gradient closure";
        let mut outputs = MlxArrayVector::empty(operation)?;
        let mut gradients = MlxArrayVector::empty(operation)?;
        // SAFETY: All handles are live and both output pointers are valid
        // writable storage.
        let status = unsafe {
            raw::mlx_closure_value_and_grad_apply(
                outputs.raw_mut(),
                gradients.raw_mut(),
                self.0,
                input.raw(),
            )
        };
        check_status(status, operation)?;
        Ok((outputs, gradients))
    }
}

impl MlxClosureValueAndGrad {
    pub(crate) fn raw_mut(&mut self) -> *mut raw::mlx_closure_value_and_grad {
        &mut self.0
    }
}

impl Drop for MlxClosureValueAndGrad {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live closure exactly once and
        // never accesses the handle afterward.
        unsafe {
            raw::mlx_closure_value_and_grad_free(self.0);
        }
    }
}

unsafe extern "C" fn value_and_grad_callback_trampoline<F>(
    outputs: *mut raw::mlx_vector_array,
    gradients: *mut raw::mlx_vector_array,
    input: raw::mlx_vector_array,
    payload: *mut c_void,
) -> i32
where
    F: FnMut(&[MlxArray]) -> Result<(Vec<MlxArray>, Vec<MlxArray>), MlxCError> + 'static,
{
    let Some(result) = run_payload_closure::<F, Result<(Vec<MlxArray>, Vec<MlxArray>), MlxCError>>(
        payload,
        "apply a Rust value-and-gradient closure",
        |callback| owned_input_arrays(input).and_then(|inputs| callback(&inputs)),
    ) else {
        return 1;
    };
    let outcome = result.and_then(|(owned_outputs, owned_gradients)| {
        write_output_arrays(outputs, owned_outputs)?;
        write_output_arrays(gradients, owned_gradients)
    });
    match outcome {
        Ok(()) => 0,
        Err(failure) => {
            crate::error::set_closure_error(failure);
            1
        }
    }
}
