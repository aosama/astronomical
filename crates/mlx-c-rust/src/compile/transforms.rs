//! The MLX compile and transform domain: graph compilation, the compile
//! cache, and the checkpoint/custom/JVP/VJP/value-and-gradient transforms.

use crate::error::check_status;
use crate::raw;
use crate::types::closure_custom::{MlxClosureCustom, MlxClosureCustomJvp, MlxClosureCustomVmap};
use crate::types::closures::{MlxClosure, MlxClosureValueAndGrad};
use crate::types::vectors::MlxVectorInt;
use crate::{MlxArray, MlxArrayVector, MlxCError};

impl MlxClosure {
    /// Wraps the closure so every call is compiled and cached by MLX.
    pub fn compiled(&self, shapeless: bool) -> Result<Self, MlxCError> {
        let operation: &'static str = "compile an MLX closure";
        let mut output = Self::empty();
        // SAFETY: Both handles are live and the output pointer is valid
        // writable storage.
        let status = unsafe { raw::mlx_compile(output.raw_mut(), self.0, shapeless) };
        check_status(status, operation)?;
        output.require_populated(operation)?;
        Ok(output)
    }

    /// Wraps the closure so calls stop at the traced boundary, matching
    /// MLX's checkpoint transform.
    pub fn checkpointed(&self) -> Result<Self, MlxCError> {
        let operation: &'static str = "checkpoint an MLX closure";
        let mut output = Self::empty();
        // SAFETY: Both handles are live and the output pointer is valid
        // writable storage.
        let status = unsafe { raw::mlx_checkpoint(output.raw_mut(), self.0) };
        check_status(status, operation)?;
        output.require_populated(operation)?;
        Ok(output)
    }

    /// Wraps the closure with a custom vector-Jacobian product.
    pub fn with_custom_vjp(&self, backward: &MlxClosureCustom) -> Result<Self, MlxCError> {
        let operation: &'static str = "attach a custom VJP to an MLX closure";
        let mut output = Self::empty();
        // SAFETY: All handles are live and the output pointer is valid
        // writable storage.
        let status = unsafe { raw::mlx_custom_vjp(output.raw_mut(), self.0, backward.raw()) };
        check_status(status, operation)?;
        output.require_populated(operation)?;
        Ok(output)
    }

    /// Wraps the closure with custom VJP, JVP, and VMAP rules.
    #[allow(clippy::too_many_arguments)]
    pub fn with_custom_rules(
        &self,
        backward: &MlxClosureCustom,
        forward: &MlxClosureCustomJvp,
        linearize: &MlxClosureCustomVmap,
    ) -> Result<Self, MlxCError> {
        let operation: &'static str = "attach custom rules to an MLX closure";
        let mut output = Self::empty();
        // SAFETY: All handles are live and the output pointer is valid
        // writable storage.
        let status = unsafe {
            raw::mlx_custom_function(
                output.raw_mut(),
                self.0,
                backward.raw(),
                forward.raw(),
                linearize.raw(),
            )
        };
        check_status(status, operation)?;
        output.require_populated(operation)?;
        Ok(output)
    }
}

/// Computes the Jacobian-vector product of a closure.
///
/// # Errors
/// Returns the captured MLX-C description when the transform fails.
pub fn jvp_of_closure(
    function: &MlxClosure,
    primals: &MlxArrayVector,
    tangents: &MlxArrayVector,
) -> Result<(MlxArrayVector, MlxArrayVector), MlxCError> {
    let operation: &'static str = "apply the MLX JVP transform";
    let mut outputs = MlxArrayVector::empty(operation)?;
    let mut tangent_outputs = MlxArrayVector::empty(operation)?;
    // SAFETY: All handles are live and both output pointers are valid
    // writable storage.
    let status = unsafe {
        raw::mlx_jvp(
            outputs.raw_mut(),
            tangent_outputs.raw_mut(),
            function.raw(),
            primals.raw(),
            tangents.raw(),
        )
    };
    check_status(status, operation)?;
    Ok((outputs, tangent_outputs))
}

/// Computes the vector-Jacobian product of a closure.
///
/// # Errors
/// Returns the captured MLX-C description when the transform fails.
pub fn vjp_of_closure(
    function: &MlxClosure,
    primals: &MlxArrayVector,
    cotangents: &MlxArrayVector,
) -> Result<(MlxArrayVector, MlxArrayVector), MlxCError> {
    let operation: &'static str = "apply the MLX VJP transform";
    let mut outputs = MlxArrayVector::empty(operation)?;
    let mut cotangent_outputs = MlxArrayVector::empty(operation)?;
    // SAFETY: All handles are live and both output pointers are valid
    // writable storage.
    let status = unsafe {
        raw::mlx_vjp(
            outputs.raw_mut(),
            cotangent_outputs.raw_mut(),
            function.raw(),
            primals.raw(),
            cotangents.raw(),
        )
    };
    check_status(status, operation)?;
    Ok((outputs, cotangent_outputs))
}

/// Builds the value-and-gradient closure for the given closure.
///
/// # Errors
/// Returns the captured MLX-C description when the transform fails.
pub fn value_and_grad_of_closure(
    function: &MlxClosure,
    argument_indices: &[i32],
) -> Result<MlxClosureValueAndGrad, MlxCError> {
    let operation: &'static str = "build the MLX value-and-gradient transform";
    let mut output = MlxClosureValueAndGrad::empty();
    // SAFETY: The closure handle is live, the slice remains valid for the
    // call, and the output pointer is valid writable storage.
    let status = unsafe {
        raw::mlx_value_and_grad(
            output.raw_mut(),
            function.raw(),
            argument_indices.as_ptr(),
            argument_indices.len(),
        )
    };
    check_status(status, operation)?;
    Ok(output)
}

/// Owned MLX compile cache handle released exactly once.
#[derive(Debug)]
pub struct MlxCompileCache(raw::mlx_compile_cache);

impl MlxCompileCache {
    /// Creates an empty compile cache handle.
    pub fn empty() -> Self {
        let mut raw_cache = raw::mlx_compile_cache {
            ctx: std::ptr::null_mut(),
        };
        // SAFETY: The output pointer is valid writable storage.
        let status = unsafe { raw::mlx_detail_compile_cache(&mut raw_cache) };
        debug_assert_eq!(status, 0);
        Self(raw_cache)
    }

    /// Erases every compiled entry in this cache.
    pub fn clear(&mut self) -> Result<(), MlxCError> {
        // SAFETY: The cache handle is live.
        let status = unsafe { raw::mlx_detail_compile_clear_cache(self.0) };
        check_status(status, "clear an MLX compile cache")
    }

    /// Erases the compiled entry registered under `function_identity`.
    pub fn erase(&mut self, function_identity: usize) -> Result<(), MlxCError> {
        // SAFETY: The cache handle is live and the identity is a plain value.
        let status = unsafe { raw::mlx_detail_compile_erase(self.0, function_identity) };
        check_status(status, "erase an MLX compile cache entry")
    }
}

impl Drop for MlxCompileCache {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live cache exactly once and never
        // accesses the handle afterward.
        unsafe {
            raw::mlx_compile_cache_free(self.0);
        }
    }
}

impl MlxClosure {
    /// Compiles the closure with explicit tracing details into the process
    /// compile cache.
    ///
    /// The `constants` array is the identity MLX uses for compile-time
    /// constant folding.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when compilation fails.
    pub fn compile_detailed(
        &self,
        function_identity: usize,
        shapeless: bool,
        constants: &[u64],
    ) -> Result<Self, MlxCError> {
        let operation: &'static str = "compile an MLX closure with tracing details";
        let mut output = Self::empty();
        // SAFETY: The closure handle is live, the slice remains valid for the
        // call, and the output pointer is valid writable storage.
        let status = unsafe {
            raw::mlx_detail_compile(
                output.raw_mut(),
                self.0,
                function_identity,
                shapeless,
                constants.as_ptr(),
                constants.len(),
            )
        };
        check_status(status, operation)?;
        output.require_populated(operation)?;
        Ok(output)
    }
}

/// Disables MLX's compilation transform globally.
pub fn disable_compile() -> Result<(), MlxCError> {
    // SAFETY: Disabling takes no inputs and reports status through the
    // captured-error machinery.
    let status = unsafe { raw::mlx_disable_compile() };
    check_status(status, "disable MLX compilation")
}

/// Re-enables MLX's compilation transform globally.
pub fn enable_compile() -> Result<(), MlxCError> {
    // SAFETY: Enabling takes no inputs and reports status through the
    // captured-error machinery.
    let status = unsafe { raw::mlx_enable_compile() };
    check_status(status, "enable MLX compilation")
}

/// Selects the global compilation mode.
pub fn set_compile_mode(mode: raw::mlx_compile_mode) -> Result<(), MlxCError> {
    // SAFETY: The mode is a plain value and reports status through the
    // captured-error machinery.
    let status = unsafe { raw::mlx_set_compile_mode(mode) };
    check_status(status, "set the MLX compile mode")
}

/// Replaces one traced array in a vector-of-arrays graph, the building block
/// of the VMAP transform.
///
/// # Errors
/// Returns the captured MLX-C description when the replacement fails.
pub fn vmap_replace(
    inputs: &MlxArrayVector,
    traced_inputs: &MlxArrayVector,
    traced_outputs: &MlxArrayVector,
    input_axes: &[i32],
    output_axes: &[i32],
) -> Result<MlxArrayVector, MlxCError> {
    let operation: &'static str = "apply the MLX VMAP replacement";
    let mut output = MlxArrayVector::empty(operation)?;
    // SAFETY: All handles are live, the slices remain valid for the call,
    // and the output pointer is valid writable storage.
    let status = unsafe {
        raw::mlx_detail_vmap_replace(
            output.raw_mut(),
            inputs.raw(),
            traced_inputs.raw(),
            traced_outputs.raw(),
            input_axes.as_ptr(),
            input_axes.len(),
            output_axes.as_ptr(),
            output_axes.len(),
        )
    };
    check_status(status, operation)?;
    Ok(output)
}

/// Traces a closure for the VMAP transform, producing traced inputs and
/// outputs plus the resolved axes.
///
/// # Errors
/// Returns the captured MLX-C description when tracing fails.
pub fn vmap_trace(
    function: &MlxClosure,
    inputs: &MlxArrayVector,
    input_axes: &[i32],
) -> Result<(MlxArrayVector, MlxArrayVector, MlxVectorInt), MlxCError> {
    let operation: &'static str = "apply the MLX VMAP trace";
    let mut traced_inputs = MlxArrayVector::empty(operation)?;
    let mut traced_outputs = MlxArrayVector::empty(operation)?;
    let output_axes = MlxVectorInt::empty();
    // SAFETY: All handles are live, the slice remains valid for the call,
    // and all output pointers are valid writable storage.
    let status = unsafe {
        raw::mlx_detail_vmap_trace(
            traced_inputs.raw_mut(),
            traced_outputs.raw_mut(),
            function.raw(),
            inputs.raw(),
            input_axes.as_ptr(),
            input_axes.len(),
        )
    };
    check_status(status, operation)?;
    Ok((traced_inputs, traced_outputs, output_axes))
}

impl MlxArray {
    /// Schedules this array's evaluation after the dependencies and returns
    /// the evaluated handle, matching MLX-C `mlx_depends`.
    ///
    /// # Errors
    /// Returns the captured MLX-C description when scheduling fails.
    pub fn evaluate_after(
        &self,
        dependencies: &MlxArrayVector,
    ) -> Result<MlxArrayVector, MlxCError> {
        let operation: &'static str = "evaluate an MLX array after its dependencies";
        let outputs = MlxArrayVector::from_array(self)?;
        let mut result = MlxArrayVector::empty(operation)?;
        // SAFETY: All handles are live and the output pointer is valid
        // writable storage.
        let status =
            unsafe { raw::mlx_depends(result.raw_mut(), outputs.raw(), dependencies.raw()) };
        check_status(status, operation)?;
        Ok(result)
    }
}
