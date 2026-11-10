//! Passing data between MLX and a loaded program: the activation reorder on
//! the GPU, the host copy into the input surface, and the output surface back
//! into MLX.
//!
//! This is the crate's only MLX-facing code, and it holds the raw MLX calls so
//! that serving, which denies unsafe code, can stay safe.

use std::ffi::c_void;

use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError, MlxDtype, raw};
use thiserror::Error;

use crate::core_ml::{CoreMlProgram, CoreMlProgramError, LockedOutputSurface};

#[derive(Debug, Error)]
pub enum MlxHandOffError {
    #[error(transparent)]
    Mlx(#[from] MlxCError),
    #[error(transparent)]
    Program(#[from] CoreMlProgramError),
    #[error("the prepared activations are not row-contiguous fp16")]
    PreparedActivationsNotContiguous,
    #[error("the output surface pads its rows, so it cannot be read as one array")]
    PaddedOutputSurface,
    #[error("MLX did not accept the output surface as an array")]
    SurfaceArrayRejected,
}

impl MlxHandOffError {
    /// Whether this came from MLX rather than the program. An allocation can fail near
    /// the memory ceiling with every program healthy.
    #[must_use]
    pub const fn is_from_mlx(&self) -> bool {
        matches!(self, Self::Mlx(_))
    }
}

/// How the output surface reaches MLX; the journeys compare the two.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputHandOff {
    ZeroCopySurface,
    CopiedSurface,
}

/// Which axis of `[1, tokens, channels]` the two engines divided.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitAxis {
    Tokens,
    Channels,
}

impl SplitAxis {
    const fn merged_axis(self) -> i32 {
        match self {
            Self::Tokens => 1,
            Self::Channels => 2,
        }
    }
}

/// `[1, tokens, channels]` becomes lazy row-contiguous `[channels, tokens]`
/// fp16, the input surface's layout.
///
/// A transpose alone only changes strides, and neither `reshape` nor a cast
/// rewrites strided storage, so a host copy would read tokens in the wrong
/// order. Concatenating two halves is what forces MLX to allocate and fill
/// one row-contiguous buffer, on the GPU, where the reorder is cheapest.
pub fn channel_major_fp16(
    context: &MlxBindingsContext,
    token_major_activations: &MlxArray,
) -> Result<MlxArray, MlxCError> {
    let token_major_shape = token_major_activations.shape();
    let token_count = token_major_shape[1];
    let channel_count = token_major_shape[2];
    let half_precision = context.astype(token_major_activations, MlxDtype::Float16)?;
    let channels_first = context.transpose_axes(&half_precision, &[2, 0, 1])?;
    let strided = context.reshape(&channels_first, &[channel_count, token_count])?;
    let middle_channel = channel_count / 2;
    let first_half = context.slice(&strided, &[0, 0], &[middle_channel, token_count], &[1, 1])?;
    let second_half = context.slice(
        &strided,
        &[middle_channel, 0],
        &[channel_count, token_count],
        &[1, 1],
    )?;
    context.concatenate_axis(&[&first_half, &second_half], 0)
}

/// Strides are signed: a reversed view reports a negative stride (-1 on the last axis,
/// minus the row length on the first), so neither equality below holds and the view is
/// refused instead of being copied in the wrong order.
fn is_row_contiguous_fp16(prepared: &MlxArray) -> bool {
    let strides = prepared.strides();
    prepared.dtype() == MlxDtype::Float16
        && strides.len() == 2
        && strides[1] == 1
        && strides[0] == i64::from(prepared.shape()[1])
}

/// Copies evaluated [`channel_major_fp16`] activations into the input surface.
pub fn write_prepared_input(
    prepared: &MlxArray,
    program: &mut CoreMlProgram,
) -> Result<(), MlxHandOffError> {
    if !is_row_contiguous_fp16(prepared) {
        return Err(MlxHandOffError::PreparedActivationsNotContiguous);
    }
    // SAFETY: the check above and the caller's evaluation mean the storage holds
    // `element_count` fp16 values, alive and unchanged for this borrow.
    let fp16_activations = unsafe {
        std::slice::from_raw_parts(
            raw::mlx_array_data_float16(prepared.raw()).cast::<u16>(),
            prepared.element_count(),
        )
    };
    program.write_activations(fp16_activations)?;
    Ok(())
}

/// The surface outlives the array, so MLX has nothing to free here.
unsafe extern "C" fn keep_surface_owned_by_program(_unused_payload: *mut c_void) {}

/// The locked output surface as a `surface_shape` fp16 MLX array.
///
/// # Safety
/// For [`OutputHandOff::ZeroCopySurface`] the array borrows the surface, so it
/// must be evaluated into other storage or dropped before `surface` drops, and
/// `surface_shape` must match the surface's `[channels, tokens]`.
pub unsafe fn surface_as_array(
    surface: &LockedOutputSurface<'_>,
    surface_shape: [i32; 2],
    output_hand_off: OutputHandOff,
) -> Result<MlxArray, MlxHandOffError> {
    let fp16_values = surface
        .contiguous_fp16_values()
        .ok_or(MlxHandOffError::PaddedOutputSurface)?;
    // SAFETY: the guard keeps the surface mapped, the caller vouches for the
    // shape, and the new handle moves into an owner below. MLX falls back to
    // copying if Metal refuses the mapping.
    unsafe {
        let new_handle = match output_hand_off {
            OutputHandOff::ZeroCopySurface => raw::mlx_array_new_data_managed_payload(
                fp16_values.cast_mut().cast(),
                surface_shape.as_ptr(),
                2,
                MlxDtype::Float16.to_raw(),
                std::ptr::null_mut(),
                Some(keep_surface_owned_by_program),
            ),
            OutputHandOff::CopiedSurface => raw::mlx_array_new_data(
                fp16_values.cast(),
                surface_shape.as_ptr(),
                2,
                MlxDtype::Float16.to_raw(),
            ),
        };
        let mut owned = MlxArray::from_bool(false);
        let set_status = raw::mlx_array_set(owned.raw_mut(), new_handle);
        raw::mlx_array_free(new_handle);
        if set_status != 0 {
            return Err(MlxHandOffError::SurfaceArrayRejected);
        }
        Ok(owned)
    }
}

/// The program's whole output as an evaluated `[1, tokens, channels]` array in
/// `result_dtype`. Evaluating into a copy is what frees the surface for the next
/// prediction, because a transpose, a reshape and a same-type cast only borrow it.
pub fn surface_as_token_major(
    context: &MlxBindingsContext,
    program: &CoreMlProgram,
    result_dtype: MlxDtype,
    output_hand_off: OutputHandOff,
) -> Result<MlxArray, MlxHandOffError> {
    let token_count = program.token_count() as i32;
    let channel_count = program.output_channel_count() as i32;
    let surface = program.lock_output_surface()?;
    // SAFETY: the shape is the program's own, and `owned` is evaluated into its
    // own storage below while `surface` is still alive; the borrowing array and the
    // lazy graph above it drop before the guard does.
    let surface_array =
        unsafe { surface_as_array(&surface, [channel_count, token_count], output_hand_off)? };
    let tokens_first = context.transpose_axes(&surface_array, &[1, 0])?;
    let batched = context.reshape(&tokens_first, &[1, token_count, channel_count])?;
    let converted = context.astype(&batched, result_dtype)?;
    let owned = context.copy(&converted)?;
    context.evaluate_arrays(&[&owned])?;
    Ok(owned)
}

/// Joins the program's output before `gpu_slice` along `split_axis`, as an
/// evaluated `[1, tokens, channels]` array in `gpu_slice`'s dtype.
///
/// Evaluating here is what makes the surface free for the next prediction: the
/// concatenation fills its own storage, so the result borrows nothing.
pub fn merge_surface_with_gpu_slice(
    context: &MlxBindingsContext,
    program: &CoreMlProgram,
    gpu_slice: &MlxArray,
    split_axis: SplitAxis,
    output_hand_off: OutputHandOff,
) -> Result<MlxArray, MlxHandOffError> {
    let token_count = program.token_count() as i32;
    let channel_count = program.output_channel_count() as i32;
    let surface = program.lock_output_surface()?;
    // SAFETY: the shape is the program's own, and `merged` is evaluated into
    // its own storage below while `surface` is still alive; the borrowing array
    // and the lazy graph above it drop before the guard does.
    let surface_array =
        unsafe { surface_as_array(&surface, [channel_count, token_count], output_hand_off)? };
    let tokens_first = context.transpose_axes(&surface_array, &[1, 0])?;
    let batched = context.reshape(&tokens_first, &[1, token_count, channel_count])?;
    let converted = context.astype(&batched, gpu_slice.dtype())?;
    let merged = context.concatenate_axis(&[&converted, gpu_slice], split_axis.merged_axis())?;
    context.evaluate_arrays(&[&merged])?;
    Ok(merged)
}
