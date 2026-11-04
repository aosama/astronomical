//! The steps of handing data between MLX and a Core ML projection, shared by
//! the hand-off and token-split journeys: the activation reorder on the GPU,
//! the host copy into the input surface, the output surface as an MLX array,
//! and the stage timings every hand-off call reports.

use std::ffi::c_void;
use std::time::Duration;

use astronomical_neural_engine::core_ml_projection::{CoreMlProjection, LockedOutputSurface};
use astronomical_neural_engine::timing_summary::TimingSummary;
use astronomical_runtime_integration::MlxRuntime;
use mlx_c_rust::{MlxArray, MlxDtype, raw};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum OutputHandOff {
    ZeroCopySurface,
    CopiedSurface,
}

pub(super) struct StageDurations {
    pub(super) input_preparation: Duration,
    pub(super) surface_write: Duration,
    pub(super) engines_after_write: Duration,
    pub(super) merge: Duration,
    pub(super) total: Duration,
}

pub(super) struct StageSummaries {
    pub(super) input_preparation: TimingSummary,
    pub(super) surface_write: TimingSummary,
    pub(super) engines_after_write: TimingSummary,
    pub(super) merge: TimingSummary,
    pub(super) total: TimingSummary,
}

/// `[1, tokens, channels]` bf16 becomes row-contiguous `[channels, tokens]`
/// fp16, the input surface's layout.
///
/// A transpose alone only changes strides, and neither `reshape` nor a cast
/// rewrites strided storage, so a host copy would read tokens in the wrong
/// order. Concatenating two halves is what forces MLX to allocate and fill
/// one row-contiguous buffer, on the GPU, where the reorder is cheapest.
pub(super) fn channel_major_fp16(runtime: &MlxRuntime, token_major_bf16: &MlxArray) -> MlxArray {
    let token_major_shape = token_major_bf16.shape();
    let token_count = token_major_shape[1];
    let channel_count = token_major_shape[2];
    let half_precision = runtime
        .astype(token_major_bf16, MlxDtype::Float16)
        .expect("activations convert");
    let channels_first = runtime
        .transpose_axes(&half_precision, &[2, 0, 1])
        .expect("activations transpose");
    let strided = runtime
        .reshape(&channels_first, &[channel_count, token_count])
        .expect("activations reshape");
    let middle_channel = channel_count / 2;
    let first_half = runtime
        .slice(&strided, &[0, 0], &[middle_channel, token_count], &[1, 1])
        .expect("first half slices");
    let second_half = runtime
        .slice(
            &strided,
            &[middle_channel, 0],
            &[channel_count, token_count],
            &[1, 1],
        )
        .expect("second half slices");
    runtime
        .concatenate_axis(&[&first_half, &second_half], 0)
        .expect("halves concatenate")
}

fn assert_row_contiguous(evaluated: &MlxArray) {
    let strides = evaluated.strides();
    assert_eq!(
        (strides[0], strides[1]),
        (evaluated.shape()[1] as usize, 1),
        "the prepared activations must be row-contiguous for the host copy"
    );
}

pub(super) fn write_prepared_input(prepared: &MlxArray, projection: &mut CoreMlProjection) {
    assert_row_contiguous(prepared);
    // SAFETY: `prepared` is evaluated, fp16, and row-contiguous (checked by
    // `assert_row_contiguous`), so its storage holds `element_count` values
    // and stays alive and unchanged for this borrow.
    let fp16_activations = unsafe {
        std::slice::from_raw_parts(
            raw::mlx_array_data_float16(prepared.raw()).cast::<u16>(),
            prepared.element_count(),
        )
    };
    projection
        .write_activations(fp16_activations)
        .expect("the activations fit");
}

/// The surface outlives the array: the caller holds the lock until the merge
/// has been evaluated into its own storage, so MLX has nothing to free here.
unsafe extern "C" fn keep_surface_owned_by_projection(_unused_payload: *mut c_void) {}

pub(super) fn surface_as_array(
    surface: &LockedOutputSurface<'_>,
    surface_shape: [i32; 2],
    output_hand_off: OutputHandOff,
) -> MlxArray {
    let fp16_values = surface
        .contiguous_fp16_values()
        .expect("a 2,048-token surface has no row padding");
    // SAFETY: the guard keeps the surface mapped for reading, the shape
    // matches the surface's element count, and the new handle moves into an
    // owner below. MLX falls back to copying if Metal refuses the mapping.
    unsafe {
        let new_handle = match output_hand_off {
            OutputHandOff::ZeroCopySurface => raw::mlx_array_new_data_managed_payload(
                fp16_values.cast_mut().cast(),
                surface_shape.as_ptr(),
                2,
                MlxDtype::Float16.to_raw(),
                std::ptr::null_mut(),
                Some(keep_surface_owned_by_projection),
            ),
            OutputHandOff::CopiedSurface => raw::mlx_array_new_data(
                fp16_values.cast(),
                surface_shape.as_ptr(),
                2,
                MlxDtype::Float16.to_raw(),
            ),
        };
        let mut owned = MlxArray::from_bool(false);
        assert_eq!(raw::mlx_array_set(owned.raw_mut(), new_handle), 0);
        raw::mlx_array_free(new_handle);
        owned
    }
}

pub(super) fn summarize_stages(stage_samples: &[StageDurations]) -> StageSummaries {
    let mut input_preparation = Vec::with_capacity(stage_samples.len());
    let mut surface_write = Vec::with_capacity(stage_samples.len());
    let mut engines_after_write = Vec::with_capacity(stage_samples.len());
    let mut merge = Vec::with_capacity(stage_samples.len());
    let mut total = Vec::with_capacity(stage_samples.len());
    for stages in stage_samples {
        input_preparation.push(stages.input_preparation);
        surface_write.push(stages.surface_write);
        engines_after_write.push(stages.engines_after_write);
        merge.push(stages.merge);
        total.push(stages.total);
    }
    StageSummaries {
        input_preparation: TimingSummary::from_samples(&input_preparation).expect("timed"),
        surface_write: TimingSummary::from_samples(&surface_write).expect("timed"),
        engines_after_write: TimingSummary::from_samples(&engines_after_write).expect("timed"),
        merge: TimingSummary::from_samples(&merge).expect("timed"),
        total: TimingSummary::from_samples(&total).expect("timed"),
    }
}
