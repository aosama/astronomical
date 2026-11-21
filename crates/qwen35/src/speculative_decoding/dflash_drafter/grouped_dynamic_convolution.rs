//! DFlash2's grouped dynamic causal convolution around each sublayer.
//!
//! Every block row mixes itself with the rows just before it, inside the
//! drafted block only (the anchor row included), using a fixed per-channel
//! kernel plus a per-group kernel the row itself predicts. One prediction
//! serves both stages: its first half shapes the input of the attention or MLP
//! sublayer, its second half the sublayer's output. The model card credits
//! these convolutions with keeping late block positions from decaying.

use mlx_bridge::MlxArray;
use runtime_integration::{MlxRuntime, MlxRuntimeError};

use super::drafter_configuration::GroupedConvolutionShape;
use super::drafter_weights::GroupedConvolutionWeights;

/// The convolved sublayer input, and the predicted kernel kept for the
/// sublayer's output.
pub struct ConvolvedSublayerInput {
    pub convolved_rows: MlxArray,
    pub finishing_kernel: MlxArray,
}

/// Stage one: predicts both stages' kernels from `normalized_rows` [1, L, H]
/// and convolves those rows with the first.
pub fn convolve_sublayer_input(
    runtime: &MlxRuntime,
    weights: &GroupedConvolutionWeights,
    normalized_rows: &MlxArray,
) -> Result<ConvolvedSublayerInput, MlxRuntimeError> {
    let row_count = normalized_rows.shape()[1];
    let kernel_size = weights.shape.kernel_size;
    let group_count = weights.shape.group_count();
    let predicted_kernels = runtime.reshape(
        &weights
            .kernel_projection
            .forward(runtime, normalized_rows)?,
        &[1, row_count, 2, kernel_size, group_count],
    )?;
    let starting_kernel = stage_slice(runtime, &predicted_kernels, 0)?;
    let finishing_kernel = stage_slice(runtime, &predicted_kernels, 1)?;
    let convolved_rows = convolve(
        runtime,
        normalized_rows,
        &starting_kernel,
        &weights.base_kernel,
        0,
        &weights.shape,
    )?;
    Ok(ConvolvedSublayerInput {
        convolved_rows,
        finishing_kernel,
    })
}

/// Stage two: convolves the sublayer's output with the kept kernel.
pub fn convolve_sublayer_output(
    runtime: &MlxRuntime,
    weights: &GroupedConvolutionWeights,
    sublayer_output: &MlxArray,
    finishing_kernel: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    convolve(
        runtime,
        sublayer_output,
        finishing_kernel,
        &weights.base_kernel,
        1,
        &weights.shape,
    )
}

/// `[1, L, 2, kernel, groups]` → one stage's `[1, L, kernel, groups]`.
fn stage_slice(
    runtime: &MlxRuntime,
    predicted_kernels: &MlxArray,
    stage: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    let shape = predicted_kernels.shape();
    let staged = runtime.slice(
        predicted_kernels,
        &[0, 0, stage, 0, 0],
        &[shape[0], shape[1], stage + 1, shape[3], shape[4]],
        &[1, 1, 1, 1, 1],
    )?;
    Ok(runtime.reshape(&staged, &[shape[0], shape[1], shape[3], shape[4]])?)
}

/// `output[t] = sum over taps d of (base[d] + predicted[t, d]) * rows[t - d]`,
/// per channel group, with rows before the block taken as zero. The fixed and
/// predicted products are added separately, as the reference does.
fn convolve(
    runtime: &MlxRuntime,
    rows: &MlxArray,
    predicted_kernel: &MlxArray,
    base_kernel: &MlxArray,
    stage: i32,
    shape: &GroupedConvolutionShape,
) -> Result<MlxArray, MlxRuntimeError> {
    let row_count = rows.shape()[1];
    let group_count = shape.group_count();
    let group_size = shape.group_size;
    let grouped_rows = runtime.reshape(rows, &[1, row_count, group_count, group_size])?;
    let mut convolved: Option<MlxArray> = None;
    for tap in 0..shape.kernel_size {
        let shifted_rows = shifted_down(runtime, &grouped_rows, tap)?;
        let tap_kernels = TapKernels::for_tap(
            runtime,
            shape,
            base_kernel,
            predicted_kernel,
            stage,
            tap,
            rows,
        )?;
        let fixed_product = runtime.multiply(&tap_kernels.fixed, &shifted_rows)?;
        let predicted_product = runtime.multiply(&tap_kernels.predicted, &shifted_rows)?;
        let running = match convolved {
            Some(previous) => runtime.add(&previous, &fixed_product)?,
            None => fixed_product,
        };
        convolved = Some(runtime.add(&running, &predicted_product)?);
    }
    let convolved = convolved.ok_or(MlxRuntimeError::RuntimeOperation {
        operation: "convolve DFlash2 block rows",
        description: "the convolution kernel has no taps".to_owned(),
    })?;
    Ok(runtime.reshape(&convolved, &[1, row_count, shape.channel_count])?)
}

/// One tap's two kernels, shaped to broadcast over `[1, rows, groups, group size]`.
struct TapKernels {
    /// `[1, 1, groups, group size]`, the checkpoint's fixed per-channel weight.
    fixed: MlxArray,
    /// `[1, rows, groups, 1]`, the weight each row predicted for its group.
    predicted: MlxArray,
}

impl TapKernels {
    fn for_tap(
        runtime: &MlxRuntime,
        shape: &GroupedConvolutionShape,
        base_kernel: &MlxArray,
        predicted_kernel: &MlxArray,
        stage: i32,
        tap: i32,
        rows: &MlxArray,
    ) -> Result<Self, MlxRuntimeError> {
        let row_count = rows.shape()[1];
        let group_count = shape.group_count();
        let fixed_slice = runtime.slice(
            base_kernel,
            &[stage, tap, 0],
            &[stage + 1, tap + 1, shape.channel_count],
            &[1, 1, 1],
        )?;
        let fixed = runtime.astype(
            &runtime.reshape(&fixed_slice, &[1, 1, group_count, shape.group_size])?,
            rows.dtype(),
        )?;
        let predicted_slice = runtime.slice(
            predicted_kernel,
            &[0, 0, tap, 0],
            &[1, row_count, tap + 1, group_count],
            &[1, 1, 1, 1],
        )?;
        let predicted = runtime.reshape(&predicted_slice, &[1, row_count, group_count, 1])?;
        Ok(Self { fixed, predicted })
    }
}

/// The block rows moved `tap` rows later, zeros entering at the top.
fn shifted_down(
    runtime: &MlxRuntime,
    grouped_rows: &MlxArray,
    tap: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    if tap == 0 {
        return Ok(grouped_rows.retain()?);
    }
    let shape = grouped_rows.shape();
    let kept_rows = runtime.slice(
        grouped_rows,
        &[0, 0, 0, 0],
        &[1, shape[1] - tap, shape[2], shape[3]],
        &[1, 1, 1, 1],
    )?;
    let leading_zeros = runtime.zeros(&[1, tap, shape[2], shape[3]], grouped_rows.dtype())?;
    Ok(runtime.concatenate_axis(&[&leading_zeros, &kept_rows], 1)?)
}
