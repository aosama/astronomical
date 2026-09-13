//! GPU execution of `qwen4_exp` hyper-connection stream mixing.
//!
//! This owner enacts the pinned algebra from `stream_algebra` on MLX arrays:
//! grouped normalization per stream, the low-rank gated mix into a block
//! input, and the learned per-stream injection on the way out. It adds no
//! arithmetic of its own — every constant and operation order comes from the
//! pinned owner, and the direct-MLX oracle proved this composition reproduces
//! the hermetic contract's exact values before any production code existed.
//!
//! Two conventions matter and both are pinned here rather than re-derived at
//! call sites: the checkpoint stores the norm affine as an additive offset
//! around one, so the GPU path adds one before MLX's fused RMS normalization,
//! which multiplies by the weight directly; and gate arithmetic runs in
//! float32 before the result returns to the activation dtype, so a
//! bfloat16 activation cannot narrow the sigmoid reduction.
//!
//! A custom Metal kernel is deliberately absent: the mixing is two small
//! matmuls and elementwise work that MLX fuses well, and a kernel must be
//! justified by measurement, not by taste. The kernel-capability family
//! gains one when a measured win exists.

use astronomical_runtime_integration::{MlxArray, MlxRuntime, MlxRuntimeError};

use crate::performance_attribution::{PerformanceAttribution, PerformanceOperation};

use super::stream_algebra::StreamMixingPlan;

/// Executes the pinned stream algebra on the GPU.
///
/// All functions take hyper inputs shaped `[token_count, hyper_width]` and
/// return block-shaped outputs `[token_count, stream_width]`; the caller owns
/// batch and token axes.
pub struct HyperConnectionExecutor {
    plan: StreamMixingPlan,
}

impl HyperConnectionExecutor {
    /// Builds an executor from the plan the configuration validated.
    #[must_use]
    pub const fn new(plan: StreamMixingPlan) -> Self {
        Self { plan }
    }

    /// The plan this executor enacts.
    #[must_use]
    pub const fn plan(&self) -> &StreamMixingPlan {
        &self.plan
    }

    fn hyper_width(&self) -> i32 {
        (self.plan.stream_count * self.plan.stream_width) as i32
    }

    fn validate_hyper_shape(
        &self,
        operation: &'static str,
        array: &MlxArray,
    ) -> Result<(), MlxRuntimeError> {
        let hyper = self.hyper_width();
        if array.shape().last() != Some(&hyper) {
            return Err(MlxRuntimeError::RuntimeOperation {
                operation,
                description: format!(
                    "the final axis must hold {hyper} elements, got {:?}",
                    array.shape()
                ),
            });
        }
        Ok(())
    }

    fn shifted_norm_weights(
        &self,
        runtime: &MlxRuntime,
        norm_weights: &MlxArray,
    ) -> Result<MlxArray, MlxRuntimeError> {
        let hyper = self.hyper_width();
        // The checkpoint's affine is additive around one; MLX's fused RMS
        // normalization multiplies by the weight directly, so the reference
        // adds one before the fused operation.
        let ones = runtime.array_from_f32(&vec![1.0_f32; hyper as usize], &[hyper])?;
        runtime.add(norm_weights, &ones)
    }

    /// Grouped RMSNorm over each stream with the checkpoint's additive
    /// affine convention.
    ///
    /// # Errors
    /// When any MLX operation fails or a shape disagrees with the plan.
    pub fn grouped_normalize(
        &self,
        runtime: &MlxRuntime,
        norm_weights: &MlxArray,
        hyper_input: &MlxArray,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, MlxRuntimeError> {
        performance_attribution.measure_operation(
            PerformanceOperation::Qwen4ExpHyperConnectionNormalization,
            |_| {
                self.validate_hyper_shape("hyper-connection normalization", hyper_input)?;
                self.validate_hyper_shape("hyper-connection normalization", norm_weights)?;
                let shifted_weights = self.shifted_norm_weights(runtime, norm_weights)?;
                runtime.rms_norm(
                    hyper_input,
                    &shifted_weights,
                    self.plan.rms_norm_epsilon as f32,
                )
            },
        )
    }

    /// Gated mix: normalize, project down, SiLU scaled by one over the
    /// stream count, project up, sigmoid, multiply into the normalized
    /// streams, and average over streams.
    ///
    /// Returns the mixed block input plus the normalized streams, so the
    /// paired combine reuses the same normalization exactly as the pinned
    /// algebra requires.
    ///
    /// # Errors
    /// When any MLX operation fails or a shape disagrees with the plan.
    pub fn gated_mix(
        &self,
        runtime: &MlxRuntime,
        norm_weights: &MlxArray,
        down_weights: &MlxArray,
        up_weights: &MlxArray,
        hyper_input: &MlxArray,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(MlxArray, MlxArray), MlxRuntimeError> {
        performance_attribution
            .measure_operation(PerformanceOperation::Qwen4ExpHyperConnectionMixing, |_| {
                self.gated_mix_inner(runtime, norm_weights, down_weights, up_weights, hyper_input)
            })
    }

    fn gated_mix_inner(
        &self,
        runtime: &MlxRuntime,
        norm_weights: &MlxArray,
        down_weights: &MlxArray,
        up_weights: &MlxArray,
        hyper_input: &MlxArray,
    ) -> Result<(MlxArray, MlxArray), MlxRuntimeError> {
        let stream_count = self.plan.stream_count;
        let width = self.plan.stream_width;
        let low_rank = self.plan.low_rank;
        let hyper = self.hyper_width();
        self.validate_hyper_shape("hyper-connection mixing", hyper_input)?;
        self.validate_hyper_shape("hyper-connection mixing", norm_weights)?;
        let input_shape = hyper_input.shape();
        if input_shape.len() != 2 {
            return Err(MlxRuntimeError::RuntimeOperation {
                operation: "hyper-connection mixing",
                description: format!(
                    "hyper input must be [token_count, {hyper}], got {:?}",
                    input_shape
                ),
            });
        }
        let token_count = input_shape[0];
        let normalized = self.grouped_normalize_inner(runtime, norm_weights, hyper_input)?;
        // Project down: [low_rank, hyper] by [hyper, token_count].
        let down_array = runtime.reshape(down_weights, &[low_rank as i32, hyper])?;
        let transposed = runtime.transpose_axes(&normalized, &[1, 0])?;
        let hidden = runtime.matmul(&down_array, &transposed)?;
        // SiLU scaled by one over the stream count, in float32 so a
        // bfloat16 activation cannot narrow the sigmoid reduction.
        let scale = runtime.array_from_f32(
            &vec![1.0_f32 / stream_count as f32; low_rank as usize],
            &[low_rank as i32, 1],
        )?;
        let scaled = runtime.multiply(&hidden, &scale)?;
        let activated = runtime.sigmoid(&scaled)?;
        let silu = runtime.multiply(&scaled, &activated)?;
        // Project up and reduce by sigmoid: [hyper, low_rank] by
        // [low_rank, token_count].
        let up_array = runtime.reshape(up_weights, &[hyper, low_rank as i32])?;
        let gate = runtime.matmul(&up_array, &silu)?;
        let gate = runtime.sigmoid(&gate)?;
        // Reshape the gate to token-major before the elementwise multiply:
        // [hyper, token_count] transposes to [token_count, hyper].
        let gate = runtime.transpose_axes(&gate, &[1, 0])?;
        // Multiply the gate into the normalized streams and average over
        // streams: [token_count, streams, width], mean over the stream axis.
        let gated = runtime.multiply(&gate, &normalized)?;
        let grouped = runtime.reshape(&gated, &[token_count, stream_count as i32, width as i32])?;
        let summed = runtime.sum_axis(&grouped, 1, false)?;
        let divisor = runtime.array_from_f32(&vec![stream_count as f32], &[1])?;
        let mixed = runtime.divide(&summed, &divisor)?;
        Ok((mixed, normalized))
    }

    /// Gated combine: a per-stream injection weight of
    /// `2 · sigmoid(project(normalized) / stream_count)` scales the block
    /// output before it is added to the raw residual.
    ///
    /// # Errors
    /// When any MLX operation fails or a shape disagrees with the plan.
    pub fn gated_combine(
        &self,
        runtime: &MlxRuntime,
        inject_weights: &MlxArray,
        block_output: &MlxArray,
        hyper_input: &MlxArray,
        normalized: &MlxArray,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, MlxRuntimeError> {
        performance_attribution.measure_operation(
            PerformanceOperation::Qwen4ExpHyperConnectionInjection,
            |_| {
                self.validate_hyper_shape("hyper-connection injection", hyper_input)?;
                self.validate_hyper_shape("hyper-connection injection", normalized)?;
                let stream_count = self.plan.stream_count;
                let width = self.plan.stream_width;
                let hyper = self.hyper_width();
                let input_shape = hyper_input.shape();
                if input_shape.len() != 2 {
                    return Err(MlxRuntimeError::RuntimeOperation {
                        operation: "hyper-connection injection",
                        description: format!(
                            "hyper input must be [token_count, {hyper}], got {:?}",
                            input_shape
                        ),
                    });
                }
                let token_count = input_shape[0];
                let output_shape = block_output.shape();
                if output_shape.last() != Some(&(width as i32)) {
                    return Err(MlxRuntimeError::RuntimeOperation {
                        operation: "hyper-connection injection",
                        description: format!(
                            "block output must end in {width} elements, got {:?}",
                            output_shape
                        ),
                    });
                }
                // Injection: [streams, hyper] by [hyper, token_count], then
                // sigmoid scaled by two.
                let inject_array =
                    runtime.reshape(inject_weights, &[stream_count as i32, hyper])?;
                let transposed = runtime.transpose_axes(normalized, &[1, 0])?;
                let injection = runtime.matmul(&inject_array, &transposed)?;
                // The pinned algebra scales the projection by one over the
                // stream count before the sigmoid.
                let stream_divisor = runtime.array_from_f32(&vec![stream_count as f32], &[1])?;
                let injection = runtime.divide(&injection, &stream_divisor)?;
                // The matmul yields [streams, token_count]; the elementwise
                // scale below broadcasts against [token_count, streams, ...],
                // so the stream axis must become the middle axis.
                let injection = runtime.transpose_axes(&injection, &[1, 0])?;
                let injection = runtime.sigmoid(&injection)?;
                let two = runtime.array_from_f32(&vec![2.0_f32], &[1])?;
                let injection = runtime.multiply(&injection, &two)?;
                // The injection scales one width per stream: expand the
                // final axis so the multiply broadcasts against
                // [token_count, streams, width].
                let injection = runtime.expand_dims(&injection, -1)?;
                // Scale the block output into every stream and add to the raw
                // residual: [token_count, 1, width] broadcast against
                // [token_count, streams, width].
                let residual = runtime.reshape(
                    hyper_input,
                    &[token_count, stream_count as i32, width as i32],
                )?;
                let expanded_output =
                    runtime.reshape(block_output, &[token_count, 1, width as i32])?;
                let scaled = runtime.multiply(&expanded_output, &injection)?;
                let combined = runtime.add(&residual, &scaled)?;
                runtime.reshape(&combined, &[token_count, hyper])
            },
        )
    }

    /// Average-pooling mix: the plain mean over streams.
    ///
    /// # Errors
    /// When any MLX operation fails or a shape disagrees with the plan.
    pub fn average_mix(
        &self,
        runtime: &MlxRuntime,
        hyper_input: &MlxArray,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, MlxRuntimeError> {
        performance_attribution.measure_operation(
            PerformanceOperation::Qwen4ExpHyperConnectionMixing,
            |_| {
                self.validate_hyper_shape("hyper-connection mixing", hyper_input)?;
                let input_shape = hyper_input.shape();
                let token_count = input_shape[0];
                let grouped = runtime.reshape(
                    hyper_input,
                    &[
                        token_count,
                        self.plan.stream_count as i32,
                        self.plan.stream_width as i32,
                    ],
                )?;
                let summed = runtime.sum_axis(&grouped, 1, false)?;
                let divisor = runtime.array_from_f32(&vec![self.plan.stream_count as f32], &[1])?;
                runtime.divide(&summed, &divisor)
            },
        )
    }

    /// Average-pooling combine: the block output added to every stream.
    ///
    /// # Errors
    /// When any MLX operation fails or a shape disagrees with the plan.
    pub fn average_combine(
        &self,
        runtime: &MlxRuntime,
        block_output: &MlxArray,
        hyper_input: &MlxArray,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, MlxRuntimeError> {
        performance_attribution.measure_operation(
            PerformanceOperation::Qwen4ExpHyperConnectionInjection,
            |_| {
                self.validate_hyper_shape("hyper-connection injection", hyper_input)?;
                let input_shape = hyper_input.shape();
                let token_count = input_shape[0];
                let width = self.plan.stream_width;
                let output_shape = block_output.shape();
                if output_shape.last() != Some(&(width as i32)) {
                    return Err(MlxRuntimeError::RuntimeOperation {
                        operation: "hyper-connection injection",
                        description: format!(
                            "block output must end in {width} elements, got {:?}",
                            output_shape
                        ),
                    });
                }
                let grouped = runtime.reshape(
                    hyper_input,
                    &[token_count, self.plan.stream_count as i32, width as i32],
                )?;
                let expanded = runtime.reshape(block_output, &[token_count, 1, width as i32])?;
                let combined = runtime.add(&grouped, &expanded)?;
                runtime.reshape(
                    &combined,
                    &[token_count, (self.plan.stream_count * width) as i32],
                )
            },
        )
    }

    fn grouped_normalize_inner(
        &self,
        runtime: &MlxRuntime,
        norm_weights: &MlxArray,
        hyper_input: &MlxArray,
    ) -> Result<MlxArray, MlxRuntimeError> {
        // MLX's fused RMS normalization reduces over the final axis and
        // requires the weight to match it, so each stream normalizes as its
        // own [token_count, width] group with its own weight slice. The
        // stream count is small and fixed by configuration, and the
        // direct-MLX oracle proved this exact composition.
        let stream_count = self.plan.stream_count;
        let width = self.plan.stream_width;
        let input_shape = hyper_input.shape();
        let token_count = input_shape[0];
        let shifted_weights = self.shifted_norm_weights(runtime, norm_weights)?;
        let mut normalized_parts = Vec::with_capacity(stream_count as usize);
        for stream in 0..stream_count {
            let input_slice = runtime.slice(
                hyper_input,
                &[0, stream as i32 * width as i32],
                &[token_count, (stream + 1) as i32 * width as i32],
                &[1, 1],
            )?;
            let weight_slice = runtime.slice(
                &shifted_weights,
                &[(stream as i32) * width as i32],
                &[(stream + 1) as i32 * width as i32],
                &[1],
            )?;
            normalized_parts.push(runtime.rms_norm(
                &input_slice,
                &weight_slice,
                self.plan.rms_norm_epsilon as f32,
            )?);
        }
        let stacked = runtime.stack_axis(&normalized_parts.iter().collect::<Vec<_>>(), 1)?;
        runtime.reshape(&stacked, input_shape.as_slice())
    }
}
