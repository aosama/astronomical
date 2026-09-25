//! The Qwen-Image-2.1 transformer owner: load once, forward with bounded staged evaluation.
//!
//! The forward mirrors the reference's non-cached path (`kv_cache = None`): the full joint
//! sequence runs prefill semantics with the exact multi-segment block-causal attention. Each
//! stage — preparation, every block, the output head — builds its graph and synchronizes before
//! the next, so intermediate tensors release wired memory between stages.

use std::fs::File;

use astronomical_runtime_integration::{MlxArray, MlxRuntime};

use crate::qwen_image_21::QwenImage21EngineError;
use crate::qwen_image_21::rope::QwenImage21Rope;
use crate::{PerformanceAttribution, PerformanceOperation};

use super::blocks::{BlockContext, forward_block};
use super::preparation::{QwenImage21TransformerRequest, forward_output_head, prepare_forward};
use super::weights::{HEAD_COUNT, HEAD_WIDTH, QwenImage21TransformerWeights};

/// The reviewed artifact's block count (`num_layers`).
const BLOCK_COUNT: usize = 32;

pub struct QwenImage21Transformer {
    weights: QwenImage21TransformerWeights,
    rope: QwenImage21Rope,
}

impl QwenImage21Transformer {
    pub fn load(runtime: &MlxRuntime, weights_file: File) -> Result<Self, QwenImage21EngineError> {
        let mut attribution = PerformanceAttribution::disabled();
        Self::load_with_performance_attribution(runtime, weights_file, &mut attribution)
    }

    pub fn load_with_performance_attribution(
        runtime: &MlxRuntime,
        weights_file: File,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Self, QwenImage21EngineError> {
        let file_read_metrics = performance_attribution.positional_file_read_metrics();
        let tensors = performance_attribution.measure_operation(
            PerformanceOperation::ImageTransformerComponentMapping,
            |_| runtime.load_safetensors(weights_file, file_read_metrics),
        )?;
        let weights = performance_attribution.measure_operation(
            PerformanceOperation::ImageTransformerComponentLoading,
            |_| QwenImage21TransformerWeights::load(&tensors, BLOCK_COUNT),
        )?;
        Ok(Self {
            weights,
            rope: QwenImage21Rope::default(),
        })
    }

    pub fn block_count(&self) -> usize {
        self.weights.block_count()
    }

    /// Runs one denoising forward and returns the target-image slice
    /// `(batch, target_token_count, 64)`.
    pub fn forward(
        &self,
        runtime: &MlxRuntime,
        request: &QwenImage21TransformerRequest<'_>,
    ) -> Result<MlxArray, QwenImage21EngineError> {
        let mut attribution = PerformanceAttribution::disabled();
        self.forward_with_performance_attribution(runtime, request, &mut attribution)
    }

    pub fn forward_with_performance_attribution(
        &self,
        runtime: &MlxRuntime,
        request: &QwenImage21TransformerRequest<'_>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, QwenImage21EngineError> {
        let prepared = performance_attribution.measure_operation(
            PerformanceOperation::ImageTransformerBlockGroupGraphConstruction,
            |_| prepare_forward(runtime, &self.weights, request, &self.rope),
        )?;
        performance_attribution.measure_operation(
            PerformanceOperation::ImageTransformerBlockGroupSynchronizationWait,
            |_| prepared.joint_hidden_states.evaluate(),
        )?;

        let block_context = BlockContext {
            attention_scale: prepared.attention_scale,
            rope_cosines: &prepared.rope_cosines,
            rope_sines: &prepared.rope_sines,
            segments: &prepared.segments,
            segment_masks: &prepared.segment_masks,
            attention_one_plus_scale: &prepared.attention_one_plus_scale,
            attention_gate: &prepared.attention_gate,
            feed_forward_one_plus_scale: &prepared.feed_forward_one_plus_scale,
            feed_forward_gate: &prepared.feed_forward_gate,
            head_count: HEAD_COUNT,
            head_width: HEAD_WIDTH,
        };

        // `retain` duplicates the handle so the prepared context stays intact for the head.
        let mut hidden_states = prepared.joint_hidden_states.retain()?;
        for block_weights in &self.weights.blocks {
            let advanced = performance_attribution.measure_operation(
                PerformanceOperation::ImageTransformerBlockGroupGraphConstruction,
                |_| forward_block(runtime, block_weights, &hidden_states, &block_context),
            )?;
            performance_attribution.measure_operation(
                PerformanceOperation::ImageTransformerBlockGroupSynchronizationWait,
                |_| advanced.evaluate(),
            )?;
            hidden_states = advanced;
        }

        let output = performance_attribution.measure_operation(
            PerformanceOperation::ImageTransformerBlockGroupGraphConstruction,
            |_| forward_output_head(runtime, &self.weights, &prepared, &hidden_states),
        )?;
        performance_attribution.measure_operation(
            PerformanceOperation::ImageTransformerBlockGroupSynchronizationWait,
            |_| output.evaluate(),
        )?;
        Ok(output)
    }
}
