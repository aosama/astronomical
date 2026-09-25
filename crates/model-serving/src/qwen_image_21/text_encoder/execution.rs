//! The Qwen-Image-2.1 text encoder owner: load once, encode with staged evaluation.
//!
//! Port of the Qwen3-VL language model for the text-only conditioning path: token embedding →
//! 36 decoder layers (GQA, QK-norm, half-split rope, SwiGLU) → final RMS norm → hidden states
//! `(1, tokens, 4096)`, exactly what the denoising transformer's `text_embeddings` input
//! expects. Each layer builds its graph and synchronizes before the next, keeping wired memory
//! bounded through the 4 GB quantized stack.

use std::fs::File;

use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime};

use crate::qwen_image_21::QwenImage21EngineError;
use crate::qwen_image_21::mlx_math::attention_scale;
use crate::{PerformanceAttribution, PerformanceOperation};

use super::blocks::{EncoderBlockContext, build_causal_mask, build_rope_tables, forward_layer};
use super::weights::{HEAD_WIDTH, LAYER_COUNT, QwenImage21TextEncoderWeights};

pub struct QwenImage21TextEncoder {
    weights: QwenImage21TextEncoderWeights,
}

impl QwenImage21TextEncoder {
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
        let tensors = performance_attribution
            .measure_operation(PerformanceOperation::ImageTextComponentMapping, |_| {
                runtime.load_safetensors(weights_file, file_read_metrics)
            })?;
        let weights = performance_attribution
            .measure_operation(PerformanceOperation::ImageTextComponentLoading, |_| {
                QwenImage21TextEncoderWeights::load(&tensors, LAYER_COUNT)
            })?;
        Ok(Self { weights })
    }

    pub fn layer_count(&self) -> usize {
        self.weights.layers().len()
    }

    /// Encodes one prompt's token ids into hidden states `(1, tokens, 4096)` in BF16.
    pub fn encode(
        &self,
        runtime: &MlxRuntime,
        token_ids: &[u32],
    ) -> Result<MlxArray, QwenImage21EngineError> {
        let mut attribution = PerformanceAttribution::disabled();
        self.encode_with_performance_attribution(runtime, token_ids, &mut attribution)
    }

    pub fn encode_with_performance_attribution(
        &self,
        runtime: &MlxRuntime,
        token_ids: &[u32],
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, QwenImage21EngineError> {
        let sequence_length = token_ids.len();
        let (rope_cosines, rope_sines) = build_rope_tables(runtime, sequence_length)?;
        let causal_mask = build_causal_mask(runtime, sequence_length)?;
        let context = EncoderBlockContext {
            rope_cosines: &rope_cosines,
            rope_sines: &rope_sines,
            causal_mask: &causal_mask,
            attention_scale: attention_scale(HEAD_WIDTH)?,
        };

        let mut hidden_states = performance_attribution.measure_operation(
            PerformanceOperation::ImageQwenLayerGraphConstruction,
            |_| self.weights.embedding().forward(runtime, token_ids),
        )?;
        performance_attribution.measure_operation(
            PerformanceOperation::ImageQwenLayerSynchronizationWait,
            |_| hidden_states.evaluate(),
        )?;

        for layer_weights in self.weights.layers() {
            let advanced = performance_attribution.measure_operation(
                PerformanceOperation::ImageQwenLayerGraphConstruction,
                |_| forward_layer(runtime, layer_weights, &hidden_states, &context),
            )?;
            performance_attribution.measure_operation(
                PerformanceOperation::ImageQwenLayerSynchronizationWait,
                |_| advanced.evaluate(),
            )?;
            hidden_states = advanced;
        }

        // The reference deliberately bypasses the final RMSNorm: it registers a forward hook
        // that returns the norm module's *input*, because the denoising transformer was trained
        // on the last decoder layer's pre-norm hidden states. Applying the norm here would
        // strip most of the signal the transformer reads — the reference comment calls it "a
        // third of the signal", visible first in rendered text.
        let last_layer_output = runtime.astype(&hidden_states, MlxDtype::BFloat16)?;
        performance_attribution.measure_operation(
            PerformanceOperation::ImageQwenLayerSynchronizationWait,
            |_| last_layer_output.evaluate(),
        )?;
        Ok(last_layer_output)
    }
}
