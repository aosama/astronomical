//! The folded ternary family's straight text model.
//!
//! Owns the loaded weights, the MLX runtime policy, and the compiled
//! elementwise graphs, and drives the straight forward pass (embedding,
//! decoder layers, final norm, language-model head). The Hadamard transform
//! is applied unconditionally inside the packed projection types — there is
//! no branch anywhere in this path that could skip it.

use astronomical_runtime_integration::{MlxRuntime, MlxSafetensors};
use mlx_c_rust::{
    MlxArray, MlxCompiledDualHeadNorm, MlxCompiledElementwiseGraphs,
    MlxCompiledGatedDeltaRateActivation, MlxCompiledSwiGlu, MlxMetalKernel,
};

use crate::PerformanceAttribution;

use super::artifacts::ValidatedFoldedPack;
use super::configuration::FoldedPackConfigError;
use super::hadamard::HadamardContractError;
use weights::{PrismHadamardWeights, build_gdn_normalization_scale_weights, load_family_weights};

mod forward;
mod fused_projection;
mod fwht;
mod measurement;
pub(crate) mod signed_rotation_kernel;
mod weights;

#[cfg(test)]
mod tests;

pub(crate) use forward::{PrismStateSlotLayout, StraightGenerationState};
pub use measurement::{
    PrismHadamardGenerationState, PrismHadamardGreedySession, PrismHadamardProbeDimensions,
    PrismHadamardProbeProjection,
};

/// The loaded, ready-to-run folded ternary text model.
#[derive(Debug)]
pub struct PrismHadamardTextModel {
    pub(crate) pack: ValidatedFoldedPack,
    pub(crate) runtime: MlxRuntime,
    pub(crate) compiled_elementwise_graphs: MlxCompiledElementwiseGraphs,
    pub(crate) compiled_swiglu: MlxCompiledSwiGlu,
    /// The dual-output rate-activation replay: one launch produces the
    /// sigmoid update rates and the decay factors every gated-delta layer
    /// consumed as two separate launches before (issue #1366).
    pub(crate) compiled_gated_delta_rate_activation: MlxCompiledGatedDeltaRateActivation,
    /// The dual-output query/key head-normalization replay for the
    /// gated-delta prework, with this model's RMS epsilon baked in.
    pub(crate) compiled_dual_head_norm: MlxCompiledDualHeadNorm,
    /// The fused gated-delta Metal kernel, retained only when this GPU passes
    /// the capability probe; `None` demotes the recurrence to the shared
    /// one-token MLX ops route, which is orders of magnitude slower on
    /// prompt processing and exists only as the documented fallback.
    pub(crate) gated_delta_kernel: Option<MlxMetalKernel>,
    /// The one-launch signed Hadamard rotation, retained when this GPU
    /// passes its bit-exact probe and the pack's block size is one the
    /// kernel reproduces; `None` keeps the composed three-launch route.
    pub(crate) fused_rotation_kernel: Option<MlxMetalKernel>,
    pub(crate) weights: PrismHadamardWeights,
    pub(crate) gdn_query_scale_weight: MlxArray,
    pub(crate) gdn_key_scale_weight: MlxArray,
    /// Each decoder layer's index within its own attention kind: the slot its
    /// convolution/recurrent (gated-delta) or key-value (full-attention) state
    /// occupies in the generation state.
    pub(crate) decoder_layer_state_slots: Vec<usize>,
}

impl PrismHadamardTextModel {
    /// Loads the complete weight set from a validated pack into the runtime.
    pub fn new(
        runtime: MlxRuntime,
        mut pack: ValidatedFoldedPack,
    ) -> Result<Self, PrismHadamardModelError> {
        let weights_file = pack
            .take_weights_file()
            .map_err(PrismHadamardModelError::Artifact)?;
        let loaded_tensors: MlxSafetensors = runtime
            .load_safetensors(weights_file.into_file(), None)
            .map_err(PrismHadamardModelError::SafetensorsLoading)?;
        let weights = load_family_weights(&runtime, &pack, &loaded_tensors)?;
        // Fusing stacked copies of the sibling projections, so the members'
        // own arrays are garbage once the tensor map drops; without this
        // release their buffers stay in the MLX allocator cache for the
        // life of the process, inflating the worker's resident footprint by
        // most of the pack size.
        drop(loaded_tensors);
        runtime
            .clear_allocator_cache()
            .map_err(PrismHadamardModelError::ForwardRuntime)?;
        let head_dimension = pack.config().text().linear_key_head_dim;
        let (gdn_query_scale_weight, gdn_key_scale_weight) =
            build_gdn_normalization_scale_weights(&runtime, head_dimension)?;
        let compiled_elementwise_graphs =
            MlxCompiledElementwiseGraphs::new().map_err(PrismHadamardModelError::Forward)?;
        let compiled_swiglu = MlxCompiledSwiGlu::new().map_err(PrismHadamardModelError::Forward)?;
        // The dual head-norm graph bakes the pack's RMS epsilon at
        // compilation; the rate-activation graph is epsilon-free.
        let rms_norm_epsilon = f32::from_bits(pack.config().text().rms_norm_epsilon_bits);
        let compiled_gated_delta_rate_activation =
            MlxCompiledGatedDeltaRateActivation::new().map_err(PrismHadamardModelError::Forward)?;
        let compiled_dual_head_norm = MlxCompiledDualHeadNorm::new(rms_norm_epsilon)
            .map_err(PrismHadamardModelError::Forward)?;
        // The capability probes are cached per worker process, so this
        // mirrors the core loader's cost and its fused-vs-ops decision exactly.
        let kernel_capabilities = crate::kernel_capability::worker_process_kernel_capabilities(
            &runtime,
            &mut PerformanceAttribution::disabled(),
        );
        let gated_delta_kernel = if kernel_capabilities
            .is_custom_kernel_supported(crate::CustomMetalKernelFamily::GATED_DELTA_SEQUENCE)
        {
            Some(
                crate::qwen3_5_core::model_math::gated_delta_sequence::qwen3_5_gated_delta_kernel()
                    .map_err(PrismHadamardModelError::KernelRetention)?,
            )
        } else {
            tracing::info!(
                "fused gated-delta kernel demoted to the MLX ops fallback for this worker process"
            );
            None
        };
        let hadamard_block_size = pack.hadamard_contract().block_size();
        let fused_rotation_kernel = if kernel_capabilities.is_custom_kernel_supported(
            crate::CustomMetalKernelFamily::PRISM_HADAMARD_SIGNED_ROTATION,
        ) && signed_rotation_kernel::is_signed_rotation_eligible(
            hadamard_block_size,
        ) {
            Some(
                signed_rotation_kernel::signed_rotation_kernel(hadamard_block_size)
                    .map_err(PrismHadamardModelError::Forward)?,
            )
        } else {
            tracing::info!(
                "fused Hadamard rotation demoted to the composed MLX route for this worker process"
            );
            None
        };
        let decoder_layer_state_slots = decoder_layer_state_slots(&weights);
        Ok(Self {
            pack,
            runtime,
            compiled_elementwise_graphs,
            compiled_swiglu,
            compiled_gated_delta_rate_activation,
            compiled_dual_head_norm,
            gated_delta_kernel,
            fused_rotation_kernel,
            weights,
            gdn_query_scale_weight,
            gdn_key_scale_weight,
            decoder_layer_state_slots,
        })
    }

    /// The validated pack configuration.
    pub const fn pack(&self) -> &ValidatedFoldedPack {
        &self.pack
    }

    /// Carries `activations` into the rotated basis of their width, through
    /// the retained one-launch kernel or the composed route.
    pub(crate) fn rotate(
        &self,
        activations: &MlxArray,
    ) -> Result<fwht::RotatedActivations, mlx_c_rust::MlxCError> {
        self.weights.rotate(
            &self.runtime,
            &self.compiled_elementwise_graphs,
            self.fused_rotation_kernel.as_ref(),
            activations,
        )
    }

    /// Greedy generation: feeds the whole prompt as one prefill pass, then
    /// decodes one token at a time, stopping when an end-of-sequence id
    /// appears or the token budget is spent. Returns the generated ids,
    /// excluding the stop token.
    pub fn generate_greedy(
        &self,
        prompt_token_ids: &[i32],
        maximum_new_token_count: usize,
        stop_token_ids: &[u32],
    ) -> Result<Vec<u32>, PrismHadamardModelError> {
        if maximum_new_token_count == 0 {
            validate_greedy_generation_request(
                prompt_token_ids,
                maximum_new_token_count,
                self.pack.config().text().vocab_size,
                self.pack.config().text().max_position_embeddings,
            )?;
            return Ok(Vec::new());
        }
        let prompt_processing_chunk_tokens = u32::try_from(prompt_token_ids.len())
            .unwrap_or(u32::MAX)
            .max(1);
        let mut session = self.begin_greedy_session(
            prompt_token_ids,
            prompt_processing_chunk_tokens,
            maximum_new_token_count,
            stop_token_ids,
        )?;
        let mut generated_token_ids = Vec::new();
        while let Some(next_token) = session.next_token()? {
            generated_token_ids.push(next_token);
        }
        Ok(generated_token_ids)
    }

    /// The highest-probability vocabulary id at the final sequence position.
    pub(crate) fn argmax_last_token(
        &self,
        logits: &MlxArray,
    ) -> Result<u32, PrismHadamardModelError> {
        let per_position_indices = self
            .runtime
            .argmax_axis(logits, 2)
            .and_then(|indices| indices.to_vec_u32())
            .map_err(PrismHadamardModelError::Forward)?;
        per_position_indices
            .last()
            .copied()
            .ok_or(PrismHadamardModelError::EmptyGeneration)
    }
}

/// Numbers each decoder layer within its attention kind so a layer can find
/// its own convolution/recurrent or key-value state slot directly.
fn decoder_layer_state_slots(weights: &PrismHadamardWeights) -> Vec<usize> {
    let mut linear_layer_count = 0_usize;
    let mut full_layer_count = 0_usize;
    weights
        .decoder_layers
        .iter()
        .map(|decoder_layer| match decoder_layer.attention {
            weights::DecoderLayerAttentionWeights::Linear(_) => {
                let slot = linear_layer_count;
                linear_layer_count += 1;
                slot
            }
            weights::DecoderLayerAttentionWeights::Full(_) => {
                let slot = full_layer_count;
                full_layer_count += 1;
                slot
            }
        })
        .collect()
}

pub(crate) fn validate_greedy_generation_request(
    prompt_token_ids: &[i32],
    maximum_new_token_count: usize,
    vocabulary_size: u32,
    maximum_context_tokens: u32,
) -> Result<(), PrismHadamardModelError> {
    let invalid_request = |description: &str| PrismHadamardModelError::InvalidGenerationRequest {
        description: description.to_owned(),
    };
    if prompt_token_ids.is_empty() {
        return Err(invalid_request(
            "the prompt must contain at least one token",
        ));
    }
    if prompt_token_ids
        .iter()
        .any(|token_id| *token_id < 0 || *token_id as u32 >= vocabulary_size)
    {
        return Err(invalid_request(
            "the prompt contains a token outside the model vocabulary",
        ));
    }
    if i32::try_from(prompt_token_ids.len()).is_err() {
        return Err(invalid_request(
            "the prompt token count exceeds the model shape range",
        ));
    }
    let requested_context_tokens = prompt_token_ids
        .len()
        .checked_add(maximum_new_token_count)
        .ok_or_else(|| invalid_request("the requested context token count overflowed"))?;
    if requested_context_tokens > maximum_context_tokens as usize {
        return Err(invalid_request(
            "the prompt and output budget exceed the model context window",
        ));
    }
    Ok(())
}

/// A cause-preserving failure while loading or running the family model.
#[derive(Debug, thiserror::Error)]
pub enum PrismHadamardModelError {
    #[error("folded pack validation failed")]
    Artifact(#[from] super::artifacts::FoldedPackValidationError),
    #[error("folded pack configuration failed")]
    Config(#[from] FoldedPackConfigError),
    #[error("folded pack hadamard contract refused")]
    Hadamard(#[from] HadamardContractError),
    #[error("folded pack weight casting failed: {0}")]
    WeightCasting(#[source] mlx_c_rust::MlxCError),
    #[error("folded pack safetensors loading failed: {0}")]
    SafetensorsLoading(#[source] astronomical_runtime_integration::MlxRuntimeError),
    #[error("folded pack tensor lookup failed for {tensor_name:?}: {source}")]
    TensorLookup {
        tensor_name: String,
        #[source]
        source: astronomical_runtime_integration::MlxRuntimeError,
    },
    #[error("folded pack weight slicing failed: {description}")]
    WeightSlicing { description: String },
    #[error("folded pack forward pass failed: {0}")]
    Forward(#[source] mlx_c_rust::MlxCError),
    #[error("folded pack forward-pass runtime state failed: {0}")]
    ForwardRuntime(#[source] astronomical_runtime_integration::MlxRuntimeError),
    #[error("folded pack declares unsupported layer type {layer_type:?}")]
    UnsupportedLayerType { layer_type: String },
    #[error("the fused gated-delta kernel could not be retained: {0}")]
    KernelRetention(#[source] astronomical_runtime_integration::MlxRuntimeError),
    #[error("the generation pass produced no position to decode from")]
    EmptyGeneration,
    #[error("folded pack greedy generation request is invalid: {description}")]
    InvalidGenerationRequest { description: String },
}
