//! The shared Qwen3.5 model base: one loaded artifact's runtime, configuration,
//! typed weights, vision tower, retained Metal kernels, and compiled graphs.
//!
//! Issue #1132: the engine wrapper (`Qwen3_5TextModel`) derefs to this base
//! for every shared field and method.

use mlx_bridge::{MlxArray, MlxCompiledElementwiseGraphs, MlxCompiledSwiGlu, MlxMetalKernel};
use runtime_integration::MlxRuntime;

use std::cell::RefCell;

use crate::qwen3_5_core::configuration::Qwen3_5Config;
use crate::qwen3_5_core::model::mlx_layout_projection::QuantizedProjectionParts;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AffineWeights;
use crate::qwen3_5_core::model_math::weights::{EmbeddingResidency, Qwen3_5Weights};
use crate::qwen3_5_core::vision::Qwen3_5VisionModel;
use crate::{AstroDecoderStateCacheLayout, ModelMemoryFacts};
use text_generation::execution_error::TextExecutionError;

use text_generation::model::model_chunking_configuration::TextModelChunkingConfiguration;

/// Faster paths a model turns on only after proving them on this machine.
/// The resident loader turns on the fused expert combination when its
/// capability probe passes; every other path starts off and runs MLX's own
/// kernels until something enables it.
#[derive(Debug, Default)]
pub struct Qwen3_5OptionalAccelerators {
    /// Flash-Next's multi-token-prediction head, when the checkpoint ships
    /// one; its arrays stay lazy until a request first drafts.
    pub mtp_head: Option<Box<crate::qwen4_exp::mtp_head_weights::MtpHeadWeights>>,
    /// Flash-Next's fused lane-mixing kernels, once
    /// `enable_fused_lane_mixing` has found them bit for bit equal to the
    /// composed steps on this GPU.
    pub lane_mixing_kernels: Option<crate::qwen4_exp::lane_mixing_kernels::LaneMixingKernels>,
    /// Opt-in engine for linear-attention output gates during prefill; `None`
    /// keeps them on the GPU.
    /// Whole dense MLP programs on the Neural Engine, one a decoder layer,
    /// attached once the opt-in prompt reading has built them (issue #161).
    pub neural_engine_mlp_prefill:
        Option<crate::qwen3_5_core::neural_engine::neural_engine_mlp_prefill::NeuralEngineMlpPrefill>,
    pub output_gate_engine: Option<
        Box<dyn crate::qwen3_5_core::model::output_gate_engine::LinearAttentionOutputGateEngine>,
    >,
    /// The DFlash drafter the engine hosts beside this model,
    /// when one that pairs with it is installed.
    pub draft_runtime: Option<
        Box<crate::qwen3_5_core::draft_tree_decode::engine_draft_runtime::Qwen3_5SpeculativeDraftRuntime>,
    >,
    /// What the engine's search for a drafter found, for status:
    /// drafting when `draft_runtime` holds one, otherwise why not. `None`
    /// until the engine has looked.
    pub draft_model_status: Option<ipc_protocol::DraftModelStatus>,
    /// The register-ordered 4-bit matmul's kernels; set once
    /// `declare_register_ordered_kernels` has declared them for a drafter,
    /// and every dense projection converted since multiplies through them.
    pub register_ordered_matmul_kernels: Option<crate::row_batched_matmul::RowBatchedMatmulKernels>,
    /// The gated-delta recurrence over a draft tree's rows; set by
    /// `enable_draft_tree_verification`.
    pub gated_delta_tree_kernel: Option<mlx_bridge::MlxMetalKernel>,
    /// Advances a DeltaNet layer over a verified tree's kept rows; set with
    /// the tree kernel.
    pub gated_delta_replay_kernel: Option<mlx_bridge::MlxMetalKernel>,
    /// Full attention for a draft tree's rows on the M5 matrix units, each
    /// row exact to its one-row step; set with the tree kernel when the
    /// target's attention shape fits it.
    pub row_exact_tree_attention_kernels: Option<
        crate::qwen3_5_core::model_math::row_exact_tree_attention::RowExactTreeAttentionKernels,
    >,
    /// Runs MLX's tree attention on float32 copies of its bf16 inputs, the
    /// exact reference, for measurements of how attention precision moves
    /// the drafted tokens a round keeps. Serving leaves it off.
    pub tree_attention_in_float32_for_measurement: bool,
    /// Builds a whole verify pass before the GPU starts on it, so a
    /// profiler handed the pass sees every kernel in it. Serving starts the
    /// GPU on each few layers while the next are built, and leaves it off.
    pub tree_pass_built_whole_for_measurement: bool,
    /// When set, every tree attention through the row-exact kernel also runs
    /// MLX's bf16 and float32 attention on the same inputs and records how
    /// many output elements each path's bf16 result differs from float32's.
    /// For measurements only: it evaluates every layer on the spot.
    pub tree_attention_error_record:
        Option<std::cell::RefCell<Vec<super::tree_attention_measurement::TreeAttentionErrorCount>>>,
    /// One Metal pass for the sorted routed-expert sum plus the gated shared
    /// expert on prompt chunks, bit for bit equal to the two-step path it
    /// replaces. The model keeps it when the GPU's capability probe passed
    /// and the sorted reduction is kept.
    pub sorted_expert_shared_combination_kernel: Option<MlxMetalKernel>,
}

/// One resident native Qwen3.5 text model, optional vision tower, and its direct MLX runtime.
#[derive(Debug)]
pub struct Qwen3_5ModelBase {
    pub runtime: MlxRuntime,
    pub config: Qwen3_5Config,
    pub astro_decoder_state_cache_layout: AstroDecoderStateCacheLayout,
    pub weights: Qwen3_5Weights,
    /// Embedding rows read from the SSD per forward instead of a resident
    /// table; `None` keeps the table resident. Only a dense model that
    /// streams from the SSD sets it (issue #1985).
    pub disk_embedding_rows: Option<super::disk_embedding_rows::Qwen3_5DiskEmbeddingRows>,
    /// Flash-Next's n-gram table, read row by row from the SSD; absent for
    /// Qwen 3.5.
    pub ngram_table: Option<crate::qwen4_exp::ngram_table::NgramTable>,
    pub vision_model: Option<Qwen3_5VisionModel>,
    /// Single-source MLX RAM split for context, activations, streaming, and experts.
    pub memory_facts: RefCell<ModelMemoryFacts>,

    pub gated_delta_kernel: Option<MlxMetalKernel>,
    pub gated_delta_checkpoint_kernel: Option<MlxMetalKernel>,
    /// Fused decode prework kernel; None when demoted or environment-disabled.
    pub gdn_decode_prework_kernel: Option<MlxMetalKernel>,
    /// Faster paths a model turns on only after proving them on this machine.
    pub optional_accelerators: Qwen3_5OptionalAccelerators,
    pub sorted_expert_weighted_sum_kernel: Option<MlxMetalKernel>,
    pub compiled_swiglu: MlxCompiledSwiGlu,
    pub compiled_elementwise_graphs: MlxCompiledElementwiseGraphs,
    pub chunking: TextModelChunkingConfiguration,
    /// Model-owned BF16 scalar for the query normalization scale in every
    /// linear-attention forward pass.
    pub inverse_linear_head_dimension_scale: MlxArray,
    /// Model-owned BF16 scalar for the key normalization scale in every
    /// linear-attention forward pass.
    pub inverse_square_root_linear_head_dimension_scale: MlxArray,
    /// Model-owned BF16 per-channel weight folding the query normalization
    /// scale into one `fast_rms_norm` launch on the prefill composed path
    /// (issue #915 item 5); per-channel values equal the scalar scale exactly.
    pub query_normalization_scale_weight: MlxArray,
    /// Model-owned BF16 per-channel weight folding the key normalization
    /// scale into one `fast_rms_norm` launch on the prefill composed path
    /// (issue #915 item 5); per-channel values equal the scalar scale exactly.
    pub key_normalization_scale_weight: MlxArray,
    /// What a decision model adds to the forward pass; empty for chat.
    pub layer_additions:
        RefCell<Option<std::rc::Rc<super::layer_additions::Qwen3_5LayerAdditions>>>,
}

impl Qwen3_5ModelBase {
    /// Returns highest-logit token IDs for each row in a logits tensor.
    pub fn highest_logit_token_ids(
        &self,
        position_logits: &MlxArray,
    ) -> Result<Vec<u32>, TextExecutionError> {
        let selected_tokens = self.select_highest_logit_token(position_logits)?;
        Ok(selected_tokens.to_vec_u32()?)
    }

    pub fn select_highest_logit_token(
        &self,
        final_position_logits: &MlxArray,
    ) -> Result<MlxArray, TextExecutionError> {
        Ok(self.runtime.argmax_axis(final_position_logits, -1)?)
    }

    /// Returns the MLX runtime used by this model.
    #[must_use]
    pub fn runtime(&self) -> &MlxRuntime {
        &self.runtime
    }

    /// Returns the limit this model runs under and its measured geometry.
    #[must_use]
    pub fn memory_facts(&self) -> std::cell::Ref<'_, ModelMemoryFacts> {
        self.memory_facts.borrow()
    }

    /// The MLX active-memory ceiling this model currently runs under.
    #[must_use]
    pub fn mlx_active_memory_ceiling_bytes(&self) -> u64 {
        self.memory_facts.borrow().mlx_active_memory_ceiling_bytes()
    }

    pub fn sorted_expert_weighted_sum_kernel(&self) -> Result<&MlxMetalKernel, TextExecutionError> {
        self.sorted_expert_weighted_sum_kernel
            .as_ref()
            .ok_or(TextExecutionError::InvalidInput {
                description: "sparse Qwen3.5 execution requires a sorted expert output kernel",
            })
    }

    /// Returns the validated text configuration bound to this loaded model.
    #[must_use]
    pub const fn config(&self) -> &Qwen3_5Config {
        &self.config
    }

    /// Returns the validated astro decoder state cache layout bound to this model artifact.
    #[must_use]
    pub const fn astro_decoder_state_cache_layout(&self) -> &AstroDecoderStateCacheLayout {
        &self.astro_decoder_state_cache_layout
    }

    /// Creates an empty per-request decoder state whose per-layer
    /// `astro_decoder_state_cache` dtypes match this model's loaded weights,
    /// exactly as production generation does.
    ///
    /// Direct-MLX callers (decision backbones, probes, journeys) drive raw
    /// forwards without the engine's request plumbing; this constructor keeps
    /// their decoder state identical to production. The public config-only
    /// constructor instead assumes BFloat16 in the `astro_decoder_state_cache`,
    /// which can diverge from a loaded artifact's dtypes and would quietly
    /// measure a different graph than the one serving.
    #[cfg(feature = "direct-mlx")]
    pub fn empty_request_decoder_state_matching_loaded_weights(
        &self,
        full_attention_kv_state_growth_tokens: i32,
    ) -> Result<text_generation::RequestDecoderStateStack, runtime_integration::MlxRuntimeError>
    {
        text_generation::RequestDecoderStateStack::
            empty_from_astro_decoder_state_cache_layout_with_full_attention_kv_state_growth_tokens(
                &self.astro_decoder_state_cache_layout,
                full_attention_kv_state_growth_tokens,
            )
    }

    /// Returns the optional vision tower loaded beside the language model.
    #[must_use]
    pub fn vision_model(&self) -> Option<&Qwen3_5VisionModel> {
        self.vision_model.as_ref()
    }

    /// Releases the vision tower's weights once an image request has resolved
    /// its embeddings, and returns their bytes to the system.
    ///
    /// The MLX buffer cache is cleared here rather than at request end, so the
    /// freed bytes are available to this request's own prefill: on a small
    /// Mac that prefill is the peak that matters. A failed release only logs,
    /// because the request already has its embeddings; the tower then stays
    /// resident and counted, and the next image request tries again.
    pub fn release_vision_tower_after_image_request(&mut self) {
        let Some(vision_model) = self.vision_model.as_mut() else {
            return;
        };
        match vision_model.release_used_weights(&self.runtime) {
            Ok(false) => {}
            Ok(true) => match self
                .runtime
                .synchronize_gpu_stream_and_clear_mlx_buffer_cache()
            {
                Ok(()) => tracing::debug!(
                    released_vision_payload_bytes = vision_model.tower_payload_bytes(),
                    "released the vision tower weights after the image request"
                ),
                Err(astro_ssd_prompt_cache_clear_error) => tracing::warn!(
                    error = %astro_ssd_prompt_cache_clear_error,
                    "released the vision tower weights but could not clear the MLX buffer cache"
                ),
            },
            Err(release_error) => tracing::warn!(
                error = %release_error,
                "vision tower weights stay resident because releasing them failed"
            ),
        }
    }

    /// Returns the resident target payload bytes wired after materialization.
    #[must_use]
    pub fn resident_model_payload_byte_count(&self) -> u64 {
        self.weights
            .resident_payload_bytes(self.embedding_residency())
    }

    /// Materializes the target weights, leaving the decoder layers lazy when
    /// a dense layer streamer owns them. An embedding table read from the SSD
    /// stays lazy too. The model calls this with its own layer
    /// ownership; the base has no public materialize of its own, so a caller
    /// reaching the base through deref cannot load a streamer's layers.
    pub fn materialize_target_weights_with(
        &self,
        decoder_layer_residency: crate::qwen3_5_core::model_math::weights::DecoderLayerResidency,
    ) -> Result<(), TextExecutionError> {
        self.weights.materialize(
            &self.runtime,
            self.embedding_residency(),
            decoder_layer_residency,
        )?;
        self.runtime.evaluate_arrays(&[
            &self.inverse_linear_head_dimension_scale,
            &self.inverse_square_root_linear_head_dimension_scale,
            &self.query_normalization_scale_weight,
            &self.key_normalization_scale_weight,
        ])?;
        Ok(())
    }

    /// Whether this model reads embedding rows from the SSD.
    pub const fn embedding_residency(&self) -> EmbeddingResidency {
        match self.disk_embedding_rows {
            Some(_) => EmbeddingResidency::RowsReadFromDisk,
            None => EmbeddingResidency::Resident,
        }
    }

    pub fn embedding_lookup(
        &self,
        token_indices: &MlxArray,
    ) -> Result<MlxArray, TextExecutionError> {
        match &self.disk_embedding_rows {
            Some(disk_embedding_rows) => disk_embedding_rows.embed_tokens(
                &self.runtime,
                token_indices,
                &self.weights.embedding_weights,
            ),
            None => self.resident_embedding_lookup(token_indices),
        }
    }

    fn resident_embedding_lookup(
        &self,
        token_indices: &MlxArray,
    ) -> Result<MlxArray, TextExecutionError> {
        match &self.weights.embedding_weights {
            Qwen3_5AffineWeights::NativeBfloat16 { weight } => {
                Ok(self.runtime.take_axis(weight, token_indices, 0)?)
            }
            Qwen3_5AffineWeights::Jangtq2Codebook { .. } => Err(TextExecutionError::InvalidInput {
                description: "jangtq2 codebook pages hold routed experts, never embeddings",
            }),
            Qwen3_5AffineWeights::Quantized {
                packed_weight,
                quantization_scales,
                quantization_biases,
                quantization_group_size,
                quantization_bits,
            } => gather_quantized_embedding_rows(
                &self.runtime,
                packed_weight,
                quantization_scales,
                quantization_biases,
                token_indices,
                *quantization_group_size,
                *quantization_bits,
            ),
        }
    }

    /// `activations` [..., input] times a projection's weight transposed:
    /// [..., output]. Bound weights multiply through MLX; a dense projection
    /// converted into register order through the register-ordered matmul.
    pub fn quantized_linear(
        &self,
        activations: &MlxArray,
        projection_weights: &impl Qwen3_5LinearWeights,
    ) -> Result<MlxArray, TextExecutionError> {
        projection_weights.project_rows(self, activations)
    }

    /// `activations` times weights as bound from the artifact.
    pub fn bound_linear(
        &self,
        activations: &MlxArray,
        affine_weights: &Qwen3_5AffineWeights,
    ) -> Result<MlxArray, TextExecutionError> {
        match affine_weights {
            Qwen3_5AffineWeights::NativeBfloat16 { weight } => {
                self.native_bfloat16_linear(activations, weight)
            }
            Qwen3_5AffineWeights::Jangtq2Codebook { .. } => Err(TextExecutionError::InvalidInput {
                description: "jangtq2 codebook pages hold routed experts, never dense linears",
            }),
            Qwen3_5AffineWeights::Quantized {
                packed_weight,
                quantization_scales,
                quantization_biases,
                quantization_group_size,
                quantization_bits,
            } => self.quantized_rows_matmul(
                activations,
                &QuantizedProjectionParts {
                    packed_weight,
                    scales: quantization_scales,
                    biases: quantization_biases,
                    group_size: *quantization_group_size,
                    bits: *quantization_bits,
                },
            ),
        }
    }

    fn native_bfloat16_linear(
        &self,
        activations: &MlxArray,
        weight: &MlxArray,
    ) -> Result<MlxArray, TextExecutionError> {
        let transposed_weight = self.runtime.transpose_axes(weight, &[1, 0])?;
        Ok(self.runtime.matmul(activations, &transposed_weight)?)
    }
}

/// Weights `quantized_linear` multiplies rows by: those bound from the
/// artifact, and dense projections that may have been converted at load.
pub trait Qwen3_5LinearWeights {
    fn project_rows(
        &self,
        model: &Qwen3_5ModelBase,
        activations: &MlxArray,
    ) -> Result<MlxArray, TextExecutionError>;
}

impl Qwen3_5LinearWeights for Qwen3_5AffineWeights {
    fn project_rows(
        &self,
        model: &Qwen3_5ModelBase,
        activations: &MlxArray,
    ) -> Result<MlxArray, TextExecutionError> {
        model.bound_linear(activations, self)
    }
}

/// Gathers rows of a quantized embedding and dequantizes them. The resident
/// and on-disk lookups both call it, so their embeddings are the same
/// operations by construction.
pub fn gather_quantized_embedding_rows(
    runtime: &MlxRuntime,
    packed_rows: &MlxArray,
    scale_rows: &MlxArray,
    bias_rows: &MlxArray,
    selected_row_indices: &MlxArray,
    quantization_group_size: i32,
    quantization_bits: i32,
) -> Result<MlxArray, TextExecutionError> {
    Ok(runtime.dequantize_affine(
        &runtime.take_axis(packed_rows, selected_row_indices, 0)?,
        &runtime.take_axis(scale_rows, selected_row_indices, 0)?,
        &runtime.take_axis(bias_rows, selected_row_indices, 0)?,
        quantization_group_size,
        quantization_bits,
    )?)
}
