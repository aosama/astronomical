//! The artifact-loading pipeline the Qwen3.5 engine runs: language
//! weight and vision binding, the GPU kernel capability verdicts, and the
//! invariant linear-attention scales.

use std::collections::HashMap;
use std::rc::Rc;

use mlx_bridge::{MlxDtype, MlxMetalKernel};
use runtime_integration::{MlxRuntime, MlxSafetensors};

use crate::artifact_validation::TensorSourceId;
use crate::kernel_capability;
use crate::qwen3_5_core::artifacts::{Qwen3_5ShardIndex, ValidatedQwen3_5Artifact};
use crate::qwen3_5_core::configuration::Qwen3_5Config;
use crate::qwen3_5_core::model_math::folded_ternary_weights::FoldedTernaryRotations;
use crate::qwen3_5_core::model_math::weights::{
    Qwen3_5Weights, WeightBindingDtype, bound_dtype_of,
};
use crate::qwen3_5_core::vision::{Qwen3_5VisionConfig, Qwen3_5VisionModel};
use crate::{PerformanceAttribution, PerformanceOperation, TensorInventory};
use text_generation::execution_error::TextExecutionError;

/// The bound language artifact the engine constructs before it builds the
/// expert pager.
pub struct BoundLanguageArtifact {
    pub config: Qwen3_5Config,
    pub shard_index: Qwen3_5ShardIndex,
    pub tensor_inventory: TensorInventory,
    pub weights: Qwen3_5Weights,
    pub vision_model: Option<Qwen3_5VisionModel>,
    /// A folded ternary pack's rotations, already applied to `weights`, for
    /// the layers the dense layer streamer reads later; `None` otherwise.
    pub folded_ternary_rotations: Option<Rc<FoldedTernaryRotations>>,
}

/// Binds the language artifact with every tensor exactly as stored, as the
/// engine needs (see [`bind_language_artifact_for`]).
pub fn bind_language_artifact(
    runtime: &MlxRuntime,
    validated_artifact: &mut ValidatedQwen3_5Artifact,
    should_bind_vision_weights: bool,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<BoundLanguageArtifact, TextExecutionError> {
    bind_language_artifact_for(
        runtime,
        validated_artifact,
        should_bind_vision_weights,
        WeightBindingDtype::ForComputeDevice,
        performance_attribution,
    )
}

/// How the artifact's vision tower, if any, reaches the bound model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum VisionBinding {
    /// The caller did not ask for vision weights, or the artifact has no tower.
    NotBound,
    /// The tower ships in its own sidecar file.
    SeparateSidecar,
    /// The tower's tensors sit in the language model shards.
    EmbeddedInModelShards,
}

impl VisionBinding {
    /// A sidecar wins: when the artifact ships one, it is bound from it
    /// whether or not the artifact also reports image input support.
    fn for_request(
        should_bind_vision_weights: bool,
        artifact_has_separate_vision_sidecar: bool,
        artifact_supports_image_input: bool,
    ) -> Self {
        if !should_bind_vision_weights {
            return Self::NotBound;
        }
        if artifact_has_separate_vision_sidecar {
            return Self::SeparateSidecar;
        }
        if artifact_supports_image_input {
            return Self::EmbeddedInModelShards;
        }
        Self::NotBound
    }
}

/// Binds the validated language artifact: extracts the configuration and
/// shard metadata, loads the optional sidecar vision model, maps and loads
/// the language shards, then binds the weights and (for embedded towers) the
/// vision model from those shards, including the attribution spans.
pub fn bind_language_artifact_for(
    runtime: &MlxRuntime,
    validated_artifact: &mut ValidatedQwen3_5Artifact,
    should_bind_vision_weights: bool,
    weight_binding_dtype: WeightBindingDtype,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<BoundLanguageArtifact, TextExecutionError> {
    let config = validated_artifact.config().clone();
    let hadamard_contract = validated_artifact.hadamard_contract().cloned();
    let vision_config = validated_artifact.vision_config().cloned();
    let vision_binding = VisionBinding::for_request(
        should_bind_vision_weights,
        validated_artifact.has_separate_vision_sidecar(),
        validated_artifact.supports_image_input(),
    );
    let tensor_inventory = validated_artifact.tensor_inventory().clone();
    let shard_index = validated_artifact.shard_index().clone();
    let model_shard_source_ids =
        validated_artifact.source_ids_for_file_names(shard_index.model_shard_file_names())?;
    let sidecar_vision_model = if vision_binding == VisionBinding::SeparateSidecar {
        performance_attribution.measure_operation(
            PerformanceOperation::ModelSafetensorsMapping,
            |_performance_attribution| -> Result<_, TextExecutionError> {
                Qwen3_5VisionModel::load_from_sidecar(runtime, validated_artifact)
            },
        )?
    } else {
        None
    };
    // An embedded tower reads its own lazy maps of the same files (see
    // `Qwen3_5VisionWeights`), so its descriptors are duplicated here before
    // the language loader takes ownership of the originals.
    let (model_shards, embedded_vision_shard_files) = performance_attribution.measure_operation(
        PerformanceOperation::ModelSafetensorsMapping,
        |performance_attribution| {
            load_model_shards(
                runtime,
                validated_artifact,
                &model_shard_source_ids,
                vision_binding,
                performance_attribution,
            )
        },
    )?;
    let ((weights, folded_ternary_rotations), vision_model) = performance_attribution
        .measure_operation(
            PerformanceOperation::ModelTensorBinding,
            |_performance_attribution| -> Result<_, TextExecutionError> {
                let vision_model = bind_vision_model(
                    runtime,
                    &config,
                    &shard_index,
                    vision_config.as_ref(),
                    vision_binding,
                    sidecar_vision_model,
                    embedded_vision_shard_files,
                )?;
                let bound_weights = bind_language_weights(
                    runtime,
                    &config,
                    (&shard_index, model_shards),
                    weight_binding_dtype,
                    hadamard_contract.as_ref(),
                )?;
                Ok((bound_weights, vision_model))
            },
        )?;
    Ok(BoundLanguageArtifact {
        config,
        shard_index,
        tensor_inventory,
        weights,
        vision_model,
        folded_ternary_rotations,
    })
}

/// Binds the language weights, and gives a folded ternary pack's modules
/// their rotations and float16 auxiliaries.
fn bind_language_weights(
    runtime: &MlxRuntime,
    config: &Qwen3_5Config,
    (shard_index, model_shards): (&Qwen3_5ShardIndex, Vec<MlxSafetensors>),
    weight_binding_dtype: WeightBindingDtype,
    hadamard_contract: Option<&crate::prism_hadamard_qwen35::hadamard::HadamardContract>,
) -> Result<(Qwen3_5Weights, Option<Rc<FoldedTernaryRotations>>), TextExecutionError> {
    let mut weights = Qwen3_5Weights::bind_from_model_shards(
        runtime,
        config,
        shard_index,
        model_shards,
        weight_binding_dtype,
    )?;
    let Some(hadamard_contract) = hadamard_contract else {
        return Ok((weights, None));
    };
    let folded_ternary_rotations =
        FoldedTernaryRotations::from_contract(runtime, hadamard_contract, config)?;
    folded_ternary_rotations.apply_to_model(runtime, &mut weights)?;
    Ok((weights, Some(Rc::new(folded_ternary_rotations))))
}

/// Takes the language shard files out of the validated artifact and maps each
/// one, returning the mapped shards and the duplicated descriptors. When the
/// vision tower is embedded, every descriptor is duplicated into the second
/// returned list first, because the mapped shard takes ownership of the
/// original; otherwise that list is empty.
fn load_model_shards(
    runtime: &MlxRuntime,
    validated_artifact: &mut ValidatedQwen3_5Artifact,
    model_shard_source_ids: &[TensorSourceId],
    vision_binding: VisionBinding,
    performance_attribution: &PerformanceAttribution,
) -> Result<(Vec<MlxSafetensors>, Vec<std::fs::File>), TextExecutionError> {
    let model_shard_files = validated_artifact.take_safetensors_sources(model_shard_source_ids)?;
    let mut model_shards = Vec::with_capacity(model_shard_files.len());
    let mut embedded_vision_shard_files = Vec::new();
    let positional_file_read_metrics = performance_attribution.positional_file_read_metrics();
    for model_shard_file in model_shard_files {
        let shard_source_file = model_shard_file.into_file();
        if vision_binding == VisionBinding::EmbeddedInModelShards {
            embedded_vision_shard_files.push(shard_source_file.try_clone().map_err(|source| {
                TextExecutionError::WeightSourceDescriptor {
                    description: "could not duplicate a model shard descriptor for the vision tower",
                    source,
                }
            })?);
        }
        model_shards.push(
            runtime.load_safetensors(
                shard_source_file,
                positional_file_read_metrics
                    .as_ref()
                    .map(std::sync::Arc::clone),
            )?,
        );
    }
    Ok((model_shards, embedded_vision_shard_files))
}

/// Produces the vision model the artifact binds: the sidecar's, one read from
/// the duplicated model shard descriptors, or none.
fn bind_vision_model(
    runtime: &MlxRuntime,
    config: &Qwen3_5Config,
    shard_index: &Qwen3_5ShardIndex,
    vision_config: Option<&Qwen3_5VisionConfig>,
    vision_binding: VisionBinding,
    sidecar_vision_model: Option<Qwen3_5VisionModel>,
    embedded_vision_shard_files: Vec<std::fs::File>,
) -> Result<Option<Qwen3_5VisionModel>, TextExecutionError> {
    match vision_binding {
        VisionBinding::SeparateSidecar => Ok(sidecar_vision_model),
        VisionBinding::EmbeddedInModelShards => {
            let vision_config = vision_config.ok_or(TextExecutionError::InvalidInput {
                description: "validated visual tensors have no vision configuration",
            })?;
            Ok(Some(Qwen3_5VisionModel::load_from_model_shard_files(
                runtime,
                vision_config,
                config,
                embedded_vision_shard_files,
                build_vision_tensor_shard_map(shard_index),
            )?))
        }
        VisionBinding::NotBound => Ok(None),
    }
}

/// Maps embedded canonical vision tensor names to their loaded model-shard
/// position and stored tensor spelling for physical reads.
pub fn build_vision_tensor_shard_map(
    shard_index: &Qwen3_5ShardIndex,
) -> HashMap<String, crate::qwen3_5_core::vision::VisionTensorLocation> {
    let mut vision_tensor_name_to_shard_location = HashMap::new();
    for (vision_tensor_name, shard_file_name) in shard_index.vision_tensor_name_to_shard_file_name()
    {
        let mut shard_position_option = None;
        for (candidate_shard_position, expected_file_name) in
            shard_index.model_shard_file_names().iter().enumerate()
        {
            if expected_file_name == shard_file_name {
                shard_position_option = Some(candidate_shard_position);
                break;
            }
        }
        if let Some(shard_position) = shard_position_option {
            vision_tensor_name_to_shard_location.insert(
                vision_tensor_name.clone(),
                crate::qwen3_5_core::vision::VisionTensorLocation {
                    source_file_position: shard_position,
                    stored_tensor_name: shard_index.stored_tensor_name_for(vision_tensor_name),
                },
            );
        }
    }
    vision_tensor_name_to_shard_location
}

/// The capability-verdict kernels the engine loads after binding the
/// artifact.
pub struct ModelKernelVerdicts {
    pub gated_delta_kernel: Option<MlxMetalKernel>,
    pub gated_delta_checkpoint_kernel: Option<MlxMetalKernel>,
    pub gdn_decode_prework_kernel: Option<MlxMetalKernel>,
}

/// Loads the three gated-delta-family kernels under their GPU capability
/// verdicts, including the demotion logs.
pub fn load_model_kernel_verdicts(
    runtime: &MlxRuntime,
    config: &Qwen3_5Config,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<ModelKernelVerdicts, TextExecutionError> {
    // The two gated-delta kernels are independent capability families. A
    // demoted kernel is None and the gated-delta dispatch falls back to
    // the ops-based public MLX route for the worker's lifetime — the
    // checkpoint fallback preserves the astro SSD prompt cache boundary contract.
    let gated_delta_kernel =
        if kernel_capability::worker_process_kernel_capabilities(runtime, performance_attribution)
            .is_custom_kernel_supported(crate::kernel_capability::GATED_DELTA_SEQUENCE)
        {
            Some(
                crate::qwen3_5_core::model_math::gated_delta_sequence::qwen3_5_gated_delta_kernel(
                )?,
            )
        } else {
            tracing::info!(
                "fused gated-delta kernel demoted to the MLX ops fallback for this worker process"
            );
            None
        };
    let gated_delta_checkpoint_kernel = if kernel_capability::worker_process_kernel_capabilities(
        runtime,
        performance_attribution,
    )
    .is_custom_kernel_supported(crate::kernel_capability::GATED_DELTA_BOUNDARY_CHECKPOINT)
    {
        Some(
                crate::qwen3_5_core::model_math::gated_delta_boundary_checkpoints::qwen3_5_gated_delta_checkpoint_kernel(
                )?,
            )
    } else {
        tracing::info!(
            "gated-delta checkpoint kernel demoted to the MLX ops fallback for this worker process"
        );
        None
    };
    // The fused decode prework kernel removes roughly thirteen
    // launch-bound dispatches per gated-delta layer at decode shapes.
    let gdn_decode_prework_kernel = if kernel_capability::worker_process_kernel_capabilities(
        runtime,
        performance_attribution,
    )
    .is_custom_kernel_supported(crate::kernel_capability::GDN_DECODE_PREWORK)
    {
        Some(
            crate::qwen3_5_core::model_math::gdn_decode_prework_kernel::qwen3_5_gdn_decode_prework_kernel(
                f32::from_bits(config.rms_norm_epsilon_bits()),
            )?,
        )
    } else {
        tracing::info!(
            "fused decode prework kernel demoted to the MLX ops fallback for this worker process"
        );
        None
    };
    Ok(ModelKernelVerdicts {
        gated_delta_kernel,
        gated_delta_checkpoint_kernel,
        gdn_decode_prework_kernel,
    })
}

/// The invariant linear-attention normalization scales the engine constructs
/// at load time.
pub struct InvariantLinearScales {
    pub inverse_linear_head_dimension_scale: mlx_bridge::MlxArray,
    pub inverse_square_root_linear_head_dimension_scale: mlx_bridge::MlxArray,
    pub query_normalization_scale_weight: mlx_bridge::MlxArray,
    pub key_normalization_scale_weight: mlx_bridge::MlxArray,
}

/// Constructs the invariant linear-attention scales, in the dtype the model's
/// activations are stored in (`stored_activation_dtype` of its weights):
/// BF16 for every Qwen 3.5 checkpoint, float16 for a folded ternary pack. The
/// scales must share the activations' dtype: a BF16 scale weight beside
/// float16 queries would widen the queries and keys to float32 while the
/// values stay float16, and the gated-delta kernel, typed by its queries,
/// would then read the values as float32. The two scalar scales and the two
/// folded per-channel weights are built once instead of rebuilding their
/// scalar and cast graphs for every linear-attention forward pass. The prefill
/// composed path folds each scale into `fast_rms_norm`'s per-channel weight so
/// one fused launch replaces the norm kernel plus the broadcast-multiply
/// kernel (issue #915 item 5). Every channel carries the same f32 value through
/// the same rounding to `scale_dtype` as the scalar scales: BF16 or float16
/// as stored, or float32 where the engine binds for a CPU compute device and
/// the fused decode-prework kernel is demoted anyway (#2441).
pub fn load_invariant_linear_scales(
    runtime: &MlxRuntime,
    config: &Qwen3_5Config,
    stored_activation_dtype: MlxDtype,
    weight_binding_dtype: WeightBindingDtype,
) -> Result<InvariantLinearScales, TextExecutionError> {
    invariant_linear_scales_in(
        runtime,
        config,
        bound_dtype_of(stored_activation_dtype, weight_binding_dtype)?,
    )
}

/// The dtype the model's activations are stored in: what its embedding
/// table hands the first layer (the scales of a quantized table, the table
/// itself otherwise). A codebook never holds embeddings, so it reads BF16.
#[must_use]
pub fn stored_activation_dtype(weights: &Qwen3_5Weights) -> MlxDtype {
    use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AffineWeights;
    match &weights.embedding_weights {
        Qwen3_5AffineWeights::Quantized {
            quantization_scales,
            ..
        } => quantization_scales.dtype(),
        Qwen3_5AffineWeights::NativeBfloat16 { weight } => weight.dtype(),
        Qwen3_5AffineWeights::Jangtq2Codebook { .. } => MlxDtype::BFloat16,
    }
}

fn invariant_linear_scales_in(
    runtime: &MlxRuntime,
    config: &Qwen3_5Config,
    scale_dtype: MlxDtype,
) -> Result<InvariantLinearScales, TextExecutionError> {
    let linear_head_dimension = config.linear_key_head_dimension() as f32;
    let inverse_linear_head_dimension_scale = runtime
        .array_from_f32(&[linear_head_dimension.recip()], &[])
        .and_then(|float32_scale| runtime.astype(&float32_scale, scale_dtype))?;
    let inverse_square_root_linear_head_dimension_scale = runtime
        .array_from_f32(&[linear_head_dimension.sqrt().recip()], &[])
        .and_then(|float32_scale| runtime.astype(&float32_scale, scale_dtype))?;
    let head_dimension_i32 = config.linear_key_head_dimension() as i32;
    let query_normalization_scale_weight = runtime
        .array_from_f32(
            &vec![linear_head_dimension.recip(); head_dimension_i32 as usize],
            &[head_dimension_i32],
        )
        .and_then(|float32_weight| runtime.astype(&float32_weight, scale_dtype))?;
    let key_normalization_scale_weight = runtime
        .array_from_f32(
            &vec![linear_head_dimension.sqrt().recip(); head_dimension_i32 as usize],
            &[head_dimension_i32],
        )
        .and_then(|float32_weight| runtime.astype(&float32_weight, scale_dtype))?;
    Ok(InvariantLinearScales {
        inverse_linear_head_dimension_scale,
        inverse_square_root_linear_head_dimension_scale,
        query_normalization_scale_weight,
        key_normalization_scale_weight,
    })
}

/// The model-core payload the engine budgets: the text weights
/// that load into memory plus the optional vision tower while it is in use.
pub fn model_core_payload_bytes(
    weights: &Qwen3_5Weights,
    embedding_residency: crate::qwen3_5_core::model_math::weights::EmbeddingResidency,
    vision_model: &Option<Qwen3_5VisionModel>,
) -> u64 {
    weights
        .resident_payload_bytes(embedding_residency)
        .saturating_add(
            vision_model
                .as_ref()
                .map_or(0, Qwen3_5VisionModel::resident_payload_bytes),
        )
}

#[cfg(test)]
#[path = "loading_pipeline_tests.rs"]
mod tests;
