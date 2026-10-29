//! The shared artifact-loading pipeline both Qwen3.5 engines run: language
//! weight and vision binding, the GPU kernel capability verdicts, and the
//! invariant linear-attention scales. Split from the streaming and resident
//! loaders so the two engines cannot drift on load semantics.

use std::collections::HashMap;

use astronomical_runtime_integration::MlxRuntime;
use mlx_c_rust::{MlxDtype, MlxMetalKernel};

use crate::kernel_capability;
use crate::kernel_capability::CustomMetalKernelFamily;
use crate::qwen3_5_core::artifacts::{Qwen3_5ShardIndex, ValidatedQwen3_5Artifact};
use crate::qwen3_5_core::configuration::Qwen3_5Config;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_core::model_math::weights::Qwen3_5Weights;
use crate::qwen3_5_core::vision::Qwen3_5VisionModel;
use crate::{PerformanceAttribution, PerformanceOperation, TensorInventory};

/// The bound language artifact both engines construct identically before their
/// expert-handling paths diverge (streaming builds a pager; resident builds
/// quantized layer plans).
pub(crate) struct BoundLanguageArtifact {
    pub config: Qwen3_5Config,
    pub shard_index: Qwen3_5ShardIndex,
    pub tensor_inventory: TensorInventory,
    pub weights: Qwen3_5Weights,
    pub vision_model: Option<Qwen3_5VisionModel>,
}

/// Binds the validated language artifact: extracts the configuration and
/// shard metadata, loads the optional sidecar vision model, maps and loads
/// the language shards, then binds the weights and (for embedded towers) the
/// vision model from those shards. Mirrors the inlined preamble of both
/// engines' loaders expression-for-expression, including the attribution
/// spans.
pub(crate) fn bind_language_artifact(
    runtime: &MlxRuntime,
    validated_artifact: &mut ValidatedQwen3_5Artifact,
    should_bind_vision_weights: bool,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<BoundLanguageArtifact, Qwen3_5ExecutionError> {
    let config = validated_artifact.config().clone();
    let vision_config = validated_artifact.vision_config().cloned();
    let has_separate_vision_sidecar =
        should_bind_vision_weights && validated_artifact.has_separate_vision_sidecar();
    let has_embedded_vision_tower = should_bind_vision_weights
        && validated_artifact.supports_image_input()
        && !has_separate_vision_sidecar;
    let tensor_inventory = validated_artifact.tensor_inventory().clone();
    let shard_index = validated_artifact.shard_index().clone();
    let model_shard_source_ids =
        validated_artifact.source_ids_for_file_names(shard_index.model_shard_file_names())?;
    let sidecar_vision_model = if has_separate_vision_sidecar {
        performance_attribution.measure_operation(
            PerformanceOperation::ModelSafetensorsMapping,
            |_performance_attribution| -> Result<_, Qwen3_5ExecutionError> {
                Qwen3_5VisionModel::load_from_sidecar(runtime, validated_artifact)
            },
        )?
    } else {
        None
    };
    let model_shards = performance_attribution.measure_operation(
        PerformanceOperation::ModelSafetensorsMapping,
        |performance_attribution| -> Result<_, Qwen3_5ExecutionError> {
            let model_shard_files =
                validated_artifact.take_safetensors_sources(&model_shard_source_ids)?;
            let mut model_shards = Vec::with_capacity(model_shard_files.len());
            let positional_file_read_metrics =
                performance_attribution.positional_file_read_metrics();
            for model_shard_file in model_shard_files {
                let shard_source_file = model_shard_file.into_file();
                model_shards.push(
                    runtime.load_safetensors(
                        shard_source_file,
                        positional_file_read_metrics
                            .as_ref()
                            .map(std::sync::Arc::clone),
                    )?,
                );
            }
            Ok(model_shards)
        },
    )?;
    let (weights, vision_model) = performance_attribution.measure_operation(
        PerformanceOperation::ModelTensorBinding,
        |_performance_attribution| -> Result<_, Qwen3_5ExecutionError> {
            let vision_model = if has_separate_vision_sidecar {
                sidecar_vision_model
            } else if has_embedded_vision_tower {
                let vision_config =
                    vision_config
                        .as_ref()
                        .ok_or(Qwen3_5ExecutionError::InvalidInput {
                            description: "validated visual tensors have no vision configuration",
                        })?;
                let vision_tensor_name_to_shard_location =
                    build_vision_tensor_shard_map(&shard_index);
                Some(Qwen3_5VisionModel::load_from_model_shards(
                    runtime,
                    vision_config,
                    &config,
                    &model_shards,
                    &vision_tensor_name_to_shard_location,
                )?)
            } else {
                None
            };
            let weights = Qwen3_5Weights::bind_from_model_shards(
                runtime,
                &config,
                &shard_index,
                model_shards,
            )?;
            Ok((weights, vision_model))
        },
    )?;
    Ok(BoundLanguageArtifact {
        config,
        shard_index,
        tensor_inventory,
        weights,
        vision_model,
    })
}

/// Maps embedded canonical vision tensor names to their loaded model-shard
/// position and stored tensor spelling for physical reads.
pub(crate) fn build_vision_tensor_shard_map(
    shard_index: &Qwen3_5ShardIndex,
) -> HashMap<String, crate::qwen3_5_core::vision::EmbeddedVisionTensorLocation> {
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
                crate::qwen3_5_core::vision::EmbeddedVisionTensorLocation {
                    shard_position,
                    stored_tensor_name: shard_index.stored_tensor_name_for(vision_tensor_name),
                },
            );
        }
    }
    vision_tensor_name_to_shard_location
}

/// The capability-verdict kernels both engines load identically after their
/// expert paths diverge.
pub(crate) struct ModelKernelVerdicts {
    pub gated_delta_kernel: Option<MlxMetalKernel>,
    pub gated_delta_checkpoint_kernel: Option<MlxMetalKernel>,
    pub gdn_decode_prework_kernel: Option<MlxMetalKernel>,
}

/// Loads the three gated-delta-family kernels under their GPU capability
/// verdicts. Mirrors the inlined verdict blocks of both engines' loaders,
/// including the demotion logs.
pub(crate) fn load_model_kernel_verdicts(
    runtime: &MlxRuntime,
    config: &Qwen3_5Config,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<ModelKernelVerdicts, Qwen3_5ExecutionError> {
    // The two gated-delta kernels are independent capability families. A
    // demoted kernel is None and the gated-delta dispatch falls back to
    // the ops-based public MLX route for the worker's lifetime — the
    // checkpoint fallback preserves the prompt-cache boundary contract.
    let gated_delta_kernel =
        if kernel_capability::worker_process_kernel_capabilities(runtime, performance_attribution)
            .is_custom_kernel_supported(CustomMetalKernelFamily::GATED_DELTA_SEQUENCE)
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
    .is_custom_kernel_supported(CustomMetalKernelFamily::GATED_DELTA_BOUNDARY_CHECKPOINT)
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
    .is_custom_kernel_supported(CustomMetalKernelFamily::GDN_DECODE_PREWORK)
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

/// The invariant linear-attention normalization scales both engines construct
/// identically at load time.
pub(crate) struct InvariantLinearScales {
    pub inverse_linear_head_dimension_scale: mlx_c_rust::MlxArray,
    pub inverse_square_root_linear_head_dimension_scale: mlx_c_rust::MlxArray,
    pub query_normalization_scale_weight: mlx_c_rust::MlxArray,
    pub key_normalization_scale_weight: mlx_c_rust::MlxArray,
}

/// Constructs the invariant linear-attention scales. Qwen3.5 config validation
/// accepts BF16 activations only, so the two scalar scales and the two folded
/// per-channel weights are built once instead of rebuilding their scalar and
/// cast graphs for every linear-attention forward pass. The prefill composed
/// path folds each scale into `fast_rms_norm`'s per-channel weight so one
/// fused launch replaces the norm kernel plus the broadcast-multiply kernel
/// (issue #915 item 5). Every channel carries the same f32 value through the
/// same BF16 rounding as the scalar scales, which the fused decode-prework
/// kernel still requires.
pub(crate) fn load_invariant_linear_scales(
    runtime: &MlxRuntime,
    config: &Qwen3_5Config,
) -> Result<InvariantLinearScales, Qwen3_5ExecutionError> {
    let linear_head_dimension = config.linear_key_head_dimension() as f32;
    let inverse_linear_head_dimension_scale = runtime
        .array_from_f32(&[linear_head_dimension.recip()], &[])
        .and_then(|float32_scale| runtime.astype(&float32_scale, MlxDtype::BFloat16))?;
    let inverse_square_root_linear_head_dimension_scale = runtime
        .array_from_f32(&[linear_head_dimension.sqrt().recip()], &[])
        .and_then(|float32_scale| runtime.astype(&float32_scale, MlxDtype::BFloat16))?;
    let head_dimension_i32 = config.linear_key_head_dimension() as i32;
    let query_normalization_scale_weight = runtime
        .array_from_f32(
            &vec![linear_head_dimension.recip(); head_dimension_i32 as usize],
            &[head_dimension_i32],
        )
        .and_then(|float32_weight| runtime.astype(&float32_weight, MlxDtype::BFloat16))?;
    let key_normalization_scale_weight = runtime
        .array_from_f32(
            &vec![linear_head_dimension.sqrt().recip(); head_dimension_i32 as usize],
            &[head_dimension_i32],
        )
        .and_then(|float32_weight| runtime.astype(&float32_weight, MlxDtype::BFloat16))?;
    Ok(InvariantLinearScales {
        inverse_linear_head_dimension_scale,
        inverse_square_root_linear_head_dimension_scale,
        query_normalization_scale_weight,
        key_normalization_scale_weight,
    })
}

/// The model-core payload both engines budget identically: language weights
/// plus the optional resident vision tower.
pub(crate) fn model_core_payload_bytes(
    weights: &Qwen3_5Weights,
    vision_model: &Option<Qwen3_5VisionModel>,
) -> u64 {
    weights.total_payload_bytes().saturating_add(
        vision_model
            .as_ref()
            .map_or(0, Qwen3_5VisionModel::resident_payload_bytes),
    )
}
