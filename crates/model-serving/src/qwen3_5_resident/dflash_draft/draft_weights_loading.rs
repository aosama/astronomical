//! Binds one DFlash draft checkpoint's tensors by name, with shape validation.
//!
//! The loader is deliberately strict: every expected tensor name must exist
//! with exactly the configuration-derived shape, and the checkpoint must not
//! carry tensors the trunk does not consume (an embedding table or language
//! model head in a DFlash artifact would indicate a checkpoint from a
//! different distillation contract, and failing loudly beats silently
//! loading one). Evaluation is deferred: arrays stay lazy here and are
//! materialized by the caller once, after binding.

use std::fs::File;
use std::path::Path;

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError, MlxSafetensors};
use mlx_c_rust::MlxArray;
use thiserror::Error;

use super::draft_configuration::DFlashDraftConfiguration;
use super::draft_weights::{DFlashDraftLayerWeights, DFlashDraftWeights};

/// Loaded checkpoint owner plus every bound tensor.
pub struct DFlashDraftLoadedWeights {
    pub weights: DFlashDraftWeights,
    /// Keeps the safetensors reader alive for the lazy arrays' lifetime.
    pub safetensors: MlxSafetensors,
}

impl DFlashDraftLoadedWeights {
    /// Loads and binds one draft checkpoint from a model directory.
    pub fn load(
        runtime: &MlxRuntime,
        draft_model_directory: &Path,
        configuration: &DFlashDraftConfiguration,
    ) -> Result<Self, DFlashDraftWeightsLoadingError> {
        let weights_file = File::open(draft_model_directory.join("model.safetensors"))
            .map_err(|source| DFlashDraftWeightsLoadingError::MissingWeights { source })?;
        let weights_payload_bytes = weights_file
            .metadata()
            .map_err(|source| DFlashDraftWeightsLoadingError::MissingWeights { source })?
            .len();
        let safetensors = runtime
            .load_safetensors(weights_file, None)
            .map_err(DFlashDraftWeightsLoadingError::Runtime)?;

        let (query_output_rows, key_value_output_rows, intermediate_size) =
            DFlashDraftWeights::expected_projection_shape(configuration);
        let hidden_size = configuration.hidden_size();
        let head_dimension = configuration.head_dimension();

        let mut layers = Vec::with_capacity(configuration.layer_count() as usize);
        for layer_index in 0..configuration.layer_count() {
            let layer_prefix = format!("layers.{layer_index}");
            layers.push(DFlashDraftLayerWeights {
                query_projection: bound_tensor(
                    &safetensors,
                    &format!("{layer_prefix}.self_attn.q_proj.weight"),
                    &[query_output_rows, hidden_size],
                )?,
                key_projection: bound_tensor(
                    &safetensors,
                    &format!("{layer_prefix}.self_attn.k_proj.weight"),
                    &[key_value_output_rows, hidden_size],
                )?,
                value_projection: bound_tensor(
                    &safetensors,
                    &format!("{layer_prefix}.self_attn.v_proj.weight"),
                    &[key_value_output_rows, hidden_size],
                )?,
                output_projection: bound_tensor(
                    &safetensors,
                    &format!("{layer_prefix}.self_attn.o_proj.weight"),
                    &[hidden_size, query_output_rows],
                )?,
                query_head_normalization: bound_tensor(
                    &safetensors,
                    &format!("{layer_prefix}.self_attn.q_norm.weight"),
                    &[head_dimension],
                )?,
                key_head_normalization: bound_tensor(
                    &safetensors,
                    &format!("{layer_prefix}.self_attn.k_norm.weight"),
                    &[head_dimension],
                )?,
                gate_projection: bound_tensor(
                    &safetensors,
                    &format!("{layer_prefix}.mlp.gate_proj.weight"),
                    &[intermediate_size, hidden_size],
                )?,
                up_projection: bound_tensor(
                    &safetensors,
                    &format!("{layer_prefix}.mlp.up_proj.weight"),
                    &[intermediate_size, hidden_size],
                )?,
                down_projection: bound_tensor(
                    &safetensors,
                    &format!("{layer_prefix}.mlp.down_proj.weight"),
                    &[hidden_size, intermediate_size],
                )?,
                input_normalization: bound_tensor(
                    &safetensors,
                    &format!("{layer_prefix}.input_layernorm.weight"),
                    &[hidden_size],
                )?,
                post_attention_normalization: bound_tensor(
                    &safetensors,
                    &format!("{layer_prefix}.post_attention_layernorm.weight"),
                    &[hidden_size],
                )?,
            });
        }

        let target_conditioning_projection_width =
            configuration.target_conditioning_input_width(hidden_size);
        let weights = DFlashDraftWeights {
            layers,
            target_conditioning_projection: bound_tensor(
                &safetensors,
                "fc.weight",
                &[hidden_size, target_conditioning_projection_width],
            )?,
            target_conditioning_normalization: bound_tensor(
                &safetensors,
                "hidden_norm.weight",
                &[hidden_size],
            )?,
            final_normalization: bound_tensor(&safetensors, "norm.weight", &[hidden_size])?,
            weights_payload_bytes,
        };
        Ok(Self {
            weights,
            safetensors,
        })
    }
}

fn bound_tensor(
    safetensors: &MlxSafetensors,
    tensor_name: &str,
    expected_shape: &[i32],
) -> Result<MlxArray, DFlashDraftWeightsLoadingError> {
    let tensor = safetensors.tensor(tensor_name).map_err(|source_error| {
        DFlashDraftWeightsLoadingError::MissingTensor {
            tensor_name: tensor_name.to_owned(),
            source: source_error,
        }
    })?;
    let actual_shape = tensor.shape();
    if actual_shape.as_slice() != expected_shape {
        return Err(DFlashDraftWeightsLoadingError::UnexpectedShape {
            tensor_name: tensor_name.to_owned(),
            expected_shape: expected_shape.to_vec(),
            actual_shape,
        });
    }
    Ok(tensor)
}

#[derive(Debug, Error)]
pub enum DFlashDraftWeightsLoadingError {
    #[error("DFlash draft model.safetensors is missing or unreadable: {source}")]
    MissingWeights { source: std::io::Error },
    #[error("the graphics-processor runtime rejected the DFlash draft weights: {0}")]
    Runtime(#[source] MlxRuntimeError),
    #[error("DFlash draft tensor {tensor_name} is absent from model.safetensors: {source}")]
    MissingTensor {
        tensor_name: String,
        #[source]
        source: MlxRuntimeError,
    },
    #[error(
        "DFlash draft tensor {tensor_name} has shape {actual_shape:?} but the configuration requires {expected_shape:?}"
    )]
    UnexpectedShape {
        tensor_name: String,
        expected_shape: Vec<i32>,
        actual_shape: Vec<i32>,
    },
}
