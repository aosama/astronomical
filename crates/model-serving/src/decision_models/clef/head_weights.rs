//! Clef's joint-head weights, loaded once and widened to float32.
//!
//! The release stores the head in bfloat16 under PyTorch's own names. The
//! reference runs it in the backbone's dtype; this engine runs it in float32
//! instead, so the head adds no rounding of its own to the comparison with a
//! float32 reference. The head is about 120 million weights, 480 MB widened,
//! small beside the 9B backbone. Each attention's packed input projection is
//! split into its query, key and value parts here, once.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use astronomical_decision_contract::DecisionFailure;
use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::{MlxArray, MlxDtype};

use super::head_settings::ClefHeadSettings;

pub(crate) const CLEF_HEAD_WEIGHTS_FILE_NAME: &str = "joint_head.safetensors";
/// The suffixes of an attention's packed input projection and its parts.
const PACKED_INPUT_PROJECTION_NAMES: [(&str, [&str; 3]); 2] = [
    (
        "in_proj_weight",
        ["query_weight", "key_weight", "value_weight"],
    ),
    ("in_proj_bias", ["query_bias", "key_bias", "value_bias"]),
];

/// The head's tensors outside its layers, as `JointSchemaHead` names them.
const HEAD_TENSOR_NAMES: [&str; 22] = [
    "hidden_norm.weight",
    "hidden_norm.bias",
    "memory_projection.weight",
    "question_projection.weight",
    "option_question_projection.weight",
    "global_projection.weight",
    "option_context_projection.weight",
    "option_lexical_projection.weight",
    "type_embedding.weight",
    "option_summary_norm.weight",
    "option_summary_norm.bias",
    "field_norm.weight",
    "field_norm.bias",
    "option_norm.weight",
    "option_norm.bias",
    "residual_scorer.0.weight",
    "residual_scorer.0.bias",
    "residual_scorer.3.weight",
    "residual_scorer.3.bias",
    "prior_logit_scale",
    "joint_logit_scale",
    "residual_gate",
];
/// Each `EvidenceRoutingLayer`'s tensors, after its prefix.
const ROUTING_LAYER_TENSOR_NAMES: [&str; 14] = [
    "query_norm.weight",
    "query_norm.bias",
    "memory_norm.weight",
    "memory_norm.bias",
    "attention.in_proj_weight",
    "attention.in_proj_bias",
    "attention.out_proj.weight",
    "attention.out_proj.bias",
    "feedforward_norm.weight",
    "feedforward_norm.bias",
    "feedforward.0.weight",
    "feedforward.0.bias",
    "feedforward.3.weight",
    "feedforward.3.bias",
];
/// Each `TransformerDecoderLayer`'s tensors, after its prefix.
const DECODER_LAYER_TENSOR_NAMES: [&str; 18] = [
    "norm1.weight",
    "norm1.bias",
    "norm2.weight",
    "norm2.bias",
    "norm3.weight",
    "norm3.bias",
    "self_attn.in_proj_weight",
    "self_attn.in_proj_bias",
    "self_attn.out_proj.weight",
    "self_attn.out_proj.bias",
    "multihead_attn.in_proj_weight",
    "multihead_attn.in_proj_bias",
    "multihead_attn.out_proj.weight",
    "multihead_attn.out_proj.bias",
    "linear1.weight",
    "linear1.bias",
    "linear2.weight",
    "linear2.bias",
];

/// Every tensor a head of `head_settings`' shape holds.
pub(crate) fn head_tensor_names(head_settings: &ClefHeadSettings) -> Vec<String> {
    let mut tensor_names = Vec::new();
    for tensor_name in HEAD_TENSOR_NAMES {
        tensor_names.push(tensor_name.to_owned());
    }
    for routing_layer_index in 0..head_settings.routing_layer_count {
        for tensor_suffix in ROUTING_LAYER_TENSOR_NAMES {
            tensor_names.push(format!(
                "evidence_layers.{routing_layer_index}.{tensor_suffix}"
            ));
        }
    }
    for decoder_layer_index in 0..head_settings.decoder_layer_count {
        for tensor_suffix in DECODER_LAYER_TENSOR_NAMES {
            tensor_names.push(format!("layers.{decoder_layer_index}.{tensor_suffix}"));
        }
    }
    tensor_names
}

/// Every head tensor by name, in float32.
pub(crate) struct ClefHeadWeights {
    tensors_by_name: BTreeMap<String, MlxArray>,
}

impl ClefHeadWeights {
    pub(crate) fn read(
        runtime: &MlxRuntime,
        model_directory: &Path,
        head_settings: &ClefHeadSettings,
    ) -> Result<Self, DecisionFailure> {
        let weights_file = File::open(model_directory.join(CLEF_HEAD_WEIGHTS_FILE_NAME))
            .map_err(|_| unusable_head("joint_head.safetensors cannot be opened"))?;
        let stored_tensors = runtime
            .load_safetensors(weights_file, None)
            .map_err(|load_error| unusable_head(&load_error.to_string()))?;
        let mut tensors_by_name = BTreeMap::new();
        for tensor_name in head_tensor_names(head_settings) {
            let stored_tensor = stored_tensors
                .tensor(&tensor_name)
                .map_err(|load_error| unusable_head(&load_error.to_string()))?;
            let float32_tensor = runtime
                .astype(&stored_tensor, MlxDtype::Float32)
                .map_err(|cast_error| unusable_head(&cast_error.to_string()))?;
            insert_split_projection(runtime, &mut tensors_by_name, tensor_name, float32_tensor)?;
        }
        let mut loaded_tensors = Vec::with_capacity(tensors_by_name.len());
        for loaded_tensor in tensors_by_name.values() {
            loaded_tensors.push(loaded_tensor);
        }
        runtime
            .evaluate_arrays(&loaded_tensors)
            .map_err(|evaluate_error| unusable_head(&evaluate_error.to_string()))?;
        Ok(Self { tensors_by_name })
    }

    pub(crate) fn tensor(&self, tensor_name: &str) -> Result<&MlxArray, DecisionFailure> {
        self.tensors_by_name
            .get(tensor_name)
            .ok_or_else(|| unusable_head(&format!("it has no tensor {tensor_name}")))
    }

    /// A scalar parameter's value, read once to the host.
    pub(crate) fn scalar(&self, tensor_name: &str) -> Result<f64, DecisionFailure> {
        let scalar_values = self
            .tensor(tensor_name)?
            .to_vec_f32()
            .map_err(|read_error| unusable_head(&read_error.to_string()))?;
        match scalar_values.as_slice() {
            [scalar_value] => Ok(f64::from(*scalar_value)),
            _ => Err(unusable_head(&format!("{tensor_name} is not a scalar"))),
        }
    }
}

/// Stores `tensor` under its name, or, for a packed attention input
/// projection, its three equal parts under their own names.
fn insert_split_projection(
    runtime: &MlxRuntime,
    tensors_by_name: &mut BTreeMap<String, MlxArray>,
    tensor_name: String,
    tensor: MlxArray,
) -> Result<(), DecisionFailure> {
    for (packed_suffix, part_suffixes) in PACKED_INPUT_PROJECTION_NAMES {
        let Some(attention_prefix) = tensor_name.strip_suffix(packed_suffix) else {
            continue;
        };
        let packed_shape = tensor.shape();
        let part_rows = packed_shape[0] / 3;
        for (part_index, part_suffix) in part_suffixes.iter().enumerate() {
            let mut starts = vec![0; packed_shape.len()];
            let mut stops = packed_shape.clone();
            starts[0] = part_rows * part_index as i32;
            stops[0] = part_rows * (part_index as i32 + 1);
            let strides = vec![1; packed_shape.len()];
            let part = runtime
                .slice(&tensor, &starts, &stops, &strides)
                .map_err(|slice_error| unusable_head(&slice_error.to_string()))?;
            tensors_by_name.insert(format!("{attention_prefix}{part_suffix}"), part);
        }
        return Ok(());
    }
    tensors_by_name.insert(tensor_name, tensor);
    Ok(())
}

pub(crate) fn unusable_head(reason: &str) -> DecisionFailure {
    DecisionFailure::fatal_execution(&format!("Clef's joint head cannot be used: {reason}"))
}
