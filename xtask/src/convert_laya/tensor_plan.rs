//! What becomes of each tensor of a PyTorch Laya checkpoint.
//!
//! The MLX layout differs from the PyTorch one in three ways: a few names
//! (PyTorch's packed attention projection and its numbered `Sequential`
//! children), the linear layers of the encoder and the decision head being
//! quantized, and the temperatures living in `config.json` instead of in a
//! tensor.

use super::affine_quantization::{QUANTIZATION_BITS, QUANTIZATION_GROUP_SIZE};
use super::safetensors_file::{PlannedTensor, TensorElementType, TensorRecord};

/// Quantized weights packed into one `u32`.
const QUANTIZED_WEIGHTS_PER_PACKED_WORD: u64 = (32 / QUANTIZATION_BITS) as u64;

/// What the conversion does with one source tensor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum TensorTreatment {
    /// Not part of the MLX layout.
    Dropped,
    /// Written under `output_name` as half-precision floats.
    KeptAsHalfPrecision { output_name: String },
    /// Quantized into a packed weight, scales and biases under `module_name`.
    Quantized { module_name: String },
}

/// The treatment of the source tensor named `source_name` with `shape`.
pub(crate) fn tensor_treatment(source_name: &str, shape: &[u64]) -> TensorTreatment {
    // The calibration temperatures are written to config.json, where the
    // engine reads them together with the per-option-count table.
    if source_name == "temperature" {
        return TensorTreatment::Dropped;
    }
    let output_name = mlx_tensor_name(source_name);
    if let Some(module_name) = output_name.strip_suffix(".weight")
        && shape.len() == 2
        && is_quantized_linear_module(module_name)
    {
        return TensorTreatment::Quantized {
            module_name: module_name.to_owned(),
        };
    }
    TensorTreatment::KeptAsHalfPrecision { output_name }
}

/// The MLX name of a PyTorch tensor.
fn mlx_tensor_name(source_name: &str) -> String {
    // PyTorch's multi-head attention keeps its input projection as two bare
    // parameters; MLX keeps it as a linear layer named `in_proj`.
    let renamed_projection = source_name
        .replace(".in_proj_weight", ".in_proj.weight")
        .replace(".in_proj_bias", ".in_proj.bias");
    // MLX's `Sequential` holds its children under `layers`.
    for sequential_module in ["scorer", "act_head"] {
        if let Some(child_path) = renamed_projection
            .strip_prefix(sequential_module)
            .and_then(|after_module| after_module.strip_prefix('.'))
        {
            return format!("{sequential_module}.layers.{child_path}");
        }
    }
    renamed_projection
}

/// Whether `module_name` is one of the linear layers the MLX layout stores
/// quantized: the encoder's attention and feed-forward projections, and the
/// decision head's. Embeddings, norms, the type embedding, the option scorer
/// and the act head stay in half precision, where quantization error would
/// reach the answer most directly.
fn is_quantized_linear_module(module_name: &str) -> bool {
    let projection_suffixes: &[&str] = if module_name.starts_with("encoder.layers.") {
        &[".attn.Wo", ".attn.Wqkv", ".mlp.Wi", ".mlp.Wo"]
    } else if module_name.starts_with("head.layers.") {
        &[
            ".linear1",
            ".linear2",
            ".self_attn.in_proj",
            ".self_attn.out_proj",
        ]
    } else {
        return false;
    };
    for projection_suffix in projection_suffixes {
        if module_name.ends_with(projection_suffix) {
            return true;
        }
    }
    false
}

/// The tensors written for one source tensor, in the order they are written.
pub(crate) fn planned_output_tensors(
    source_record: &TensorRecord,
) -> Result<Vec<PlannedTensor>, String> {
    match tensor_treatment(&source_record.tensor_name, &source_record.shape) {
        TensorTreatment::Dropped => Ok(Vec::new()),
        TensorTreatment::KeptAsHalfPrecision { output_name } => Ok(vec![PlannedTensor {
            tensor_name: output_name,
            element_type: TensorElementType::Float16,
            shape: source_record.shape.clone(),
        }]),
        TensorTreatment::Quantized { module_name } => {
            planned_quantized_tensors(&module_name, &source_record.shape)
        }
    }
}

fn planned_quantized_tensors(
    module_name: &str,
    source_shape: &[u64],
) -> Result<Vec<PlannedTensor>, String> {
    let (output_feature_count, input_feature_count) = (source_shape[0], source_shape[1]);
    if input_feature_count % QUANTIZATION_GROUP_SIZE as u64 != 0 {
        return Err(format!(
            "{module_name} has {input_feature_count} inputs, which is not a whole number of \
             {QUANTIZATION_GROUP_SIZE}-weight groups"
        ));
    }
    let group_count = input_feature_count / QUANTIZATION_GROUP_SIZE as u64;
    Ok(vec![
        PlannedTensor {
            tensor_name: format!("{module_name}.weight"),
            element_type: TensorElementType::Unsigned32,
            shape: vec![
                output_feature_count,
                input_feature_count / QUANTIZED_WEIGHTS_PER_PACKED_WORD,
            ],
        },
        PlannedTensor {
            tensor_name: format!("{module_name}.scales"),
            element_type: TensorElementType::Float16,
            shape: vec![output_feature_count, group_count],
        },
        PlannedTensor {
            tensor_name: format!("{module_name}.biases"),
            element_type: TensorElementType::Float16,
            shape: vec![output_feature_count, group_count],
        },
    ])
}

#[cfg(test)]
#[path = "tensor_plan_tests.rs"]
mod tests;
