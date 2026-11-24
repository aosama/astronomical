//! Which of Laya's layers carry add-on weights: every 8-bit linear layer the
//! forward pass runs, in the ModernBERT trunk and in the decision head.
//!
//! The dense scorer at the very end is trained whole instead, as the final
//! layers are; norms and embeddings stay as shipped.

use decision_contract::DecisionFailure;
use runtime_integration::MlxSafetensors;

use crate::configuration::LayaConfiguration;
use crate::decision_models::decision_adapters::AdapterTarget;
use crate::modernbert_trunk::weight_layout::ModernBertWeightLayout;

/// The trunk's four projections per layer, as the trunk names them.
const TRUNK_LAYER_PROJECTIONS: [&str; 4] = ["attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"];
/// The head's four projections per layer.
const HEAD_LAYER_PROJECTIONS: [&str; 4] = [
    "self_attn.in_proj",
    "self_attn.out_proj",
    "linear1",
    "linear2",
];

/// Every layer of this Laya that takes an add-on, trunk first, layer by
/// layer, with widths read from the stored weights.
pub fn laya_adapter_targets(
    tensors: &MlxSafetensors,
    configuration: &LayaConfiguration,
    weight_layout: &ModernBertWeightLayout,
) -> Result<Vec<AdapterTarget>, DecisionFailure> {
    let group_size = configuration.encoder.quantization_group_size as usize;
    let trunk_layer_count = configuration.encoder.layer_count as usize;
    let head_layer_count = configuration.head_layer_count as usize;
    let mut targets = Vec::with_capacity((trunk_layer_count + head_layer_count) * 4);
    for layer_index in 0..trunk_layer_count {
        for projection in TRUNK_LAYER_PROJECTIONS {
            let target_name =
                weight_layout.tensor_name(&format!("layers.{layer_index}.{projection}"));
            targets.push(AdapterTarget::of_quantized_linear(
                tensors,
                target_name,
                group_size,
            )?);
        }
    }
    for head_layer_index in 0..head_layer_count {
        for projection in HEAD_LAYER_PROJECTIONS {
            targets.push(AdapterTarget::of_quantized_linear(
                tensors,
                format!("head.layers.{head_layer_index}.{projection}"),
                group_size,
            )?);
        }
    }
    Ok(targets)
}
