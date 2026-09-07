//! Tensor names required by the stacked affine dialect for one family member.
//!
//! Names are generated from the typed config so layer counts and expert pools
//! stay parameters, not a 36B golden list.

use crate::k2_horizon_mova::configuration::{K2HorizonMoVAConfig, K2HorizonMoVALayerKind};

/// Returns every tensor name this stacked affine member must present.
#[must_use]
pub fn expected_stacked_affine_tensor_names(config: &K2HorizonMoVAConfig) -> Vec<String> {
    let mut tensor_names = Vec::new();
    push_affine_module(&mut tensor_names, "model.embed_tokens");
    tensor_names.push("model.norm.weight".to_owned());
    if !config.tie_word_embeddings() {
        push_affine_module(&mut tensor_names, "lm_head");
    }
    for decoder_layer_index in 0..config.num_hidden_layers() {
        let layer_prefix = format!("model.layers.{decoder_layer_index}");
        tensor_names.push(format!("{layer_prefix}.input_layernorm.weight"));
        tensor_names.push(format!("{layer_prefix}.post_attention_layernorm.weight"));
        for projection in ["q_proj", "k_proj", "o_proj"] {
            push_affine_module(
                &mut tensor_names,
                &format!("{layer_prefix}.self_attn.{projection}"),
            );
        }
        if config.attention_gate_func().is_some() {
            push_affine_module(
                &mut tensor_names,
                &format!("{layer_prefix}.self_attn.gate_proj"),
            );
        }
        match config.layer_kind(decoder_layer_index) {
            K2HorizonMoVALayerKind::Dense => {
                push_affine_module(
                    &mut tensor_names,
                    &format!("{layer_prefix}.self_attn.v_proj"),
                );
                for projection in ["gate_proj", "up_proj", "down_proj"] {
                    push_affine_module(
                        &mut tensor_names,
                        &format!("{layer_prefix}.mlp.{projection}"),
                    );
                }
            }
            K2HorizonMoVALayerKind::SparseFeedForward => {
                push_affine_module(
                    &mut tensor_names,
                    &format!("{layer_prefix}.self_attn.v_proj"),
                );
                push_sparse_feed_forward(&mut tensor_names, &layer_prefix, config);
            }
            K2HorizonMoVALayerKind::SparseMixtureOfValues => {
                push_affine_module(
                    &mut tensor_names,
                    &format!("{layer_prefix}.self_attn.v_experts"),
                );
                push_affine_module(
                    &mut tensor_names,
                    &format!("{layer_prefix}.self_attn.v_router"),
                );
                if config.moe_gate_bias() {
                    tensor_names.push(format!("{layer_prefix}.self_attn.v_router.bias"));
                }
                push_sparse_feed_forward(&mut tensor_names, &layer_prefix, config);
            }
        }
    }
    tensor_names
}

fn push_sparse_feed_forward(
    tensor_names: &mut Vec<String>,
    layer_prefix: &str,
    config: &K2HorizonMoVAConfig,
) {
    push_affine_module(tensor_names, &format!("{layer_prefix}.mlp.gate"));
    if config.moe_gate_bias() {
        tensor_names.push(format!("{layer_prefix}.mlp.gate.bias"));
    }
    for projection in ["gate_proj", "up_proj", "down_proj"] {
        push_affine_module(
            tensor_names,
            &format!("{layer_prefix}.mlp.switch_mlp.{projection}"),
        );
        if config.num_shared_experts() > 0 {
            push_affine_module(
                tensor_names,
                &format!("{layer_prefix}.mlp.shared_experts.{projection}"),
            );
        }
    }
}

fn push_affine_module(tensor_names: &mut Vec<String>, module_path: &str) {
    tensor_names.push(format!("{module_path}.weight"));
    tensor_names.push(format!("{module_path}.scales"));
    tensor_names.push(format!("{module_path}.biases"));
}
