//! Unit tests for the structural priority rules. Expected values are the
//! arithmetic of OptiQ's `_structural_priority` worked by hand.

use std::path::PathBuf;

use super::*;
use crate::bit_width::GroupSize;
use crate::module_inventory::{ModuleWeightSource, QuantizableModuleDescriptor, SourceTensorDtype};
use crate::module_taxonomy::{classify_module, transformer_block_index};

#[test]
fn should_rank_embeddings_and_the_head_at_the_top() {
    assert_eq!(structural_priority("model.embed_tokens", 32), 1.0);
    assert_eq!(structural_priority("lm_head", 32), 1.0);
}

#[test]
fn should_protect_the_first_and_last_blocks() {
    // 0.90 for the edge block plus 0.30 for attention, clamped to 1.0.
    assert_eq!(
        structural_priority("model.layers.0.self_attn.q_proj", 32),
        1.0
    );
    // 0.90 for the edge block, dense feed-forward adds nothing.
    assert_eq!(structural_priority("model.layers.31.mlp.up_proj", 32), 0.90);
}

#[test]
fn should_rank_routed_experts_below_the_dense_feed_forward_in_the_middle() {
    let dense_priority = structural_priority("model.layers.16.mlp.up_proj", 32);
    let expert_priority = structural_priority("model.layers.16.mlp.switch_mlp.up_proj", 32);
    let router_priority = structural_priority("model.layers.16.mlp.gate.weight", 32);

    assert!(expert_priority < dense_priority);
    assert!(router_priority > dense_priority);
}

#[test]
fn should_never_fall_below_the_floor() {
    assert!(structural_priority("model.layers.16.mlp.switch_mlp.up_proj", 32) >= 0.05);
}

fn descriptor(module_name: &str, logical_parameter_count: u64) -> QuantizableModuleDescriptor {
    QuantizableModuleDescriptor {
        module_name: module_name.to_owned(),
        logical_shape: vec![64, 64],
        logical_parameter_count,
        source_dtype: SourceTensorDtype::BFloat16,
        weight_source: ModuleWeightSource::Whole {
            shard_file_name: "model.safetensors".to_owned(),
            tensor_name: format!("{module_name}.weight"),
        },
        role: classify_module(module_name),
        transformer_block_index: transformer_block_index(module_name),
    }
}

fn inventory_of(module_names: &[&str]) -> ModuleInventory {
    let mut descriptors = Vec::new();
    for module_name in module_names {
        descriptors.push(descriptor(module_name, 4096));
    }
    ModuleInventory::from_descriptors(
        PathBuf::from("source"),
        GroupSize::new(64).expect("valid"),
        descriptors,
    )
}

#[test]
fn should_encode_the_priority_as_a_curve_falling_to_zero_at_the_highest_width() {
    let inventory = inventory_of(&["model.embed_tokens", "model.layers.0.mlp.up_proj"]);
    let ladder = CandidateBitWidths::new(&[4, 8]).expect("valid ladder");

    let table = score_static_structural(&inventory, &ladder);

    let embedding = table.result_for("model.embed_tokens").expect("scored");
    let four_bits = BitWidth::new(4).expect("valid");
    let eight_bits = BitWidth::new(8).expect("valid");
    assert_eq!(embedding.kl_divergence_at(four_bits), Some(1.0));
    assert_eq!(embedding.kl_divergence_at(eight_bits), Some(0.0));
    assert_eq!(embedding.logical_parameter_count, 4096);
    assert_eq!(table.method, SensitivityMethod::StaticStructuralRules);
}

#[test]
fn should_derive_the_block_count_from_the_inventory_names() {
    // Blocks 0..=3 make block 3 the last one, so it scores as an edge block.
    let inventory = inventory_of(&[
        "model.layers.0.mlp.up_proj",
        "model.layers.2.mlp.up_proj",
        "model.layers.3.mlp.up_proj",
    ]);
    let ladder = CandidateBitWidths::new(&[2, 4, 8]).expect("valid ladder");

    let table = score_static_structural(&inventory, &ladder);

    let last_block = table
        .result_for("model.layers.3.mlp.up_proj")
        .expect("scored");
    let two_bits = BitWidth::new(2).expect("valid");
    assert!((last_block.kl_divergence_at(two_bits).expect("measured") - 0.90).abs() < 1e-12);
}

#[test]
fn should_score_an_empty_inventory_as_an_empty_table() {
    let inventory = inventory_of(&[]);
    let ladder = CandidateBitWidths::new(&[4, 8]).expect("valid ladder");

    let table = score_static_structural(&inventory, &ladder);

    assert!(table.results.is_empty());
}

#[test]
fn should_ignore_a_layer_index_that_ends_the_name() {
    // OptiQ's pattern needs a dot after the digits, so this is mid-network.
    assert_eq!(structural_priority("model.layers.0", 32), 0.30);
}
