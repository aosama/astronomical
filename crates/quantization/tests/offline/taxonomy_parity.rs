//! The taxonomy must agree with OptiQ's four name predicates on every
//! fixture name.

use std::collections::BTreeSet;

use astronomical_quantization::module_taxonomy::{
    ModuleRole, boundary_block_modules, classify_module, is_attention_component,
    transformer_block_index,
};
use astronomical_quantization::requantize::bit_policy::is_router_tensor_key;

use super::fixtures;

#[test]
fn should_agree_with_optiq_on_block_indices() {
    let fixture = fixtures::taxonomy_parity();

    for module_name in &fixture.module_names {
        assert_eq!(
            transformer_block_index(module_name),
            fixture.block_index[module_name],
            "block index differs for {module_name}"
        );
    }
}

#[test]
fn should_agree_with_optiq_on_attention_components() {
    let fixture = fixtures::taxonomy_parity();

    for module_name in &fixture.module_names {
        assert_eq!(
            is_attention_component(classify_module(module_name)),
            fixture.is_attention_component[module_name],
            "attention classification differs for {module_name}"
        );
    }
}

#[test]
fn should_agree_with_optiq_on_the_protected_output_and_moe_sets() {
    let fixture = fixtures::taxonomy_parity();
    let expected_output: BTreeSet<&str> =
        fixture.output_layers.iter().map(String::as_str).collect();
    let expected_moe: BTreeSet<&str> = fixture
        .moe_protected_layers
        .iter()
        .map(String::as_str)
        .collect();

    let rust_output: BTreeSet<&str> = fixture
        .module_names
        .iter()
        .map(String::as_str)
        .filter(|name| {
            matches!(
                classify_module(name),
                ModuleRole::EmbeddingOrOutputHead | ModuleRole::OutsideTransformerBlocks
            )
        })
        .collect();
    let rust_moe: BTreeSet<&str> = fixture
        .module_names
        .iter()
        .map(String::as_str)
        .filter(|name| {
            matches!(
                classify_module(name),
                ModuleRole::MixtureOfExpertsRouter | ModuleRole::SharedExpert
            )
        })
        .collect();

    assert_eq!(rust_output, expected_output);
    assert_eq!(rust_moe, expected_moe);
}

#[test]
fn should_agree_with_optiq_on_boundary_modules() {
    let fixture = fixtures::taxonomy_parity();
    let names: Vec<&str> = fixture.module_names.iter().map(String::as_str).collect();

    let mut boundary = boundary_block_modules(&names, 1);
    boundary.sort_unstable();

    assert_eq!(
        boundary,
        fixture
            .boundary_layers_n1
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "pending implementation of is_router_tensor_key"]
fn should_agree_with_the_shardwise_router_predicate() {
    let fixture = fixtures::taxonomy_parity();

    for (tensor_key, expected) in &fixture.shardwise_is_router {
        assert_eq!(
            is_router_tensor_key(tensor_key),
            *expected,
            "router predicate differs for {tensor_key}"
        );
    }
}

#[test]
fn should_agree_with_the_shardwise_expert_predicate() {
    let fixture = fixtures::taxonomy_parity();

    for (tensor_key, expected) in &fixture.shardwise_is_expert {
        let module_name = tensor_key.strip_suffix(".weight").unwrap_or(tensor_key);
        assert_eq!(
            classify_module(module_name) == ModuleRole::RoutedExpert,
            *expected,
            "expert predicate differs for {tensor_key}"
        );
    }
}
