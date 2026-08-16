//! Laguna cache topology is derived from canonical layer descriptors.

use astronomical_model_serving::{
    DecoderCacheLayerLayout, PersistentPromptCacheModelContract, laguna_decoder_cache_layout,
};

use super::support::{
    LagunaQualificationSize, config_value, normalize, qualification_config_value,
};

#[test]
fn should_derive_mixed_append_only_and_rotating_topology_from_descriptors() {
    let synthetic_contract = normalize(config_value(5));
    let synthetic_layout = laguna_decoder_cache_layout(&synthetic_contract)
        .expect("a synthetic Laguna contract should produce a cache layout");
    assert_eq!(synthetic_layout.layer_count(), 5);

    let extra_small_layout = laguna_decoder_cache_layout(&normalize(qualification_config_value(
        LagunaQualificationSize::ExtraSmall,
    )))
    .expect("the XS qualification contract should produce a cache layout");
    let small_layout = laguna_decoder_cache_layout(&normalize(qualification_config_value(
        LagunaQualificationSize::Small,
    )))
    .expect("the S qualification contract should produce a cache layout");
    assert_ne!(
        extra_small_layout.layer_count(),
        small_layout.layer_count(),
        "XS and S cache topologies must remain distinct evidence rows"
    );
    assert!((0..extra_small_layout.layer_count()).any(|layer_index| {
        matches!(
            extra_small_layout.layer(layer_index),
            Some(DecoderCacheLayerLayout::RotatingWindowAttention { .. })
        )
    }));
    assert!((0..extra_small_layout.layer_count()).any(|layer_index| {
        matches!(
            extra_small_layout.layer(layer_index),
            Some(DecoderCacheLayerLayout::AppendOnlyAttention { .. })
        )
    }));
    let rotating_counter_names = extra_small_layout
        .boundary_tensor_layouts()
        .into_iter()
        .map(|persisted_tensor| persisted_tensor.persistent_tensor_name())
        .collect::<Vec<_>>();
    assert!(
        rotating_counter_names
            .iter()
            .any(|tensor_name| tensor_name.contains("absolute_position")),
        "rotating layers must persist absolute_position: {rotating_counter_names:?}"
    );
    assert!(
        rotating_counter_names
            .iter()
            .any(|tensor_name| tensor_name.contains("ring_write_index")),
        "rotating layers must persist ring_write_index: {rotating_counter_names:?}"
    );
}

#[test]
fn should_resolve_a_fifty_gigabyte_cache_contract_or_name_the_quota_limit() {
    let extra_small_contract = normalize(qualification_config_value(
        LagunaQualificationSize::ExtraSmall,
    ));
    let extra_small_layout = laguna_decoder_cache_layout(&extra_small_contract)
        .expect("the XS qualification contract should produce a cache layout");
    let fifty_gigabyte_quota_bytes = 50_000_000_000_u64;
    let resolved_contract = PersistentPromptCacheModelContract::resolve(
        "Laguna-XS".to_owned(),
        "qualification".to_owned(),
        extra_small_layout,
        extra_small_contract.model().maximum_position_count() as usize,
        40_000_000_000,
        fifty_gigabyte_quota_bytes,
        None,
        1,
    );
    match resolved_contract {
        Ok(model_contract) => {
            assert!(model_contract.block_token_count() > 0);
        }
        Err(contract_error) => {
            let contract_error_text = contract_error.to_string();
            assert!(
                contract_error_text.contains("quota") || contract_error_text.contains("SSD"),
                "an unresolved 50 GB contract must name the quota limit: {contract_error_text}"
            );
        }
    }
}
