//! Real-artifact page-peak reservation contracts for jangtq2 complete layers.
//!
//! The plan-validation counts live in `mixed_expert_storage_tests`; this file
//! owns the admission-facing bound. The complete-layer page peak — packed
//! sources plus every decoded output — is what prefill admission must fit
//! beside live request memory before any expert of the layer can load.

use crate::expert_paging::quantized_expert_manifest::QuantizedExpertLayerPlan;
use crate::qwen3_5_core::artifacts::quantized_expert_layer_plan::build_quantized_expert_layer_plans;
use crate::qwen3_5_core::artifacts::{
    Qwen3_5ArtifactValidator, Qwen3_5ShardIndex, qwen3_5_language_tensor_profiles,
};
use crate::qwen3_5_core::configuration::Qwen3_5Config;

use std::collections::HashMap;

#[test]
#[ignore = "loads real JANGH2 bundle headers; select model_directory through cargo xtask test-run"]
fn should_keep_every_jangtq2_complete_layer_page_peak_under_six_gigabytes() {
    let model_directory =
        astronomical_test_harness::test_run::TestRunContext::current().required_model_directory();
    let layer_plans = jangtq2_real_artifact_layer_plans(&model_directory);

    // The graph decoder reserved roughly 15.1 GB per complete layer of this
    // bundle before expert blocking and about 8.3 GB after, and prefill
    // admission could not fit either beside a 10,000-token request's live
    // memory. The Metal page decoder needs no workspace, so the peak is the
    // packed source (about 0.7 GB) plus the decoded output (about 4.8 GB);
    // six binary gigabytes leaves margin above that while rejecting any
    // return of a workspace term.
    let mut largest_complete_layer_page_peak_bytes = 0_u64;
    for layer_plan in &layer_plans {
        let complete_source_payload_bytes = layer_plan
            .complete_expert_payload_byte_count()
            .expect("complete-layer source bytes should compute");
        let decoded_shapes = layer_plan
            .jangtq2_decoded_weight_page_shapes(layer_plan.expert_capacity)
            .expect("complete-layer decoded shapes should compute");
        let page_peak_bytes = crate::memory::expert_page_peak_allocation_bytes(
            complete_source_payload_bytes,
            &decoded_shapes,
            std::mem::size_of::<u16>(),
        )
        .expect("the complete-layer page peak should compute");
        largest_complete_layer_page_peak_bytes =
            largest_complete_layer_page_peak_bytes.max(page_peak_bytes);
    }
    assert!(
        largest_complete_layer_page_peak_bytes < 6 * (1_u64 << 30),
        "the largest complete-layer page peak is {} bytes; the Metal page decoder must keep it under 6 GB",
        largest_complete_layer_page_peak_bytes
    );
}

/// Loads the real bundle's layer plans through the production validators.
///
/// Mirrors the loading path of the plan-validation test: production artifact
/// validation first, then the shard index against the language tensor
/// contract, then the plan builders over the shard headers.
fn jangtq2_real_artifact_layer_plans(
    model_directory: &std::path::Path,
) -> Vec<QuantizedExpertLayerPlan> {
    let config_bytes = std::fs::read(model_directory.join("config.json"))
        .expect("the JANGH2 config should be readable");
    let config = Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("the JANGH2 config should parse through the production parser");
    Qwen3_5ArtifactValidator::new()
        .validate(model_directory, 1)
        .expect("the complete JANGH2 artifact should pass production validation");
    let shard_index_bytes = std::fs::read(model_directory.join("model.safetensors.index.json"))
        .expect("the JANGH2 shard index should be readable");
    let shard_index_value: serde_json::Value = serde_json::from_slice(&shard_index_bytes)
        .expect("the JANGH2 shard index should be valid JSON");
    let language_tensor_profiles = qwen3_5_language_tensor_profiles(&config)
        .expect("the JANGH2 language tensor contract should build");
    // Constructing the index is itself the contract check: it fails when the
    // bundle's shard layout disagrees with the language tensor profiles.
    let _validated_shard_index =
        Qwen3_5ShardIndex::from_json_bytes(&shard_index_bytes, &language_tensor_profiles)
            .expect("the JANGH2 shard index should match the language tensor contract");
    let weight_map = shard_index_value["weight_map"]
        .as_object()
        .expect("the JANGH2 index should contain a weight map")
        .iter()
        .map(|(tensor_name, shard_name)| {
            (
                tensor_name.clone(),
                shard_name
                    .as_str()
                    .expect("shard names should be strings")
                    .to_owned(),
            )
        })
        .collect::<HashMap<_, _>>();

    build_quantized_expert_layer_plans(
        model_directory,
        &weight_map,
        &HashMap::new(),
        &config,
        config.layer_count() as usize,
    )
    .expect("every sparse expert layer should validate against the JANGH2 shard headers")
}
