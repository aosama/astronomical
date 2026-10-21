//! Real-artifact page-peak reservation contracts for jangtq2 complete layers.
//!
//! The plan-validation counts live in `mixed_expert_storage_tests`; this file
//! owns the admission-facing bound. The complete-layer page peak — packed
//! sources plus every decoded output, one assembly copy of the largest
//! projection, one expert block of decode workspace, and the auxiliary index
//! arrays — is what prefill admission must fit beside live request memory
//! before any expert of the layer can load.

use crate::expert_paging::quantized_expert_manifest::QuantizedExpertLayerPlan;
use crate::qwen3_5_core::artifacts::quantized_expert_layer_plan::build_quantized_expert_layer_plans;
use crate::qwen3_5_core::artifacts::{
    Qwen3_5ArtifactValidator, Qwen3_5ShardIndex, qwen3_5_language_tensor_profiles,
};
use crate::qwen3_5_core::configuration::Qwen3_5Config;

use std::collections::HashMap;

#[test]
#[ignore = "loads the real JANGH2 bundle headers through the production validators; set ASTRONOMICAL_JANGTQ2_MODEL_DIR"]
fn should_keep_every_jangtq2_complete_layer_page_peak_under_nine_gigabytes() {
    let model_directory = std::env::var_os("ASTRONOMICAL_JANGTQ2_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .expect("ASTRONOMICAL_JANGTQ2_MODEL_DIR should name the local JANGH2 artifact");
    let layer_plans = jangtq2_real_artifact_layer_plans(&model_directory);

    // The pre-blocking decoder reserved roughly 15.1 GB per complete layer of
    // this bundle, which prefill admission could not fit beside live request
    // memory and which collapsed prompt processing to one-token chunks;
    // crossing this bound again would reintroduce that collapse. Nine binary
    // gigabytes leaves margin above the measured post-blocking peak of about
    // 8.3 GB while staying far below the broken value.
    let mut largest_complete_layer_page_peak_bytes = 0_u64;
    for layer_plan in &layer_plans {
        let complete_source_payload_bytes = layer_plan
            .complete_expert_payload_byte_count()
            .expect("complete-layer source bytes should compute");
        let decoded_shapes = layer_plan
            .jangtq2_decoded_weight_page_shapes(layer_plan.expert_capacity)
            .expect("complete-layer decoded shapes should compute");
        let workspace_shapes = layer_plan
            .jangtq2_decode_workspace_page_shapes(layer_plan.expert_capacity)
            .expect("complete-layer block workspace shapes should compute");
        let assembly_copy_count =
            layer_plan.jangtq2_decoded_output_assembly_copy_count(layer_plan.expert_capacity);
        let expected_block_expert_count =
            crate::jangtq2_qwen35::codebook::decode_block_expert_count(layer_plan.expert_capacity);
        for (decoded_shape, workspace_shape) in decoded_shapes.iter().zip(&workspace_shapes) {
            assert_eq!(
                workspace_shape[0], expected_block_expert_count,
                "the decode workspace must follow the family block size"
            );
            assert_eq!(
                workspace_shape[1..], decoded_shape[1..],
                "the decode workspace must keep the projection geometry"
            );
        }
        assert_eq!(
            assembly_copy_count, 2,
            "complete layers assemble block outputs and reserve the transient copy"
        );
        let decoder_index_tensor_shapes = decoded_shapes
            .iter()
            .map(|decoded_shape| {
                vec![decoded_shape[2].max(decoded_shape[0].saturating_mul(decoded_shape[1]))]
            })
            .collect::<Vec<_>>();
        let page_peak_bytes = crate::memory::expert_page_peak_allocation_bytes(
            complete_source_payload_bytes,
            &decoded_shapes,
            &workspace_shapes,
            assembly_copy_count,
            std::mem::size_of::<u16>(),
            crate::jangtq2_qwen35::codebook::MAXIMUM_SIMULTANEOUS_DECODE_TEMPORARY_ARRAY_COUNT,
            std::mem::size_of::<u32>(),
            &decoder_index_tensor_shapes,
            crate::jangtq2_qwen35::codebook::JANGTQ2_DECODE_AUXILIARY_ARRAY_COUNT,
            std::mem::size_of::<u32>(),
        )
        .expect("the complete-layer page peak should compute");
        largest_complete_layer_page_peak_bytes =
            largest_complete_layer_page_peak_bytes.max(page_peak_bytes);
    }
    assert!(
        largest_complete_layer_page_peak_bytes < 9 * (1_u64 << 30),
        "the largest complete-layer page peak is {} bytes; the blocked decoder must keep it under 9 GB",
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
