//! The cross-crate contract: documents the quantization crate writes are
//! accepted by this crate's strict OptiQ readers and validate against each
//! other, exactly as a daemon load would. This is the test that proves a
//! Rust-built artifact is loadable by Astronomical before any GPU time is
//! spent on one.

use std::collections::BTreeMap;

use astronomical_model_serving::OptiQMetadata;
use astronomical_quantization::artifact::{OptiQArtifactMetadata, QuantizationConfigDocument};

use super::frozen_ornith_1_0_fixtures::frozen_ornith_1_0_optiq_config_bytes;

#[test]
#[ignore = "pending implementation of the quantization crate's artifact writers"]
fn should_produce_metadata_the_strict_optiq_reader_accepts_and_validates_against_the_config() {
    // Rebuild the frozen Ornith 1.0 config's written module map through the
    // quantization crate's writer, then hand both documents to this crate's
    // readers: the metadata must parse strictly and match the config.
    let config_bytes = frozen_ornith_1_0_optiq_config_bytes();
    let ornith_config = astronomical_model_serving::Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("the frozen Ornith 1.0 OptiQ config should parse");
    let written: BTreeMap<String, (u32, u32)> = ornith_config
        .quantized_module_profiles()
        .iter()
        .filter(|(_, profile)| !profile.is_unquantized())
        .map(|(name, profile)| (name.clone(), (profile.bits, profile.group_size)))
        .collect();

    let mut metadata = OptiQArtifactMetadata {
        method: "optiq_mixed_precision".to_owned(),
        base_model: "example/base".to_owned(),
        reference: "bf16".to_owned(),
        target_bits_per_weight: 4.0,
        achieved_bits_per_weight: 4.0,
        module_count_at_highest_width: 0,
        module_count_at_lowest_width: 0,
        threshold: 0.0,
        per_layer: BTreeMap::new(),
        producer: "astronomical-quantization".to_owned(),
    };
    metadata.describe_written_artifact(&written);
    let metadata_bytes = serde_json::to_vec(&metadata).expect("the metadata should serialize");

    let parsed = OptiQMetadata::from_json_bytes(&metadata_bytes)
        .expect("the strict reader should accept the writer's document");
    assert_eq!(parsed.measured_module_count(), written.len());
    parsed
        .validate_against_config(&ornith_config)
        .expect("every written profile should match the config override");
}

#[test]
#[ignore = "pending implementation of QuantizationConfigDocument::from_written_modules"]
fn should_produce_a_quantization_document_the_config_parser_accepts() {
    // The writer's document, embedded under both keys of a minimal config,
    // must parse through Qwen3_5Config and resolve the same per-module
    // profiles it was built from.
    let written = BTreeMap::from([
        (
            "model.layers.0.mlp.switch_mlp.gate_proj".to_owned(),
            (4_u32, 64_u32),
        ),
        (
            "model.layers.0.self_attn.q_proj".to_owned(),
            (8_u32, 64_u32),
        ),
    ]);
    let document = QuantizationConfigDocument::from_written_modules(64, 4, &written);
    let document_value = serde_json::to_value(&document).expect("the document should serialize");

    // The exact minimal config shape is the frozen fixture's with both
    // quantization keys replaced; the implementer splices `document_value`
    // in and asserts `quantized_module_profiles()` reports (4, 64) and
    // (8, 64) for the two modules.
    assert_eq!(document_value["mode"], "affine");
}
