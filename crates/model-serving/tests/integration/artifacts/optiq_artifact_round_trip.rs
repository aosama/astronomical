//! The cross-crate contract: documents the quantization crate writes are
//! accepted by this crate's strict OptiQ readers and validate against each
//! other, exactly as a daemon load would. This is the test that proves a
//! Rust-built artifact is loadable by Astronomical before any GPU time is
//! spent on one.

use std::collections::BTreeMap;

use astronomical_model_serving::OptiQMetadata;
use astronomical_quantization::artifact::{OptiQArtifactMetadata, QuantizationConfigDocument};
use astronomical_quantization::sensitivity::SensitivityMethod;

use super::frozen_ornith_1_0_fixtures::frozen_ornith_1_0_optiq_config_bytes;

#[test]
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

    let mut metadata = OptiQArtifactMetadata::for_method(
        SensitivityMethod::ExactBfloat16Reference,
        "example/base",
    );
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

    // Splice the writer's document into the frozen Ornith config under both
    // keys, then parse it the way a daemon load does.
    let mut config_value: serde_json::Value =
        serde_json::from_slice(&frozen_ornith_1_0_optiq_config_bytes())
            .expect("the frozen config should be JSON");
    config_value["quantization"] = document_value.clone();
    config_value["quantization_config"] = document_value;
    let config_bytes = serde_json::to_vec(&config_value).expect("the config should serialize");
    let parsed_config = astronomical_model_serving::Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("the spliced config should parse");

    let profiles = parsed_config.quantized_module_profiles();
    let gate_profile = &profiles["model.layers.0.mlp.switch_mlp.gate_proj"];
    let query_profile = &profiles["model.layers.0.self_attn.q_proj"];
    assert_eq!((gate_profile.bits, gate_profile.group_size), (4, 64));
    assert_eq!((query_profile.bits, query_profile.group_size), (8, 64));
}
