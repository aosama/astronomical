//! A folded ternary pack as the shared Qwen 3.5 core sees it: the core's
//! artifact validator accepts the pack through its folded ternary branch,
//! reads its configuration as the Qwen 3.5 text model it extends, and
//! carries its rotation contract for the loader and its vision tower
//! (issue #2595). Reads file
//! headers only; no weight is loaded.

use model_serving::Qwen3_5ArtifactValidator;

use super::support::pack_directory;

const MAXIMUM_OUTPUT_TOKENS: u32 = 4_096;

#[test]
#[ignore = "reads an installed folded ternary pack; run through cargo test-prism-hadamard-direct with model_directory configured"]
fn should_validate_a_folded_ternary_pack_as_a_shared_core_artifact() {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(pack_directory(), MAXIMUM_OUTPUT_TOKENS)
        .expect("the shared core validates the folded pack");

    let hadamard_contract = validated_artifact
        .hadamard_contract()
        .expect("the folded pack carries its rotation contract");
    let config = validated_artifact.config();
    assert_eq!(config.layer_count(), declared_layer_count());
    assert!(hadamard_contract.is_inverse("language_model.model.embed_tokens.weight"));
    // Prism's 2B reuses its embedding as the head; the 9B and 27B ship a
    // rotated head of their own.
    assert_eq!(
        hadamard_contract.has_tied_output(),
        !hadamard_contract.is_rotated("language_model.lm_head.weight")
    );
    assert_eq!(
        hadamard_contract.has_tied_output(),
        config.has_tied_embeddings()
    );
    assert!(
        validated_artifact.supports_image_input(),
        "the folded pack's float16 vision tower is served"
    );
}

/// The layer count the pack's own `config.json` declares, so the journey
/// serves every Bonsai size (24 layers for the 2B, 32 for the 4B and 9B, 64
/// for the 27B) instead of pinning one.
fn declared_layer_count() -> u32 {
    let config_document: serde_json::Value = serde_json::from_slice(
        &std::fs::read(pack_directory().join("config.json"))
            .expect("the pack's config.json should be readable"),
    )
    .expect("the pack's config.json should be JSON");
    config_document["text_config"]["num_hidden_layers"]
        .as_u64()
        .expect("the pack declares its layer count") as u32
}
