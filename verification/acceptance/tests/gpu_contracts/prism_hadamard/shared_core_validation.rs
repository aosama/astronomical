//! A folded ternary pack as the shared Qwen 3.5 core sees it: the core's
//! artifact validator accepts the pack through its folded ternary branch,
//! reads its configuration as the Qwen 3.5 text model it extends, and
//! carries its rotation contract for the loader (issue #2595). Reads file
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
    assert_eq!(config.layer_count(), 64);
    assert!(hadamard_contract.is_rotated("language_model.lm_head.weight"));
    assert!(hadamard_contract.is_inverse("language_model.model.embed_tokens.weight"));
    assert!(
        !validated_artifact.supports_image_input(),
        "the folded pack is served as a text model"
    );
}
