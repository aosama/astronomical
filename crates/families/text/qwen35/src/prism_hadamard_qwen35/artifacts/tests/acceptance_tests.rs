//! Acceptance contracts: the structurally exact fixture pack validates
//! end to end with every accessor populated.

use super::super::{FoldedPackValidationError, FoldedPackValidator};
use super::pack_fixture::{
    FIXTURE_HIDDEN_SIZE, FIXTURE_INTERMEDIATE_SIZE, FIXTURE_VOCAB_SIZE, fixture_pack_directory,
    remove_fixture_directory,
};

#[test]
fn should_validate_the_complete_fixture_pack() {
    let fixture_directory = fixture_pack_directory();
    let validated_pack = FoldedPackValidator::new()
        .validate(&fixture_directory, 4096)
        .expect("the structurally exact fixture pack should validate");
    assert_eq!(
        validated_pack.config().text().hidden_size,
        FIXTURE_HIDDEN_SIZE
    );
    assert_eq!(
        validated_pack.config().text().intermediate_size,
        FIXTURE_INTERMEDIATE_SIZE
    );
    assert_eq!(
        validated_pack.config().text().vocab_size,
        FIXTURE_VOCAB_SIZE
    );
    assert_eq!(validated_pack.config().packed_modules().len(), 27);
    assert!(!validated_pack.config().vision_declared());
    assert_eq!(
        validated_pack.hadamard_contract().block_size(),
        super::pack_fixture::FIXTURE_HADAMARD_BLOCK
    );
    assert_eq!(validated_pack.max_output_tokens(), 4096);
    assert!(
        validated_pack.total_payload_bytes() > 0,
        "the pack should account for its payload bytes"
    );
    assert!(
        (12..=16).contains(&validated_pack.revision().len()),
        "the revision should be a zero-padded hex string"
    );
    remove_fixture_directory(&fixture_directory);
}

#[test]
fn should_refuse_a_missing_model_directory() {
    let error = FoldedPackValidator::new()
        .validate("/nonexistent/prism-pack", 4096)
        .expect_err("a missing directory must be refused");
    assert!(matches!(error, FoldedPackValidationError::Artifact(_)));
}

#[test]
fn should_refuse_when_a_required_file_is_absent() {
    let fixture_directory = super::pack_fixture::write_fixture_pack(|mutation| {
        mutation.config["hadamard_config"] = serde_json::json!("hadamard.json");
    });
    std::fs::remove_file(fixture_directory.join("tokenizer.json"))
        .expect("the tokenizer stub should be removable");
    let error = FoldedPackValidator::new()
        .validate(&fixture_directory, 4096)
        .expect_err("a missing required file must be refused");
    assert!(matches!(error, FoldedPackValidationError::Artifact(_)));
    remove_fixture_directory(&fixture_directory);
}
