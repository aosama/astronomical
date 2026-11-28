//! Tied output heads: a pack whose head is its embedding table (Prism's 2B)
//! ships no head module, and its config and manifest must say so together.

use super::super::{FoldedPackValidationError, FoldedPackValidator};
use super::pack_fixture::{remove_fixture_directory, write_fixture_pack};

#[test]
fn should_validate_a_tied_pack_that_ships_no_head_module() {
    let fixture_directory = write_fixture_pack(|mutation| mutation.tie_output_head());
    let validated_pack = FoldedPackValidator::new()
        .validate(&fixture_directory, 4096)
        .expect("a tied pack whose declarations agree should validate");
    assert!(validated_pack.config().has_tied_output());
    assert!(validated_pack.hadamard_contract().has_tied_output());
    assert!(
        validated_pack
            .config()
            .packed_modules()
            .iter()
            .all(|packed_module| packed_module.path() != "lm_head")
    );
    remove_fixture_directory(&fixture_directory);
}

#[test]
fn should_refuse_a_tied_config_over_an_untied_manifest() {
    let fixture_directory = write_fixture_pack(|mutation| {
        mutation.tie_output_head();
        mutation.hadamard["prism.hadamard.tied_output"] = serde_json::json!(false);
    });
    let error = FoldedPackValidator::new()
        .validate(&fixture_directory, 4096)
        .expect_err("a config and manifest that disagree about the head must be refused");
    assert!(
        matches!(error, FoldedPackValidationError::ContractModuleDisagreement { ref detail } if detail.contains("tied")),
        "{error}"
    );
    remove_fixture_directory(&fixture_directory);
}

#[test]
fn should_refuse_an_untied_pack_that_lost_its_head_module() {
    let fixture_directory = write_fixture_pack(|mutation| {
        mutation.tie_output_head();
        mutation.config["tie_word_embeddings"] = serde_json::json!(false);
        mutation.config["text_config"]["tie_word_embeddings"] = serde_json::json!(false);
        mutation.config["tied_output"] = serde_json::json!(false);
        mutation.hadamard["prism.hadamard.tied_output"] = serde_json::json!(false);
    });
    let error = FoldedPackValidator::new()
        .validate(&fixture_directory, 4096)
        .expect_err("an untied pack without its head must be refused");
    assert!(
        matches!(error, FoldedPackValidationError::ContractModuleDisagreement { ref detail } if detail.contains("lm_head")),
        "{error}"
    );
    remove_fixture_directory(&fixture_directory);
}
