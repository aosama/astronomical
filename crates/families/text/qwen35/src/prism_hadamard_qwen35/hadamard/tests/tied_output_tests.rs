//! The tied output head marker: a tied head reuses the inverse-rotated
//! embedding's codes, so the contract accepts the marker only over such an
//! embedding and only when the pack ships no head of its own.

use super::{
    HadamardContractError, fixture_tensor_names, minimal_valid_contract_json, parse_contract,
};

fn tied_contract_json() -> serde_json::Value {
    let mut document = minimal_valid_contract_json();
    document["prism.hadamard.weight_names"] =
        serde_json::json!(["language_model.model.layers.0.mlp.down_proj.weight"]);
    document["prism.hadamard.tied_output"] = serde_json::json!(true);
    document
}

fn tied_pack_tensor_names() -> std::collections::BTreeSet<String> {
    let mut tensor_names = fixture_tensor_names();
    tensor_names.remove("language_model.lm_head.weight");
    tensor_names
}

#[test]
fn should_read_a_tied_output_head_over_an_inverse_rotated_embedding() {
    let contract = parse_contract(&tied_contract_json(), &tied_pack_tensor_names())
        .expect("a tied head over the inverse-rotated embedding should parse");
    assert!(contract.has_tied_output());
    assert!(contract.is_inverse("language_model.model.embed_tokens.weight"));
}

#[test]
fn should_read_an_absent_tied_output_marker_as_an_untied_head() {
    let contract = parse_contract(&minimal_valid_contract_json(), &fixture_tensor_names())
        .expect("the untied fixture should parse");
    assert!(!contract.has_tied_output());
}

#[test]
fn should_refuse_a_tied_output_head_beside_a_shipped_head() {
    let error = parse_contract(&tied_contract_json(), &fixture_tensor_names())
        .expect_err("a tied marker in a pack that ships its own head must be refused");
    assert!(matches!(
        error,
        HadamardContractError::TiedOutputWithSeparateHead
    ));
}

#[test]
fn should_refuse_a_tied_output_head_without_an_inverse_rotated_embedding() {
    let mut document = tied_contract_json();
    document["prism.hadamard.weight_names"] =
        serde_json::json!(["language_model.model.embed_tokens.weight"]);
    document["prism.hadamard.inverse_weight_names"] =
        serde_json::json!(["language_model.model.layers.0.mlp.down_proj.weight"]);
    let error = parse_contract(&document, &tied_pack_tensor_names())
        .expect_err("a tied head needs the embedding stored in the inverse basis");
    assert!(matches!(
        error,
        HadamardContractError::TiedOutputWithoutInverseEmbedding
    ));
}
