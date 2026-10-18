//! Manifest contracts: every declared tensor must resolve against the
//! artifact, and the forward and inverse sets must stay disjoint.

use super::{
    fixture_tensor_names, minimal_valid_contract_json, oversize_contract_bytes, parse_contract,
};

#[test]
fn should_refuse_an_empty_transform_manifest() {
    let mut document = minimal_valid_contract_json();
    document["prism.hadamard.weight_names"] = serde_json::json!([]);
    let error = parse_contract(&document, &fixture_tensor_names())
        .expect_err("an empty transform manifest must be refused");
    assert!(matches!(error, super::HadamardContractError::EmptyManifest));
}

#[test]
fn should_refuse_an_unknown_tensor_reference() {
    let mut document = minimal_valid_contract_json();
    document["prism.hadamard.weight_names"] = serde_json::json!([
        "language_model.lm_head.weight",
        "language_model.model.layers.9.mlp.down_proj.weight",
    ]);
    let error = parse_contract(&document, &fixture_tensor_names())
        .expect_err("a manifest reference outside the artifact must be refused");
    assert!(matches!(
        error,
        super::HadamardContractError::UnknownTensorReference { ref tensor_name }
            if tensor_name == "language_model.model.layers.9.mlp.down_proj.weight"
    ));
}

#[test]
fn should_refuse_a_duplicate_tensor_reference() {
    let mut document = minimal_valid_contract_json();
    document["prism.hadamard.weight_names"] = serde_json::json!([
        "language_model.lm_head.weight",
        "language_model.lm_head.weight",
    ]);
    let error = parse_contract(&document, &fixture_tensor_names())
        .expect_err("a duplicate manifest reference must be refused");
    assert!(matches!(
        error,
        super::HadamardContractError::DuplicateTensorReference
    ));
}

#[test]
fn should_refuse_overlapping_forward_and_inverse_manifests() {
    let mut document = minimal_valid_contract_json();
    document["prism.hadamard.inverse_weight_names"] = serde_json::json!([
        "language_model.model.embed_tokens.weight",
        "language_model.lm_head.weight",
    ]);
    let error = parse_contract(&document, &fixture_tensor_names())
        .expect_err("overlapping forward and inverse manifests must be refused");
    assert!(matches!(
        error,
        super::HadamardContractError::ForwardInverseOverlap
    ));
}

#[test]
fn should_refuse_an_oversize_document_before_parsing() {
    let error = crate::prism_hadamard_qwen35::hadamard::HadamardContract::from_json_bytes(
        &oversize_contract_bytes(),
        &fixture_tensor_names(),
    )
    .expect_err("an oversize document must be refused on size alone");
    assert!(matches!(
        error,
        super::HadamardContractError::ContractTooLarge { .. }
    ));
}
