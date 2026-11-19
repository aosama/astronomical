//! Refusal contracts: every unsupported declaration is a typed refusal, never
//! a silent fallback to an untransformed load.

use super::{fixture_tensor_names, minimal_valid_contract_json, parse_contract};

#[test]
fn should_refuse_an_unsupported_schema_version() {
    let mut document = minimal_valid_contract_json();
    document["prism.hadamard.version"] = serde_json::json!(2);
    let error = parse_contract(&document, &fixture_tensor_names())
        .expect_err("an unsupported schema version must be refused");
    assert!(matches!(
        error,
        super::HadamardContractError::UnsupportedVersion { actual_version: 2 }
    ));
}

#[test]
fn should_refuse_an_unsupported_transform() {
    let mut document = minimal_valid_contract_json();
    document["prism.hadamard.transform"] = serde_json::json!("random-orthogonal");
    let error = parse_contract(&document, &fixture_tensor_names())
        .expect_err("an unsupported transform must be refused");
    assert!(matches!(
        error,
        super::HadamardContractError::UnsupportedTransform { .. }
    ));
}

#[test]
fn should_refuse_an_unsupported_axis() {
    let mut document = minimal_valid_contract_json();
    document["prism.hadamard.axis"] = serde_json::json!("output-first-dimension");
    let error = parse_contract(&document, &fixture_tensor_names())
        .expect_err("an unsupported axis must be refused");
    assert!(matches!(
        error,
        super::HadamardContractError::UnsupportedAxis { .. }
    ));
}

#[test]
fn should_refuse_an_implicit_sign_mode() {
    let mut document = minimal_valid_contract_json();
    document["prism.hadamard.sign_mode"] = serde_json::json!("implicit");
    let error = parse_contract(&document, &fixture_tensor_names())
        .expect_err("an implicit sign mode must be refused");
    assert!(matches!(
        error,
        super::HadamardContractError::UnsupportedSignMode { .. }
    ));
}

#[test]
fn should_refuse_an_unvalidated_block_size() {
    let mut document = minimal_valid_contract_json();
    document["prism.hadamard.block_size"] = serde_json::json!(768);
    let error = parse_contract(&document, &fixture_tensor_names())
        .expect_err("an unvalidated block size must be refused");
    assert!(matches!(
        error,
        super::HadamardContractError::UnsupportedBlockSize {
            actual_block_size: 768,
        }
    ));
}

#[test]
fn should_refuse_the_ungrouped_gdn_layout() {
    let mut document = minimal_valid_contract_json();
    document["prism.hadamard.gdn_v_grouped"] = serde_json::json!(false);
    let error = parse_contract(&document, &fixture_tensor_names())
        .expect_err("the ungrouped GDN layout must be refused");
    assert!(matches!(
        error,
        super::HadamardContractError::UnsupportedGdnLayout {
            actual_gdn_layout_is_grouped: false,
        }
    ));
}
