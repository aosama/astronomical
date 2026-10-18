//! Sign-table contracts: the per-width ±1 vectors must be complete, finite,
//! and free of duplicates or trailing values.

use super::{fixture_tensor_names, minimal_valid_contract_json, parse_contract};

#[test]
fn should_refuse_an_empty_sign_table() {
    let mut document = minimal_valid_contract_json();
    document["prism.hadamard.sign_widths"] = serde_json::json!([]);
    document["prism.hadamard.sign_values"] = serde_json::json!([]);
    let error = parse_contract(&document, &fixture_tensor_names())
        .expect_err("an empty sign table must be refused");
    assert!(matches!(
        error,
        super::HadamardContractError::EmptySignTable
    ));
}

#[test]
fn should_refuse_a_zero_sign_width() {
    let mut document = minimal_valid_contract_json();
    document["prism.hadamard.sign_widths"] = serde_json::json!([0]);
    document["prism.hadamard.sign_values"] = serde_json::json!([]);
    let error = parse_contract(&document, &fixture_tensor_names())
        .expect_err("a zero sign width must be refused");
    assert!(matches!(
        error,
        super::HadamardContractError::InvalidSignWidth {
            actual_sign_width: 0,
        }
    ));
}

#[test]
fn should_refuse_truncated_sign_values() {
    let mut document = minimal_valid_contract_json();
    document["prism.hadamard.sign_values"] = serde_json::json!([1, -1]);
    let error = parse_contract(&document, &fixture_tensor_names())
        .expect_err("sign values shorter than the declared widths must be refused");
    assert!(matches!(
        error,
        super::HadamardContractError::SignValuesTruncated { .. }
    ));
}

#[test]
fn should_refuse_a_sign_value_outside_the_alphabet() {
    let mut document = minimal_valid_contract_json();
    document["prism.hadamard.sign_values"] = serde_json::json!([1, 0, 1, -1, -1, 1, 1]);
    let error = parse_contract(&document, &fixture_tensor_names())
        .expect_err("a sign value outside the ±1 alphabet must be refused");
    assert!(matches!(
        error,
        super::HadamardContractError::InvalidSignValue
    ));
}

#[test]
fn should_refuse_a_duplicate_sign_width() {
    let mut document = minimal_valid_contract_json();
    document["prism.hadamard.sign_widths"] = serde_json::json!([3, 3]);
    document["prism.hadamard.sign_values"] = serde_json::json!([1, -1, 1, 1, -1, 1]);
    let error = parse_contract(&document, &fixture_tensor_names())
        .expect_err("a duplicate sign width must be refused");
    assert!(matches!(
        error,
        super::HadamardContractError::DuplicateSignWidth {
            duplicate_sign_width: 3,
        }
    ));
}

#[test]
fn should_refuse_trailing_sign_values() {
    let mut document = minimal_valid_contract_json();
    document["prism.hadamard.sign_values"] = serde_json::json!([1, -1, 1, -1, -1, 1, 1, 1]);
    let error = parse_contract(&document, &fixture_tensor_names())
        .expect_err("trailing sign values must be refused");
    assert!(matches!(
        error,
        super::HadamardContractError::TrailingSignValues {
            consumed_value_count: 7,
            actual_value_count: 8,
        }
    ));
}
