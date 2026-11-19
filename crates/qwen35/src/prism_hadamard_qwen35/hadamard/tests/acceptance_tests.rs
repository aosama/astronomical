//! Acceptance contracts: a well-formed manifest parses and answers the
//! serving path's lookup questions exactly as declared.

use super::{fixture_tensor_names, minimal_valid_contract_json, parse_contract};

#[test]
fn should_parse_the_shipped_contract_shape() {
    let contract = parse_contract(&minimal_valid_contract_json(), &fixture_tensor_names())
        .expect("the minimal fixture contract should parse");
    assert_eq!(contract.block_size(), 1024);
    assert!(contract.gdn_layout_is_grouped());
}

#[test]
fn should_report_rotated_inverse_and_untouched_tensors() {
    let contract = parse_contract(&minimal_valid_contract_json(), &fixture_tensor_names())
        .expect("the minimal fixture contract should parse");
    assert!(contract.is_rotated("language_model.lm_head.weight"));
    assert!(contract.is_rotated("language_model.model.layers.0.mlp.down_proj.weight"));
    assert!(contract.is_inverse("language_model.model.embed_tokens.weight"));
    assert!(!contract.is_rotated("language_model.model.embed_tokens.weight"));
    assert!(!contract.is_inverse("language_model.lm_head.weight"));
    assert!(!contract.is_rotated("language_model.model.layers.1.mlp.gate_proj.weight"));
}

#[test]
fn should_serve_sign_vectors_by_activation_width() {
    let contract = parse_contract(&minimal_valid_contract_json(), &fixture_tensor_names())
        .expect("the minimal fixture contract should parse");
    assert_eq!(contract.sign_vector(4), Some(&[1, -1, 1, -1][..]));
    assert_eq!(contract.sign_vector(3), Some(&[-1, 1, 1][..]));
    assert_eq!(contract.sign_vector(999), None);
}
