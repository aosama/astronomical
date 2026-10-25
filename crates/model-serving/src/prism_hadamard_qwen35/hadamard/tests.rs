//! Unit contracts for the Hadamard transform manifest.
//!
//! Grouped by concern under one `tests` child of the contract owner; every
//! file in this folder carries the `_tests.rs` suffix. Fixtures are small
//! synthetic documents shaped like the shipped 27B manifest.

mod acceptance_tests;
mod manifest_tests;
mod refusal_tests;
mod sign_table_tests;

use serde_json::{Value, json};

use super::{HadamardContract, HadamardContractError};

/// The bound every malformed-document test must stay under for the oversize
/// guard to be the only refusal in play.
pub(crate) fn oversize_contract_bytes() -> Vec<u8> {
    vec![b' '; 5 * 1024 * 1024]
}

/// A minimal internally consistent contract shaped like the shipped pack:
/// one rotated head projection, one rotated MLP projection, one inverse
/// embedding, and a sign table covering two activation widths.
pub(crate) fn minimal_valid_contract_json() -> Value {
    json!({
        "prism.hadamard.version": 1,
        "prism.hadamard.block_size": 1024,
        "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
        "prism.hadamard.axis": "input-last-dimension",
        "prism.hadamard.sign_mode": "explicit",
        "prism.hadamard.weight_names": [
            "language_model.lm_head.weight",
            "language_model.model.layers.0.mlp.down_proj.weight",
        ],
        "prism.hadamard.inverse_weight_names": [
            "language_model.model.embed_tokens.weight",
        ],
        "prism.hadamard.sign_widths": [4, 3],
        "prism.hadamard.sign_values": [1, -1, 1, -1, -1, 1, 1],
        "prism.hadamard.gdn_v_grouped": true,
    })
}

/// The artifact tensor inventory the fixture manifests resolve against.
pub(crate) fn fixture_tensor_names() -> std::collections::BTreeSet<String> {
    [
        "language_model.lm_head.weight",
        "language_model.model.embed_tokens.weight",
        "language_model.model.layers.0.mlp.down_proj.weight",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

pub(crate) fn parse_contract(
    contract_json: &Value,
    artifact_tensor_names: &std::collections::BTreeSet<String>,
) -> Result<HadamardContract, super::HadamardContractError> {
    HadamardContract::from_json_bytes(contract_json.to_string().as_bytes(), artifact_tensor_names)
}
