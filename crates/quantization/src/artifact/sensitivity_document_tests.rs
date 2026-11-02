//! Unit tests for the sensitivity document round trip.

use std::collections::BTreeMap;

use super::*;
use crate::bit_width::BitWidth;
use crate::sensitivity::SensitivityMethod;

#[test]
fn should_round_trip_a_table_through_the_optiq_document_shape() {
    let table = SensitivityTable {
        method: SensitivityMethod::ExactBfloat16Reference,
        results: vec![SensitivityResult {
            module_name: "model.layers.0.mlp.up_proj".to_owned(),
            kl_divergence_by_bit_width: BTreeMap::from([
                (BitWidth::new(4).expect("valid"), 0.01),
                (BitWidth::new(8).expect("valid"), 0.002),
            ]),
            logical_parameter_count: 4096,
        }],
    };

    let document = OptiQSensitivityDocument::from_table(
        &table,
        "example/base",
        &[4, 8],
        GroupSize::new(64).expect("valid"),
    );
    let serialized = serde_json::to_value(&document).expect("the document should serialize");
    let round_tripped = document
        .to_table()
        .expect("the document should translate back");

    assert_eq!(serialized["metric"], "kl_divergence_vs_reference");
    assert_eq!(serialized["layers"][0]["sensitivities"]["4"], 0.01);
    assert_eq!(round_tripped.results, table.results);
}

#[test]
fn should_reject_a_document_with_an_unsupported_bit_key() {
    let document = OptiQSensitivityDocument {
        base_model: "example/base".to_owned(),
        candidate_bits: vec![7],
        group_size: 64,
        metric: "kl_divergence_vs_reference".to_owned(),
        layers: vec![SensitivityLayerEntry {
            layer_name: "m".to_owned(),
            sensitivities: BTreeMap::from([("7".to_owned(), 0.1)]),
            logical_parameter_count: 1,
        }],
    };

    assert!(matches!(
        document.to_table(),
        Err(QuantizationError::UnsupportedBitWidth { bits: 7 })
    ));
}

#[test]
fn should_reject_a_bit_key_that_is_not_a_number() {
    let document = OptiQSensitivityDocument {
        base_model: "example/base".to_owned(),
        candidate_bits: vec![4],
        group_size: 64,
        metric: "kl_divergence_vs_reference".to_owned(),
        layers: vec![SensitivityLayerEntry {
            layer_name: "m".to_owned(),
            sensitivities: BTreeMap::from([("four".to_owned(), 0.1)]),
            logical_parameter_count: 1,
        }],
    };

    assert!(matches!(
        document.to_table(),
        Err(QuantizationError::UnsupportedBitWidth { .. })
    ));
}

#[test]
fn should_report_a_missing_sensitivity_document_as_unreadable() {
    let directory = tempfile::tempdir().expect("a temporary directory should be creatable");

    let read_outcome = OptiQSensitivityDocument::read(&directory.path().join("sensitivity.json"));

    assert!(matches!(
        read_outcome,
        Err(QuantizationError::SensitivityDocumentUnreadable { .. })
    ));
}

#[test]
fn should_translate_an_empty_document_to_an_empty_table() {
    let document = OptiQSensitivityDocument {
        base_model: "example/base".to_owned(),
        candidate_bits: vec![4, 8],
        group_size: 64,
        metric: "kl_divergence_vs_reference".to_owned(),
        layers: Vec::new(),
    };

    let table = document.to_table().expect("an empty document translates");

    assert!(table.results.is_empty());
}
