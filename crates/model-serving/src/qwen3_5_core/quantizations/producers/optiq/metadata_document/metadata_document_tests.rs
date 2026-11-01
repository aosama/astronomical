use serde_json::{Value, json};

use super::*;

fn mixed_document() -> Value {
    json!({
        "method": "optiq_mixed_precision",
        "base_model": "synthetic/base",
        "default_bits": 4,
        "group_size": 64,
        "bit_widths_used": [4, 8],
        "layers_per_bit_width": {"4": 1, "8": 1},
        "n_quantized_layers": 2,
        "per_layer": {
            "language_model.model.embed_tokens": {"bits": 8, "group_size": 64},
            "language_model.model.layers.0.mlp.up_proj": {"bits": 4, "group_size": 64}
        }
    })
}

fn static_document() -> Value {
    json!({
        "method": "static_mixed_precision",
        "base_model": "synthetic/base",
        "reference": "structural_rules",
        "target_bpw": 4.5,
        "achieved_bpw": 4.6,
        "n_high_bits": 1,
        "n_low_bits": 1,
        "threshold": 0.0,
        "per_layer": {"language_model.lm_head": {"bits": 6, "group_size": 64}}
    })
}

fn decode(document: &Value) -> Result<OptiQMetadataDocument, serde_json::Error> {
    OptiQMetadataDocument::from_json_bytes(&serde_json::to_vec(document).expect("serializes"))
}

#[test]
fn should_accept_the_mixed_precision_shape() {
    let document = decode(&mixed_document()).expect("mixed shape should decode");

    assert!(matches!(document, OptiQMetadataDocument::Mixed(_)));
    assert_eq!(document.into_per_layer().len(), 2);
}

#[test]
fn should_accept_the_mixed_precision_recipe_provenance_fields() {
    let mut document = mixed_document();
    document["recipe_reused_from"] = json!("synthetic/sibling");
    document["recipe_note"] = json!("allocation reused");

    assert!(decode(&document).is_ok());
}

#[test]
fn should_accept_the_static_shape() {
    let document = decode(&static_document()).expect("static shape should decode");

    assert!(matches!(document, OptiQMetadataDocument::Static(_)));
    assert_eq!(document.into_per_layer().len(), 1);
}

#[test]
fn should_route_an_unversioned_method_to_the_static_shape() {
    let mut document = static_document();
    document["method"] = json!("sensitivity_pipeline");

    assert!(matches!(
        decode(&document),
        Ok(OptiQMetadataDocument::Static(_))
    ));
}

#[test]
fn should_reject_unknown_fields_in_the_mixed_shape() {
    let mut document = mixed_document();
    document["surprise"] = json!(1);

    assert!(decode(&document).is_err());
}

#[test]
fn should_reject_static_fields_in_a_mixed_document() {
    let mut document = mixed_document();
    document["target_bpw"] = json!(4.5);

    assert!(decode(&document).is_err());
}

#[test]
fn should_reject_mixed_fields_in_a_static_document() {
    let mut document = static_document();
    document["default_bits"] = json!(4);

    assert!(decode(&document).is_err());
}

#[test]
fn should_reject_a_mixed_document_missing_required_fields() {
    let mut document = mixed_document();
    document
        .as_object_mut()
        .expect("object")
        .remove("default_bits");

    assert!(decode(&document).is_err());
}

#[test]
fn should_reject_a_per_layer_entry_carrying_a_mode() {
    let mut document = mixed_document();
    document["per_layer"]["language_model.lm_head"] =
        json!({"bits": 4, "group_size": 64, "mode": "affine"});

    assert!(decode(&document).is_err());
}

#[test]
fn should_reject_a_document_without_a_method() {
    let mut document = static_document();
    document.as_object_mut().expect("object").remove("method");

    assert!(decode(&document).is_err());
}

#[test]
fn should_recognize_only_the_exact_mixed_precision_method() {
    assert!(is_mixed_precision_method("optiq_mixed_precision"));
    assert!(!is_mixed_precision_method("static_mixed_precision"));
    assert!(!is_mixed_precision_method(""));
}
