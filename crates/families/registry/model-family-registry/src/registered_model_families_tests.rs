use std::collections::HashSet;

use super::REGISTERED_MODEL_FAMILIES;
use crate::{
    CLEF_MODEL_FAMILY, FLUX2_KLEIN_MODEL_FAMILY, KEV_MODEL_FAMILY, KOKORO_MODEL_FAMILY,
    LAYA_MODEL_FAMILY, MODERNBERT_MODEL_FAMILY, PHONON_MODEL_FAMILY,
    PRISM_HADAMARD_QWEN35_MODEL_FAMILY, QWEN_IMAGE_21_MODEL_FAMILY, QWEN3_5_MODEL_FAMILY,
};
use crate::{MODEL_FAMILY_REGISTRY, model_family_named, model_kind_of};
use model_discovery::{ModelFamily, ModelKind};

/// Every family constant with the kind it declares. The identifiers are the
/// spellings the catalog, stored settings and the worker wire already use.
const EXPECTED_FAMILY_KINDS: [(ModelFamily, &str, ModelKind); 10] = [
    (
        FLUX2_KLEIN_MODEL_FAMILY,
        "flux2_klein",
        ModelKind::ImageGeneration,
    ),
    (
        QWEN_IMAGE_21_MODEL_FAMILY,
        "qwen_image_21",
        ModelKind::ImageGeneration,
    ),
    (KEV_MODEL_FAMILY, "kev", ModelKind::Decision),
    (CLEF_MODEL_FAMILY, "clef", ModelKind::Decision),
    (KOKORO_MODEL_FAMILY, "kokoro", ModelKind::Speech),
    (PHONON_MODEL_FAMILY, "phonon", ModelKind::Speech),
    (QWEN3_5_MODEL_FAMILY, "qwen3_5", ModelKind::Chat),
    (
        PRISM_HADAMARD_QWEN35_MODEL_FAMILY,
        "prism_hadamard_qwen35",
        ModelKind::Chat,
    ),
    (MODERNBERT_MODEL_FAMILY, "modernbert", ModelKind::Embedding),
    (LAYA_MODEL_FAMILY, "laya", ModelKind::Decision),
];

#[test]
fn should_register_every_family_constant_with_its_identifier_and_kind() {
    for (model_family, expected_identifier, expected_kind) in EXPECTED_FAMILY_KINDS {
        assert_eq!(model_family.as_str(), expected_identifier);
        assert_eq!(
            model_kind_of(model_family),
            Some(expected_kind),
            "{expected_identifier}"
        );
        assert_eq!(model_family_named(expected_identifier), Some(model_family));
    }
    assert_eq!(REGISTERED_MODEL_FAMILIES.len(), EXPECTED_FAMILY_KINDS.len());
}

#[test]
fn should_register_each_identifier_once() {
    let mut registered_identifiers = HashSet::new();
    for family_description in REGISTERED_MODEL_FAMILIES {
        assert!(
            registered_identifiers.insert(family_description.model_family.as_str()),
            "{} is registered twice",
            family_description.model_family.as_str()
        );
    }
}

#[test]
fn should_give_every_family_a_way_to_be_recognized() {
    for family_description in REGISTERED_MODEL_FAMILIES {
        assert!(
            family_description.recognizes_pipeline_index.is_some()
                || family_description.recognizes_directory.is_some()
                || family_description.recognizes_model_type.is_some(),
            "{} can never be classified",
            family_description.model_family.as_str()
        );
    }
}

#[test]
fn should_find_no_description_for_an_unregistered_identifier() {
    assert!(MODEL_FAMILY_REGISTRY.description_named("qwen3_6").is_none());
    assert_eq!(model_family_named(""), None);
    assert_eq!(model_kind_of(ModelFamily::named("qwen3_6")), None);
}

#[test]
fn should_let_only_laya_bundle_further_models() {
    for family_description in REGISTERED_MODEL_FAMILIES {
        assert_eq!(
            family_description.discover_bundled_models.is_some(),
            family_description.model_family == LAYA_MODEL_FAMILY,
            "{}",
            family_description.model_family.as_str()
        );
    }
}
