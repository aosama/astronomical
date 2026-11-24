use std::path::Path;

use super::{ModelFamily, ModelFamilyDescription, ModelKind};
use crate::model_discovery::DiscoveredModel;

fn discovers_nothing(_model_directory: &Path, _model_id: &str) -> Option<DiscoveredModel> {
    None
}

fn names_fictional_type(model_type: Option<&str>) -> bool {
    model_type == Some("fictional")
}

fn holds_any_marker(_model_directory: &Path) -> bool {
    true
}

fn refuses_to_parse(pipeline_index_bytes: &[u8]) -> Result<bool, serde_json::Error> {
    serde_json::from_slice::<serde_json::Value>(pipeline_index_bytes).map(|_| true)
}

const FICTIONAL_FAMILY: ModelFamily = ModelFamily::named("fictional");

#[test]
fn should_claim_only_the_model_type_its_recognizer_names() {
    let family_description = ModelFamilyDescription::recognized_by_model_type(
        FICTIONAL_FAMILY,
        ModelKind::Chat,
        names_fictional_type,
        discovers_nothing,
    );

    assert!(family_description.claims_model_type(Some("fictional")));
    assert!(!family_description.claims_model_type(Some("qwen3_5")));
    assert!(!family_description.claims_model_type(None));
}

#[test]
fn should_claim_nothing_it_has_no_recognizer_for() {
    let family_description = ModelFamilyDescription::recognized_by_model_type(
        FICTIONAL_FAMILY,
        ModelKind::Chat,
        names_fictional_type,
        discovers_nothing,
    );

    assert!(!family_description.claims_directory(Path::new("fictional-model")));
    assert!(matches!(
        family_description.claims_pipeline_index(b"not json"),
        Ok(false)
    ));
}

#[test]
fn should_claim_a_folder_by_its_marker_file() {
    let family_description = ModelFamilyDescription::recognized_by_marker_file(
        FICTIONAL_FAMILY,
        ModelKind::Speech,
        holds_any_marker,
        discovers_nothing,
    );

    assert!(family_description.claims_directory(Path::new("fictional-model")));
    assert!(!family_description.claims_model_type(Some("fictional")));
}

#[test]
fn should_pass_on_a_pipeline_index_it_cannot_parse() {
    let family_description = ModelFamilyDescription::recognized_by_pipeline_index(
        FICTIONAL_FAMILY,
        ModelKind::ImageGeneration,
        refuses_to_parse,
        discovers_nothing,
    );

    assert!(
        family_description
            .claims_pipeline_index(b"not json")
            .is_err()
    );
    assert!(matches!(
        family_description.claims_pipeline_index(b"{}"),
        Ok(true)
    ));
}

#[test]
fn should_read_a_marked_build_revision_from_the_file_it_names() {
    let family_description = ModelFamilyDescription::recognized_by_pipeline_index(
        FICTIONAL_FAMILY,
        ModelKind::ImageGeneration,
        refuses_to_parse,
        discovers_nothing,
    );
    assert_eq!(family_description.revision_file_name, "config.json");

    let marked_build_description = family_description
        .also_recognized_by_marker_file(holds_any_marker, "transformer/index.json");

    assert_eq!(
        marked_build_description.revision_file_name,
        "transformer/index.json"
    );
    assert!(marked_build_description.claims_directory(Path::new("fictional-model")));
}
