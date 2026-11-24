//! Per-directory discovery of one executable model artifact.
//!
//! Neutral scan orchestration walks model roots (including Hugging Face hub cache entries) and asks
//! this module to classify each candidate directory. The family's registered description then
//! names its own projection, in the family's recognition crate, that turns its family-owned
//! metadata into the neutral DiscoveredModel DTO. The kind-level projections every speech or
//! marked decision family shares live here.

use super::model_family_registry::ModelFamilyRegistry;
use std::fs;
use std::path::Path;

use crate::marked_decision_folder::MarkedDecisionMetadata;
use crate::{
    DecisionModelCapabilities, DiscoveredModel, ModelCapabilities, ModelFamily,
    SpeechModelCapabilities, classified_artifacts, derive_revision_from_config_bytes,
    model_family_classification,
};

pub fn try_discover_model(
    model_family_registry: &ModelFamilyRegistry,
    model_directory: &Path,
) -> Option<DiscoveredModel> {
    let model_id = model_directory
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".to_owned());
    try_discover_model_with_id(model_family_registry, model_directory, &model_id)
}

/// Attempts to discover an executable model from a directory with an explicit model ID.
///
/// This variant accepts a custom `model_id`, useful for Hugging Face hub cache entries
/// where the model_id is derived from the decoded `models--org--repo` directory name
/// rather than the snapshot hash.
pub fn try_discover_model_with_id(
    model_family_registry: &ModelFamilyRegistry,
    model_directory: &Path,
    model_id: &str,
) -> Option<DiscoveredModel> {
    // The typed classifier rejects ambiguous duplicate family markers before
    // the looser metadata document can participate in executable discovery.
    let model_family = model_family_classification::classify_model_directory(
        model_family_registry,
        model_directory,
    )
    .ok()
    .flatten()?;
    (model_family_registry
        .description_of(model_family)?
        .discover_model)(model_directory, model_id)
}

/// Projects a speech folder into the neutral discovery contract: its voices,
/// default voice and pace support, and the weight file's size.
pub fn discover_speech_model(
    model_directory: &Path,
    model_id: &str,
    speech_capabilities: SpeechModelCapabilities,
    weights_file_name: &str,
) -> Option<DiscoveredModel> {
    let configuration_bytes = fs::read(model_directory.join("config.json")).ok()?;
    let (provider_model_id, revision) =
        provenance_or_config_revision(model_directory, &configuration_bytes);
    Some(DiscoveredModel {
        model_id: model_id.to_owned(),
        provider_model_id,
        model_family: speech_capabilities.model_family,
        revision,
        model_directory: model_directory.to_path_buf(),
        model_size_bytes: fs::metadata(model_directory.join(weights_file_name))
            .ok()?
            .len(),
        capabilities: ModelCapabilities::Speech(speech_capabilities),
        license: None,
    })
}

/// Projects a decision-model folder its marker file identified (Kev, Clef)
/// into the neutral discovery contract.
pub fn discover_marked_decision_model(
    model_directory: &Path,
    model_id: &str,
    model_family: ModelFamily,
    family_metadata: MarkedDecisionMetadata,
) -> Option<DiscoveredModel> {
    let settings_bytes = fs::read(model_directory.join(family_metadata.settings_file_name)).ok()?;
    let (provider_model_id, revision) =
        provenance_or_config_revision(model_directory, &settings_bytes);
    Some(DiscoveredModel {
        model_id: model_id.to_owned(),
        provider_model_id,
        model_family,
        revision,
        model_directory: model_directory.to_path_buf(),
        capabilities: ModelCapabilities::Decisions(DecisionModelCapabilities {
            maximum_tokens_per_question: family_metadata.maximum_tokens_per_question,
        }),
        license: None,
        model_size_bytes: family_metadata.model_size_bytes,
    })
}

/// Derives the provider model id and revision for a config-document family:
/// provenance wins when the directory is an immutable Hugging Face snapshot,
/// otherwise the revision is derived from the config bytes themselves.
pub fn provenance_or_config_revision(
    model_directory: &Path,
    config_bytes: &[u8],
) -> (Option<String>, String) {
    let immutable_provenance = classified_artifacts::immutable_model_provenance(model_directory);
    (
        immutable_provenance
            .as_ref()
            .map(|(provider_model_id, _)| provider_model_id.clone()),
        immutable_provenance.map_or_else(
            || derive_revision_from_config_bytes(config_bytes),
            |(_, revision)| revision,
        ),
    )
}

/// Discovers the checkpoints a repository bundles in its subfolders, when
/// its family bundles any. Each becomes a model of its own, named after the
/// repository and the subfolder.
pub fn discover_bundled_checkpoints(
    model_family_registry: &ModelFamilyRegistry,
    repository_model: &DiscoveredModel,
) -> Vec<DiscoveredModel> {
    let Some(discover_bundled_models) = model_family_registry
        .description_of(repository_model.model_family)
        .and_then(|family_description| family_description.discover_bundled_models)
    else {
        return Vec::new();
    };
    discover_bundled_models(model_family_registry, repository_model)
}
