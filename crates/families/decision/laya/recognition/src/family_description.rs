//! How discovery recognizes the Laya decision family and reads one of its
//! folders, as one description the family list names (issue #2473, Step 5).

use std::fs;
use std::path::Path;

use model_discovery::*;

use crate::*;

/// How discovery recognizes and reads a folder of this family.
pub const LAYA_FAMILY_DESCRIPTION: ModelFamilyDescription =
    ModelFamilyDescription::recognized_by_model_type(
        LAYA_MODEL_FAMILY,
        ModelKind::Decision,
        laya::recognizes_model_type,
        discover_laya_model,
    )
    .with_bundled_models(discover_laya_bundled_checkpoints);

/// Projects a Laya decision-model directory into the neutral discovery contract.
pub fn discover_laya_model(model_directory: &Path, model_id: &str) -> Option<DiscoveredModel> {
    let config_bytes = fs::read(model_directory.join("config.json")).ok()?;
    let config_value: serde_json::Value = serde_json::from_slice(&config_bytes).ok()?;
    let family_metadata = laya::discover_model_metadata(model_directory, &config_value)?;
    let (provider_model_id, revision) =
        provenance_or_config_revision(model_directory, &config_bytes);
    Some(DiscoveredModel {
        model_id: model_id.to_owned(),
        provider_model_id,
        model_family: laya::LAYA_MODEL_FAMILY,
        revision,
        model_directory: model_directory.to_path_buf(),
        capabilities: ModelCapabilities::Decisions(DecisionModelCapabilities {
            maximum_tokens_per_question: family_metadata.maximum_tokens_per_question,
        }),
        license: None,
        model_size_bytes: family_metadata.model_size_bytes,
    })
}

/// The checkpoints a Laya repository bundles, each a model of its own.
pub fn discover_laya_bundled_checkpoints(
    model_family_registry: &ModelFamilyRegistry,
    repository_model: &DiscoveredModel,
) -> Vec<DiscoveredModel> {
    let mut bundled_models = Vec::new();
    for bundled_checkpoint in laya::bundled_checkpoints(&repository_model.model_directory) {
        let bundled_model_id = laya::bundled_checkpoint_model_id(
            &repository_model.model_id,
            &bundled_checkpoint.checkpoint_name,
        );
        if let Some(bundled_model) = try_discover_model_with_id(
            model_family_registry,
            &bundled_checkpoint.checkpoint_directory,
            &bundled_model_id,
        ) {
            bundled_models.push(bundled_model);
        }
    }
    bundled_models
}
