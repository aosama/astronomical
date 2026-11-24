//! How discovery recognizes the ModernBERT embedding family and reads one of its
//! folders, as one description the family list names (issue #2473, Step 5).

use std::fs;
use std::path::Path;

use model_discovery::{
    DiscoveredModel, EmbeddingModelCapabilities, ModelCapabilities, ModelFamilyDescription,
    ModelKind, provenance_or_config_revision,
};

use crate::{MODERNBERT_MODEL_FAMILY, modernbert};

/// How discovery recognizes and reads a folder of this family.
pub const MODERNBERT_FAMILY_DESCRIPTION: ModelFamilyDescription =
    ModelFamilyDescription::recognized_by_model_type(
        MODERNBERT_MODEL_FAMILY,
        ModelKind::Embedding,
        modernbert::recognizes_model_type,
        discover_modernbert_model,
    );

/// Projects a ModernBERT embeddings directory into the neutral discovery contract.
pub fn discover_modernbert_model(
    model_directory: &Path,
    model_id: &str,
) -> Option<DiscoveredModel> {
    let config_bytes = fs::read(model_directory.join("config.json")).ok()?;
    let config_value: serde_json::Value = serde_json::from_slice(&config_bytes).ok()?;
    let family_metadata = modernbert::discover_model_metadata(model_directory, &config_value)?;
    let (provider_model_id, revision) =
        provenance_or_config_revision(model_directory, &config_bytes);
    Some(DiscoveredModel {
        model_id: model_id.to_owned(),
        provider_model_id,
        model_family: modernbert::MODERNBERT_MODEL_FAMILY,
        revision,
        model_directory: model_directory.to_path_buf(),
        capabilities: ModelCapabilities::Embeddings(EmbeddingModelCapabilities {
            vector_width: family_metadata.vector_width,
            max_input_tokens: family_metadata.max_input_tokens,
        }),
        license: None,
        model_size_bytes: family_metadata.model_size_bytes,
    })
}
