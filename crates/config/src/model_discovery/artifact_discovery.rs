//! Per-directory discovery of one executable model artifact.
//!
//! Neutral scan orchestration walks model roots (including Hugging Face cache entries) and asks
//! this module to classify each candidate directory and project its family-owned metadata into
//! the neutral DiscoveredModel DTO.

use std::fs;
use std::path::Path;

use crate::model_discovery::{
    ChatModelCapabilities, DiscoveredModel, EmbeddingModelCapabilities, ModelCapabilities,
    ModelFamily, ModelLicense, classified_artifacts, derive_revision_from_config_bytes,
    flux2_klein, k2_horizon_mova, laguna, model_family, modernbert, qwen3_5,
};

pub fn try_discover_model(model_directory: &Path) -> Option<DiscoveredModel> {
    let model_id = model_directory
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".to_owned());
    try_discover_model_with_id(model_directory, &model_id)
}

/// Attempts to discover an executable model from a directory with an explicit model ID.
///
/// This variant accepts a custom `model_id`, useful for HuggingFace cache entries
/// where the model_id is derived from the decoded `models--org--repo` directory name
/// rather than the snapshot hash.
pub fn try_discover_model_with_id(
    model_directory: &Path,
    model_id: &str,
) -> Option<DiscoveredModel> {
    // The typed classifier rejects ambiguous duplicate family markers before
    // the looser metadata document can participate in executable discovery.
    let model_family = model_family::classify_model_directory(model_directory)
        .ok()
        .flatten()?;
    match model_family {
        ModelFamily::Qwen3_5 => {
            let config_bytes = fs::read(model_directory.join("config.json")).ok()?;
            let config_value: serde_json::Value = serde_json::from_slice(&config_bytes).ok()?;
            let family_metadata = qwen3_5::discover_model_metadata(model_directory, &config_value)?;
            let immutable_provenance =
                classified_artifacts::immutable_model_provenance(model_directory);
            Some(DiscoveredModel {
                model_id: model_id.to_owned(),
                provider_model_id: immutable_provenance
                    .as_ref()
                    .map(|(provider_model_id, _)| provider_model_id.clone()),
                model_family,
                revision: immutable_provenance.map_or_else(
                    || derive_revision_from_config_bytes(&config_bytes),
                    |(_, revision)| revision,
                ),
                model_directory: model_directory.to_path_buf(),
                capabilities: ModelCapabilities::Chat(ChatModelCapabilities {
                    context_window: family_metadata.context_window,
                    max_input_tokens: family_metadata.max_input_tokens,
                    max_output_tokens: family_metadata.max_output_tokens,
                    supports_vision: family_metadata.has_vision,
                    supports_reasoning: family_metadata.supports_reasoning,
                    supports_tool_calls: family_metadata.supports_tool_calls,
                }),
                license: None,
                model_size_bytes: family_metadata.model_size_bytes,
            })
        }
        ModelFamily::Laguna => {
            let config_bytes = fs::read(model_directory.join("config.json")).ok()?;
            let laguna_metadata = laguna::discover_model_metadata(model_directory, &config_bytes)?;
            let provider_model_id =
                classified_artifacts::immutable_model_provenance(model_directory)
                    .filter(|(_, revision)| revision == &laguna_metadata.revision)
                    .map(|(provider_model_id, _)| provider_model_id);
            Some(DiscoveredModel {
                model_id: model_id.to_owned(),
                provider_model_id,
                model_family,
                revision: laguna_metadata.revision,
                model_directory: model_directory.to_path_buf(),
                capabilities: ModelCapabilities::Chat(ChatModelCapabilities {
                    context_window: laguna_metadata.context_window,
                    max_input_tokens: laguna_metadata.max_input_tokens,
                    max_output_tokens: laguna_metadata.max_output_tokens,
                    supports_vision: laguna_metadata.has_vision,
                    supports_reasoning: laguna_metadata.supports_reasoning,
                    supports_tool_calls: laguna_metadata.supports_tool_calls,
                }),
                license: None,
                model_size_bytes: laguna_metadata.model_size_bytes,
            })
        }
        ModelFamily::Flux2Klein => {
            let verified_evidence = flux2_klein::verify_model_directory(model_directory).ok()?;
            Some(DiscoveredModel {
                model_id: verified_evidence.canonical_model_id,
                provider_model_id: Some(verified_evidence.provider_model_id),
                model_family,
                revision: verified_evidence.revision,
                model_directory: model_directory.to_path_buf(),
                capabilities: ModelCapabilities::ImageGeneration(verified_evidence.capabilities),
                license: Some(verified_evidence.license),
                model_size_bytes: verified_evidence.model_size_bytes,
            })
        }
        ModelFamily::ModernBert => {
            let config_bytes = fs::read(model_directory.join("config.json")).ok()?;
            let config_value: serde_json::Value = serde_json::from_slice(&config_bytes).ok()?;
            let family_metadata =
                modernbert::discover_model_metadata(model_directory, &config_value)?;
            let immutable_provenance =
                classified_artifacts::immutable_model_provenance(model_directory);
            Some(DiscoveredModel {
                model_id: model_id.to_owned(),
                provider_model_id: immutable_provenance
                    .as_ref()
                    .map(|(provider_model_id, _)| provider_model_id.clone()),
                model_family,
                revision: immutable_provenance.map_or_else(
                    || derive_revision_from_config_bytes(&config_bytes),
                    |(_, revision)| revision,
                ),
                model_directory: model_directory.to_path_buf(),
                capabilities: ModelCapabilities::Embeddings(EmbeddingModelCapabilities {
                    vector_width: family_metadata.vector_width,
                    max_input_tokens: family_metadata.max_input_tokens,
                }),
                license: None,
                model_size_bytes: family_metadata.model_size_bytes,
            })
        }
        ModelFamily::K2HorizonMoVA => {
            let config_bytes = fs::read(model_directory.join("config.json")).ok()?;
            let family_metadata =
                k2_horizon_mova::discover_model_metadata(model_directory, &config_bytes)?;
            let immutable_provenance =
                classified_artifacts::immutable_model_provenance(model_directory);
            Some(DiscoveredModel {
                model_id: model_id.to_owned(),
                provider_model_id: immutable_provenance
                    .as_ref()
                    .map(|(provider_model_id, _)| provider_model_id.clone()),
                model_family,
                revision: immutable_provenance.map_or_else(
                    || derive_revision_from_config_bytes(&config_bytes),
                    |(_, revision)| revision,
                ),
                model_directory: model_directory.to_path_buf(),
                capabilities: ModelCapabilities::Chat(ChatModelCapabilities {
                    context_window: family_metadata.context_window,
                    max_input_tokens: family_metadata.max_input_tokens,
                    max_output_tokens: family_metadata.max_output_tokens,
                    supports_vision: family_metadata.has_vision,
                    supports_reasoning: family_metadata.supports_reasoning,
                    supports_tool_calls: family_metadata.supports_tool_calls,
                }),
                license: Some(ModelLicense::Apache20),
                model_size_bytes: family_metadata.model_size_bytes,
            })
        }
        // Classification is intentionally broader than executable discovery.
        ModelFamily::DeepSeekV4 => None,
    }
}
