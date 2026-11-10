//! Per-directory discovery of one executable model artifact.
//!
//! Neutral scan orchestration walks model roots (including Hugging Face cache entries) and asks
//! this module to classify each candidate directory and project its family-owned metadata into
//! the neutral DiscoveredModel DTO.

use std::fs;
use std::path::Path;

use crate::model_discovery::{
    ChatModelCapabilities, DecisionModelCapabilities, DiscoveredModel, EmbeddingModelCapabilities,
    ModelCapabilities, ModelFamily, ModelLicense, classified_artifacts,
    derive_revision_from_config_bytes, flux2_klein, laya, model_family, modernbert,
    prism_hadamard_qwen35, qwen_image_21, qwen3_5,
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
        ModelFamily::Qwen3_5 => discover_qwen3_5_model(model_directory, model_id, model_family),
        ModelFamily::Flux2Klein => discover_flux2_klein_model(model_directory, model_family),
        ModelFamily::ModernBert | ModelFamily::Laya => {
            discover_encoder_model(model_directory, model_id, model_family)
        }
        ModelFamily::PrismHadamardQwen35 => {
            discover_prism_hadamard_qwen35_model(model_directory, model_id, model_family)
        }
        // The native T2I engine is wired end to end (worker factory arm, daemon image
        // policy, preflight, catalog), so a verified Qwen-Image-2.1 artifact now advertises
        // with the same evidence shape as FLUX.2 Klein.
        ModelFamily::QwenImage21 => discover_qwen_image21_model(model_directory, model_family),
    }
}

/// Discovers a model built on a ModernBERT encoder: the decision model when
/// the family says so, and the embedding model otherwise.
fn discover_encoder_model(
    model_directory: &Path,
    model_id: &str,
    encoder_family: ModelFamily,
) -> Option<DiscoveredModel> {
    if encoder_family == ModelFamily::Laya {
        return discover_laya_model(model_directory, model_id, encoder_family);
    }
    discover_modernbert_model(model_directory, model_id, encoder_family)
}

/// Derives the provider model id and revision for a config-document family:
/// provenance wins when the directory is an immutable Hugging Face snapshot,
/// otherwise the revision is derived from the config bytes themselves.
fn provenance_or_config_revision(
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

/// Projects a Qwen3.5 chat directory into the neutral discovery contract.
fn discover_qwen3_5_model(
    model_directory: &Path,
    model_id: &str,
    model_family: ModelFamily,
) -> Option<DiscoveredModel> {
    let config_bytes = fs::read(model_directory.join("config.json")).ok()?;
    let config_value: serde_json::Value = serde_json::from_slice(&config_bytes).ok()?;
    let family_metadata = qwen3_5::discover_model_metadata(model_directory, &config_value)?;
    let (provider_model_id, revision) =
        provenance_or_config_revision(model_directory, &config_bytes);
    Some(DiscoveredModel {
        model_id: model_id.to_owned(),
        provider_model_id,
        model_family,
        revision,
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

/// Projects a FLUX.2 Klein image-generation directory into the neutral
/// discovery contract from its self-verifying evidence.
fn discover_flux2_klein_model(
    model_directory: &Path,
    model_family: ModelFamily,
) -> Option<DiscoveredModel> {
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

/// Projects a ModernBERT embeddings directory into the neutral discovery contract.
fn discover_modernbert_model(
    model_directory: &Path,
    model_id: &str,
    model_family: ModelFamily,
) -> Option<DiscoveredModel> {
    let config_bytes = fs::read(model_directory.join("config.json")).ok()?;
    let config_value: serde_json::Value = serde_json::from_slice(&config_bytes).ok()?;
    let family_metadata = modernbert::discover_model_metadata(model_directory, &config_value)?;
    let (provider_model_id, revision) =
        provenance_or_config_revision(model_directory, &config_bytes);
    Some(DiscoveredModel {
        model_id: model_id.to_owned(),
        provider_model_id,
        model_family,
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

/// Projects a Laya decision-model directory into the neutral discovery contract.
fn discover_laya_model(
    model_directory: &Path,
    model_id: &str,
    model_family: ModelFamily,
) -> Option<DiscoveredModel> {
    let config_bytes = fs::read(model_directory.join("config.json")).ok()?;
    let config_value: serde_json::Value = serde_json::from_slice(&config_bytes).ok()?;
    let family_metadata = laya::discover_model_metadata(model_directory, &config_value)?;
    let (provider_model_id, revision) =
        provenance_or_config_revision(model_directory, &config_bytes);
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

/// Discovers the checkpoints a Laya repository bundles in its subfolders.
///
/// Each becomes a model of its own, named after the repository and the
/// subfolder. Other families bundle nothing, so they yield no models here.
pub(super) fn discover_bundled_checkpoints(
    repository_model: &DiscoveredModel,
) -> Vec<DiscoveredModel> {
    if repository_model.model_family != ModelFamily::Laya {
        return Vec::new();
    }
    let mut bundled_models = Vec::new();
    for bundled_checkpoint in laya::bundled_checkpoints(&repository_model.model_directory) {
        let bundled_model_id = laya::bundled_checkpoint_model_id(
            &repository_model.model_id,
            &bundled_checkpoint.checkpoint_name,
        );
        if let Some(bundled_model) =
            try_discover_model_with_id(&bundled_checkpoint.checkpoint_directory, &bundled_model_id)
        {
            bundled_models.push(bundled_model);
        }
    }
    bundled_models
}

/// Projects a prism-hadamard Qwen3.5 pack directory into the neutral discovery
/// contract. The vision tower ships inside the pack, but the executable
/// surface in this phase is text-only, so discovery must not advertise image
/// input.
fn discover_prism_hadamard_qwen35_model(
    model_directory: &Path,
    model_id: &str,
    model_family: ModelFamily,
) -> Option<DiscoveredModel> {
    let config_bytes = fs::read(model_directory.join("config.json")).ok()?;
    let config_value: serde_json::Value = serde_json::from_slice(&config_bytes).ok()?;
    let family_metadata =
        prism_hadamard_qwen35::discover_model_metadata(model_directory, &config_value)?;
    let (provider_model_id, revision) =
        provenance_or_config_revision(model_directory, &config_bytes);
    Some(DiscoveredModel {
        model_id: model_id.to_owned(),
        provider_model_id,
        model_family,
        revision,
        model_directory: model_directory.to_path_buf(),
        capabilities: ModelCapabilities::Chat(ChatModelCapabilities {
            context_window: family_metadata.context_window,
            max_input_tokens: family_metadata.max_input_tokens,
            max_output_tokens: family_metadata.max_output_tokens,
            supports_vision: false,
            supports_reasoning: family_metadata.supports_reasoning,
            supports_tool_calls: family_metadata.supports_tool_calls,
        }),
        license: Some(ModelLicense::Apache20),
        model_size_bytes: family_metadata.model_size_bytes,
    })
}

/// Projects a Qwen-Image-2.1 directory into the neutral discovery contract
/// from its self-verifying evidence.
fn discover_qwen_image21_model(
    model_directory: &Path,
    model_family: ModelFamily,
) -> Option<DiscoveredModel> {
    let verified_evidence = qwen_image_21::verify_model_directory(model_directory).ok()?;
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
