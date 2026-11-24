//! Per-directory discovery of one executable model artifact.
//!
//! Neutral scan orchestration walks model roots (including Hugging Face hub cache entries) and asks
//! this module to classify each candidate directory. The family's registered description then
//! names the projection below that turns its family-owned metadata into the neutral
//! DiscoveredModel DTO.

use super::model_family_registry::ModelFamilyRegistry;
use std::fs;
use std::path::Path;

use crate::marked_decision_folder::MarkedDecisionMetadata;
use crate::{
    ChatModelCapabilities, DecisionModelCapabilities, DiscoveredModel, EmbeddingModelCapabilities,
    ModelCapabilities, ModelFamily, ModelLicense, SpeechModelCapabilities, classified_artifacts,
    clef, derive_revision_from_config_bytes, flux2_klein, kev, laya, model_family_classification,
    modernbert, prism_hadamard_qwen35, qwen_image_21, qwen3_5,
};
use crate::{
    KOKORO_WEIGHTS_FILE_NAME, PHONON_WEIGHTS_FILE_NAME, kokoro_speech_capabilities,
    phonon_speech_capabilities,
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

/// Projects a Kokoro folder into the neutral discovery contract.
pub fn discover_kokoro_model(model_directory: &Path, model_id: &str) -> Option<DiscoveredModel> {
    discover_speech_model(
        model_directory,
        model_id,
        kokoro_speech_capabilities(model_directory)?,
        KOKORO_WEIGHTS_FILE_NAME,
    )
}

/// Projects a Phonon folder into the neutral discovery contract.
pub fn discover_phonon_model(model_directory: &Path, model_id: &str) -> Option<DiscoveredModel> {
    discover_speech_model(
        model_directory,
        model_id,
        phonon_speech_capabilities(model_directory)?,
        PHONON_WEIGHTS_FILE_NAME,
    )
}

/// Projects a speech folder into the neutral discovery contract: its voices,
/// default voice and pace support, and the weight file's size.
fn discover_speech_model(
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

/// Projects a Kev folder into the neutral discovery contract.
pub fn discover_kev_model(model_directory: &Path, model_id: &str) -> Option<DiscoveredModel> {
    discover_marked_decision_model(
        model_directory,
        model_id,
        kev::KEV_MODEL_FAMILY,
        kev::discover_model_metadata(model_directory)?,
    )
}

/// Projects a Clef folder into the neutral discovery contract.
pub fn discover_clef_model(model_directory: &Path, model_id: &str) -> Option<DiscoveredModel> {
    discover_marked_decision_model(
        model_directory,
        model_id,
        clef::CLEF_MODEL_FAMILY,
        clef::discover_model_metadata(model_directory)?,
    )
}

/// Projects a decision-model folder its marker file identified (Kev, Clef)
/// into the neutral discovery contract.
fn discover_marked_decision_model(
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
pub fn discover_qwen3_5_model(model_directory: &Path, model_id: &str) -> Option<DiscoveredModel> {
    let config_bytes = fs::read(model_directory.join("config.json")).ok()?;
    let config_value: serde_json::Value = serde_json::from_slice(&config_bytes).ok()?;
    let family_metadata = qwen3_5::discover_model_metadata(model_directory, &config_value)?;
    let (provider_model_id, revision) =
        provenance_or_config_revision(model_directory, &config_bytes);
    Some(DiscoveredModel {
        model_id: model_id.to_owned(),
        provider_model_id,
        model_family: qwen3_5::QWEN3_5_MODEL_FAMILY,
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
/// discovery contract from its self-verifying evidence. It is served under
/// the canonical id its evidence verifies, never the folder's name.
pub fn discover_flux2_klein_model(
    model_directory: &Path,
    _folder_model_id: &str,
) -> Option<DiscoveredModel> {
    let verified_evidence = flux2_klein::verify_model_directory(model_directory).ok()?;
    Some(DiscoveredModel {
        model_id: verified_evidence.canonical_model_id,
        provider_model_id: Some(verified_evidence.provider_model_id),
        model_family: flux2_klein::FLUX2_KLEIN_MODEL_FAMILY,
        revision: verified_evidence.revision,
        model_directory: model_directory.to_path_buf(),
        capabilities: ModelCapabilities::ImageGeneration(verified_evidence.capabilities),
        license: Some(verified_evidence.license),
        model_size_bytes: verified_evidence.model_size_bytes,
    })
}

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

/// Projects a prism-hadamard Qwen3.5 pack directory into the neutral discovery
/// contract. The vision tower ships inside the pack, but the executable
/// surface in this phase is text-only, so discovery must not advertise image
/// input.
pub fn discover_prism_hadamard_qwen35_model(
    model_directory: &Path,
    model_id: &str,
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
        model_family: prism_hadamard_qwen35::PRISM_HADAMARD_QWEN35_MODEL_FAMILY,
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
/// from its self-verifying evidence. It is served under the canonical id its
/// evidence verifies, never the folder's name.
pub fn discover_qwen_image_21_model(
    model_directory: &Path,
    _folder_model_id: &str,
) -> Option<DiscoveredModel> {
    let verified_evidence = qwen_image_21::verify_model_directory(model_directory).ok()?;
    Some(DiscoveredModel {
        model_id: verified_evidence.canonical_model_id,
        provider_model_id: Some(verified_evidence.provider_model_id),
        model_family: qwen_image_21::QWEN_IMAGE_21_MODEL_FAMILY,
        revision: verified_evidence.revision,
        model_directory: model_directory.to_path_buf(),
        capabilities: ModelCapabilities::ImageGeneration(verified_evidence.capabilities),
        license: Some(verified_evidence.license),
        model_size_bytes: verified_evidence.model_size_bytes,
    })
}
