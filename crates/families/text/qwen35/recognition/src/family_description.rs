//! How discovery recognizes the Qwen 3.5 and Prism Hadamard chat families and reads one of its
//! folders, as one description the family list names (issue #2473, Step 5).

use std::fs;
use std::path::Path;

use model_discovery::{
    ChatModelCapabilities, DiscoveredModel, ModelCapabilities, ModelFamilyDescription, ModelKind,
    ModelLicense, provenance_or_config_revision,
};

use crate::{
    PRISM_HADAMARD_QWEN35_MODEL_FAMILY, QWEN3_5_MODEL_FAMILY, prism_hadamard_qwen35, qwen3_5,
};

/// How discovery recognizes and reads a folder of this family.
pub const QWEN3_5_FAMILY_DESCRIPTION: ModelFamilyDescription =
    ModelFamilyDescription::recognized_by_model_type(
        QWEN3_5_MODEL_FAMILY,
        ModelKind::Chat,
        qwen3_5::recognizes_model_type,
        discover_qwen3_5_model,
    );

/// How discovery recognizes and reads a folder of this family.
pub const PRISM_HADAMARD_QWEN35_FAMILY_DESCRIPTION: ModelFamilyDescription =
    ModelFamilyDescription::recognized_by_model_type(
        PRISM_HADAMARD_QWEN35_MODEL_FAMILY,
        ModelKind::Chat,
        prism_hadamard_qwen35::recognizes_model_type,
        discover_prism_hadamard_qwen35_model,
    );

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
            supports_vision: qwen3_5::serves_vision(family_metadata.has_vision, &config_value),
            supports_reasoning: family_metadata.supports_reasoning,
            supports_tool_calls: family_metadata.supports_tool_calls,
        }),
        license: None,
        model_size_bytes: family_metadata.model_size_bytes,
    })
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
