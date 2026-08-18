//! Family-owned shallow discovery rules for executable Laguna artifacts.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Component, Path};

use serde::Deserialize;

use super::classified_artifacts::immutable_model_revision;
use crate::{
    LagunaRootChatTemplateSource, LagunaStandaloneChatTemplateState,
    select_laguna_root_chat_template, validate_laguna_standalone_chat_template_role,
};

const MAXIMUM_INDEX_BYTES: u64 = 32 * 1024 * 1024;
const MAXIMUM_TEXT_DOCUMENT_BYTES: u64 = 32 * 1024 * 1024;
const MAXIMUM_TEMPLATE_BYTES: u64 = 512 * 1024;
const MAXIMUM_TEMPLATE_SOURCE_COUNT: usize = 16;
const MAXIMUM_TEMPLATE_INCLUDE_DEPTH: usize = 8;
const SUPPORTED_PARSER_ID: &str = "poolside_v1";
const STANDALONE_CHAT_TEMPLATE_FILE_NAME: &str = "chat_template.jinja";

/// Family-derived metadata returned to neutral discovery orchestration.
pub(super) struct LagunaDiscoveredModelMetadata {
    pub revision: String,
    pub context_window: u32,
    pub max_input_tokens: u32,
    pub max_output_tokens: u32,
    pub has_vision: bool,
    pub supports_reasoning: bool,
    pub supports_tool_calls: bool,
    pub model_size_bytes: u64,
}

/// Recognizes the authoritative Laguna family marker.
pub(super) fn recognizes_model_type(model_type: Option<&str>) -> bool {
    model_type == Some("laguna")
}

/// Predicts whether startup can execute one Laguna artifact without reading weight payloads.
pub(super) fn discover_model_metadata(
    model_directory: &Path,
    config_bytes: &[u8],
    configured_max_output_tokens: u32,
) -> Option<LagunaDiscoveredModelMetadata> {
    let config_document: LagunaConfigDocument = serde_json::from_slice(config_bytes).ok()?;
    if config_document.model_type != "laguna" {
        return None;
    }
    let context_window = config_document
        .text_config
        .as_ref()
        .unwrap_or(&config_document.language_fields)
        .max_position_embeddings?;
    if context_window < 2 || configured_max_output_tokens == 0 {
        return None;
    }
    let max_output_tokens = configured_max_output_tokens.min(context_window - 1);

    // Startup requires all three text sidecars. Discovery reads only bounded
    // metadata needed to predict the supported public text contract.
    read_bounded_file(
        model_directory.join("tokenizer.json"),
        MAXIMUM_TEXT_DOCUMENT_BYTES,
    )?;
    let tokenizer_config_bytes = read_bounded_file(
        model_directory.join("tokenizer_config.json"),
        MAXIMUM_TEXT_DOCUMENT_BYTES,
    )?;
    let generation_config_bytes = read_bounded_file(
        model_directory.join("generation_config.json"),
        MAXIMUM_TEXT_DOCUMENT_BYTES,
    )?;
    let standalone_template_state = standalone_template_state(model_directory)?;
    let selected_root_template =
        select_laguna_root_chat_template(&tokenizer_config_bytes, standalone_template_state)
            .ok()?;
    let root_template_source = match &selected_root_template {
        LagunaRootChatTemplateSource::Embedded {
            template_source, ..
        } => template_source.clone(),
        LagunaRootChatTemplateSource::Standalone => {
            let standalone_template_bytes = read_bounded_file(
                model_directory.join(STANDALONE_CHAT_TEMPLATE_FILE_NAME),
                MAXIMUM_TEMPLATE_BYTES,
            )?;
            String::from_utf8(standalone_template_bytes).ok()?
        }
    };
    let standalone_root_file_name = matches!(
        &selected_root_template,
        LagunaRootChatTemplateSource::Standalone
    )
    .then_some(STANDALONE_CHAT_TEMPLATE_FILE_NAME);
    let selected_template_include_names = validate_template_sources(
        model_directory,
        &root_template_source,
        standalone_root_file_name,
    )?;
    validate_laguna_standalone_chat_template_role(
        &selected_root_template,
        selected_template_include_names.contains(STANDALONE_CHAT_TEMPLATE_FILE_NAME),
    )
    .ok()?;
    let generation_config: LagunaGenerationConfigDocument =
        serde_json::from_slice(&generation_config_bytes).ok()?;
    let supports_reasoning = generation_config.reasoning_parser == SUPPORTED_PARSER_ID;
    let supports_tool_calls = generation_config.tool_call_parser == SUPPORTED_PARSER_ID;
    if !supports_reasoning || !supports_tool_calls {
        return None;
    }

    let model_size_bytes = validate_indexed_payload(model_directory)?;
    let revision = immutable_model_revision(model_directory)?;
    if !is_immutable_revision(&revision) {
        return None;
    }

    Some(LagunaDiscoveredModelMetadata {
        revision,
        context_window,
        max_input_tokens: context_window - max_output_tokens,
        max_output_tokens,
        has_vision: false,
        supports_reasoning,
        supports_tool_calls,
        model_size_bytes,
    })
}

fn validate_indexed_payload(model_directory: &Path) -> Option<u64> {
    let index_bytes = read_bounded_file(
        model_directory.join("model.safetensors.index.json"),
        MAXIMUM_INDEX_BYTES,
    )?;
    let index_document: LagunaShardIndexDocument = serde_json::from_slice(&index_bytes).ok()?;
    if index_document.metadata.total_size == 0 || index_document.weight_map.is_empty() {
        return None;
    }
    let mut shard_file_names = BTreeSet::new();
    for shard_file_name in index_document.weight_map.values() {
        if !is_safe_safetensors_file_name(shard_file_name) {
            return None;
        }
        shard_file_names.insert(shard_file_name.as_str());
    }
    let mut model_size_bytes = 0_u64;
    for shard_file_name in shard_file_names {
        let shard_metadata = fs::metadata(model_directory.join(shard_file_name)).ok()?;
        if !shard_metadata.is_file() || shard_metadata.len() == 0 {
            return None;
        }
        model_size_bytes = model_size_bytes.checked_add(shard_metadata.len())?;
    }
    // Evinced indexes count either tensor payload bytes or serialized shard
    // bytes. Payload bytes cannot exceed their complete serialized files.
    if index_document.metadata.total_size > model_size_bytes {
        return None;
    }
    Some(model_size_bytes)
}

fn is_safe_safetensors_file_name(shard_file_name: &str) -> bool {
    let shard_path = Path::new(shard_file_name);
    !shard_file_name.is_empty()
        && !shard_file_name.contains('\\')
        && !shard_path.is_absolute()
        && shard_path
            .extension()
            .and_then(|extension| extension.to_str())
            == Some("safetensors")
        && shard_path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

fn validate_template_sources(
    model_directory: &Path,
    root_template_source: &str,
    standalone_root_file_name: Option<&str>,
) -> Option<BTreeSet<String>> {
    if root_template_source.is_empty() || root_template_source.len() as u64 > MAXIMUM_TEMPLATE_BYTES
    {
        return None;
    }
    let mut pending_includes = template_include_names(root_template_source)?
        .into_iter()
        .map(|include_name| {
            (
                include_name,
                1_usize,
                standalone_root_file_name
                    .map(|file_name| vec![file_name.to_owned()])
                    .unwrap_or_default(),
            )
        })
        .collect::<Vec<_>>();
    let mut included_names = BTreeSet::new();
    let mut total_template_bytes = root_template_source.len() as u64;
    while let Some((include_name, depth, ancestors)) = pending_includes.pop() {
        if depth > MAXIMUM_TEMPLATE_INCLUDE_DEPTH
            || ancestors.contains(&include_name)
            || !is_safe_template_include_name(&include_name)
        {
            return None;
        }
        if !included_names.insert(include_name.clone()) {
            continue;
        }
        if included_names.len() + 1 > MAXIMUM_TEMPLATE_SOURCE_COUNT {
            return None;
        }
        let include_bytes =
            read_bounded_file(model_directory.join(&include_name), MAXIMUM_TEMPLATE_BYTES)?;
        total_template_bytes = total_template_bytes.checked_add(include_bytes.len() as u64)?;
        if total_template_bytes > MAXIMUM_TEMPLATE_BYTES {
            return None;
        }
        let include_source = std::str::from_utf8(&include_bytes).ok()?;
        let mut child_ancestors = ancestors;
        child_ancestors.push(include_name);
        for child_include_name in template_include_names(include_source)? {
            pending_includes.push((child_include_name, depth + 1, child_ancestors.clone()));
        }
    }
    Some(included_names)
}

fn standalone_template_state(model_directory: &Path) -> Option<LagunaStandaloneChatTemplateState> {
    match fs::metadata(model_directory.join(STANDALONE_CHAT_TEMPLATE_FILE_NAME)) {
        Ok(template_metadata) if template_metadata.is_file() && template_metadata.len() == 0 => {
            Some(LagunaStandaloneChatTemplateState::Empty)
        }
        Ok(template_metadata) if template_metadata.is_file() => {
            Some(LagunaStandaloneChatTemplateState::NonEmpty)
        }
        Ok(_) => None,
        Err(metadata_error) if metadata_error.kind() == std::io::ErrorKind::NotFound => {
            Some(LagunaStandaloneChatTemplateState::Missing)
        }
        Err(_) => None,
    }
}

/// Finds the static single-quoted include syntax accepted by Laguna startup.
fn template_include_names(template_source: &str) -> Option<Vec<String>> {
    let mut include_names = Vec::new();
    let mut remaining_source = template_source;
    while let Some(directive_start) = remaining_source.find("{%") {
        remaining_source = &remaining_source[directive_start + 2..];
        let directive_end = remaining_source.find("%}")?;
        let directive_body = remaining_source[..directive_end]
            .trim()
            .trim_start_matches('-')
            .trim_end_matches('-')
            .trim();
        if let Some(include_expression) = directive_body.strip_prefix("include") {
            let include_name = include_expression
                .trim()
                .strip_prefix('\'')?
                .strip_suffix('\'')?;
            if include_name.is_empty() || include_name.contains('\'') {
                return None;
            }
            include_names.push(include_name.to_owned());
        }
        remaining_source = &remaining_source[directive_end + 2..];
    }
    Some(include_names)
}

fn is_safe_template_include_name(include_name: &str) -> bool {
    let include_path = Path::new(include_name);
    include_name.len() <= 255
        && !include_name.contains('\\')
        && !include_path.is_absolute()
        && include_path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

fn read_bounded_file(file_path: impl AsRef<Path>, maximum_bytes: u64) -> Option<Vec<u8>> {
    let file_path = file_path.as_ref();
    let file_metadata = fs::metadata(file_path).ok()?;
    if !file_metadata.is_file() || file_metadata.len() == 0 || file_metadata.len() > maximum_bytes {
        return None;
    }
    let file_bytes = fs::read(file_path).ok()?;
    (file_bytes.len() as u64 <= maximum_bytes).then_some(file_bytes)
}

fn is_immutable_revision(revision: &str) -> bool {
    revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[derive(Deserialize)]
struct LagunaConfigDocument {
    model_type: String,
    #[serde(default)]
    text_config: Option<LagunaLanguageFields>,
    #[serde(flatten)]
    language_fields: LagunaLanguageFields,
}

#[derive(Default, Deserialize)]
struct LagunaLanguageFields {
    #[serde(default)]
    max_position_embeddings: Option<u32>,
}

#[derive(Deserialize)]
struct LagunaGenerationConfigDocument {
    reasoning_parser: String,
    tool_call_parser: String,
}

#[derive(Deserialize)]
struct LagunaShardIndexDocument {
    metadata: LagunaShardIndexMetadata,
    weight_map: std::collections::BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct LagunaShardIndexMetadata {
    total_size: u64,
}
