use std::collections::BTreeMap;

use serde_json::{Map, Value};
use tokenizers::Tokenizer;

use crate::laguna::normalization::LagunaTargetContract;

use super::LagunaTextArtifactError;
use super::artifact_descriptor::LagunaTextArtifactDescriptor;
use super::artifact_documents::{
    model_language_fields, object_fields, parse_configured_tokens, parse_json_document,
    parse_optional_token_id_set, parse_token_id_set, parse_tokenizer_added_tokens, required_u32,
    validate_bidirectional_added_tokens, validate_configured_control_ids,
    validate_optional_matching_id, validate_tokenizer_vocabulary,
};
use super::artifact_sampler::normalize_sampler_config;
use super::artifact_template::{required_parser_id, resolve_template_sources};
use super::template_contract::derive_template_contract;
use super::template_program::LagunaTemplateProgram;

/// Borrowed artifact bytes supplied after canonical Laguna model normalization.
pub struct LagunaTextArtifactSources<'a> {
    pub model_config_bytes: &'a [u8],
    pub tokenizer_bytes: &'a [u8],
    pub tokenizer_config_bytes: &'a [u8],
    pub generation_config_bytes: Option<&'a [u8]>,
    pub root_chat_template_source: &'a str,
    pub included_template_bytes_by_name: &'a BTreeMap<String, Vec<u8>>,
}

/// Converts bounded, duplicate-aware artifact documents into one text descriptor.
#[derive(Debug)]
pub struct LagunaTextArtifactNormalizer;

impl LagunaTextArtifactNormalizer {
    /// Certifies text semantics before tokenizer construction or request rendering.
    pub fn normalize(
        target_contract: &LagunaTargetContract,
        sources: LagunaTextArtifactSources<'_>,
    ) -> Result<LagunaTextArtifactDescriptor, LagunaTextArtifactError> {
        let model_config = parse_json_document("model config", sources.model_config_bytes)?;
        let tokenizer_config =
            parse_json_document("tokenizer config", sources.tokenizer_config_bytes)?;
        // Strict parsing precedes tokenizers so duplicate-key replacement can never select meaning.
        let tokenizer_json = parse_json_document("tokenizer JSON", sources.tokenizer_bytes)?;
        let generation_config = sources
            .generation_config_bytes
            .map(|bytes| parse_json_document("generation config", bytes))
            .transpose()?
            .unwrap_or_else(|| Value::Object(Map::new()));

        let model_fields = model_language_fields(&model_config)?;
        let tokenizer_config_fields = object_fields(&tokenizer_config, "tokenizer config")?;
        let tokenizer_json_fields = object_fields(&tokenizer_json, "tokenizer JSON")?;
        let generation_fields = object_fields(&generation_config, "generation config")?;
        let model_vocabulary_size = required_u32(model_fields, "vocab_size", false)?;
        let maximum_context_tokens = required_u32(model_fields, "max_position_embeddings", false)?;
        if model_vocabulary_size != target_contract.model().vocabulary_size() {
            return Err(LagunaTextArtifactError::ModelContractMismatch {
                field_name: "vocab_size",
            });
        }
        if maximum_context_tokens != target_contract.model().maximum_position_count() {
            return Err(LagunaTextArtifactError::ModelContractMismatch {
                field_name: "max_position_embeddings",
            });
        }

        let bos_token_id = required_u32(model_fields, "bos_token_id", true)?;
        let pad_token_id = required_u32(model_fields, "pad_token_id", true)?;
        validate_optional_matching_id(generation_fields, "bos_token_id", bos_token_id)?;
        validate_optional_matching_id(generation_fields, "pad_token_id", pad_token_id)?;
        let mut end_token_ids = parse_token_id_set(model_fields, "eos_token_id")?;
        end_token_ids.extend(parse_optional_token_id_set(
            generation_fields,
            "eos_token_id",
        )?);
        if end_token_ids.is_empty() {
            return Err(LagunaTextArtifactError::InvalidField {
                field_name: "eos_token_id".to_owned(),
            });
        }

        let configured_tokens = parse_configured_tokens(tokenizer_config_fields)?;
        let tokenizer_added_tokens = parse_tokenizer_added_tokens(tokenizer_json_fields)?;
        validate_bidirectional_added_tokens(&configured_tokens, &tokenizer_added_tokens)?;
        let tokenizer = Tokenizer::from_bytes(sources.tokenizer_bytes)
            .map_err(|source| LagunaTextArtifactError::LoadTokenizer { source })?;
        validate_tokenizer_vocabulary(&tokenizer, model_vocabulary_size, &configured_tokens)?;
        validate_configured_control_ids(
            tokenizer_config_fields,
            &configured_tokens,
            bos_token_id,
            pad_token_id,
            &end_token_ids,
        )?;

        let resolved_template_sources = resolve_template_sources(
            sources.root_chat_template_source,
            sources.included_template_bytes_by_name,
        )?;
        let bos_token_content = configured_tokens
            .get(&bos_token_id)
            .cloned()
            .ok_or_else(|| LagunaTextArtifactError::SpecialTokenMismatch {
                configured_token_id: bos_token_id,
                token_content: "configured beginning token".to_owned(),
                tokenizer_token_id: None,
            })?;
        // Compilation and semantic probes happen once at startup; requests reuse this program.
        let template_program = LagunaTemplateProgram::compile(resolved_template_sources)?;
        let template_contract = derive_template_contract(&template_program, &bos_token_content)?;
        let reasoning_parser_id = required_parser_id(generation_fields, "reasoning_parser")?;
        let tool_call_parser_id = required_parser_id(generation_fields, "tool_call_parser")?;
        let generation_default_thinking_enabled = generation_fields
            .get("default_chat_template_kwargs")
            .and_then(Value::as_object)
            .and_then(|arguments| arguments.get("enable_thinking"))
            .map(|configured_default| {
                configured_default
                    .as_bool()
                    .ok_or_else(|| LagunaTextArtifactError::InvalidField {
                        field_name: "default_chat_template_kwargs.enable_thinking".to_owned(),
                    })
            })
            .transpose()?;
        let sampler_config = normalize_sampler_config(generation_fields)?;

        Ok(LagunaTextArtifactDescriptor::new(
            tokenizer,
            model_vocabulary_size,
            maximum_context_tokens,
            bos_token_id,
            pad_token_id,
            end_token_ids.into_iter().collect(),
            configured_tokens
                .into_iter()
                .map(|(token_id, token_content)| (token_content, token_id))
                .collect(),
            template_program,
            template_contract,
            generation_default_thinking_enabled,
            reasoning_parser_id,
            tool_call_parser_id,
            sampler_config,
        ))
    }
}
