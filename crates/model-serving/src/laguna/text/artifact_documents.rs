use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value};
use tokenizers::Tokenizer;

use crate::strict_json::{DUPLICATE_JSON_FIELD_MARKER, DuplicateAwareJsonValue};

use super::LagunaTextArtifactError;

const MAXIMUM_JSON_DOCUMENT_BYTES: usize = 32 * 1024 * 1024;
const MAXIMUM_ERROR_TEXT_CHARACTERS: usize = 128;

pub(super) fn bounded_artifact_text(unbounded_text: &str) -> String {
    unbounded_text
        .chars()
        .take(MAXIMUM_ERROR_TEXT_CHARACTERS)
        .collect()
}

/// Strictly parses one bounded semantic JSON document without duplicate replacement.
pub(super) fn parse_json_document(
    document_name: &'static str,
    document_bytes: &[u8],
) -> Result<Value, LagunaTextArtifactError> {
    if document_bytes.len() > MAXIMUM_JSON_DOCUMENT_BYTES {
        return Err(LagunaTextArtifactError::DocumentTooLarge {
            document_name,
            actual_bytes: document_bytes.len(),
            maximum_bytes: MAXIMUM_JSON_DOCUMENT_BYTES,
        });
    }
    serde_json::from_slice::<DuplicateAwareJsonValue>(document_bytes)
        .map(|strict_value| strict_value.0)
        .map_err(|source| {
            if source.to_string().contains(DUPLICATE_JSON_FIELD_MARKER) {
                LagunaTextArtifactError::DuplicateJsonField { document_name }
            } else {
                LagunaTextArtifactError::MalformedJson {
                    document_name,
                    source,
                }
            }
        })
}

pub(super) fn object_fields<'a>(
    document: &'a Value,
    document_name: &'static str,
) -> Result<&'a Map<String, Value>, LagunaTextArtifactError> {
    document
        .as_object()
        .ok_or(LagunaTextArtifactError::ExpectedJsonObject { document_name })
}

pub(super) fn model_language_fields(
    model_config: &Value,
) -> Result<&Map<String, Value>, LagunaTextArtifactError> {
    let root_fields = object_fields(model_config, "model config")?;
    match root_fields.get("text_config") {
        Some(Value::Object(text_fields)) => Ok(text_fields),
        Some(_) => Err(LagunaTextArtifactError::InvalidField {
            field_name: "text_config".to_owned(),
        }),
        None => Ok(root_fields),
    }
}

pub(super) fn required_u32(
    fields: &Map<String, Value>,
    field_name: &str,
    allows_zero: bool,
) -> Result<u32, LagunaTextArtifactError> {
    let unsigned_value = fields
        .get(field_name)
        .and_then(Value::as_u64)
        .ok_or_else(|| LagunaTextArtifactError::InvalidNumericField {
            field_name: field_name.to_owned(),
        })?;
    let converted_value = u32::try_from(unsigned_value).map_err(|_| {
        LagunaTextArtifactError::InvalidNumericField {
            field_name: field_name.to_owned(),
        }
    })?;
    if !allows_zero && converted_value == 0 {
        return Err(LagunaTextArtifactError::InvalidNumericField {
            field_name: field_name.to_owned(),
        });
    }
    Ok(converted_value)
}

pub(super) fn validate_optional_matching_id(
    fields: &Map<String, Value>,
    field_name: &'static str,
    expected_token_id: u32,
) -> Result<(), LagunaTextArtifactError> {
    if fields.get(field_name).is_some()
        && required_u32(fields, field_name, true)? != expected_token_id
    {
        return Err(LagunaTextArtifactError::ModelContractMismatch { field_name });
    }
    Ok(())
}

pub(super) fn parse_token_id_set(
    fields: &Map<String, Value>,
    field_name: &str,
) -> Result<BTreeSet<u32>, LagunaTextArtifactError> {
    let field_value =
        fields
            .get(field_name)
            .ok_or_else(|| LagunaTextArtifactError::InvalidField {
                field_name: field_name.to_owned(),
            })?;
    parse_token_ids(field_value, field_name)
}

pub(super) fn parse_optional_token_id_set(
    fields: &Map<String, Value>,
    field_name: &str,
) -> Result<BTreeSet<u32>, LagunaTextArtifactError> {
    fields
        .get(field_name)
        .map(|field_value| parse_token_ids(field_value, field_name))
        .transpose()
        .map(Option::unwrap_or_default)
}

fn parse_token_ids(
    field_value: &Value,
    field_name: &str,
) -> Result<BTreeSet<u32>, LagunaTextArtifactError> {
    let raw_ids = match field_value {
        Value::Array(raw_ids) => raw_ids.as_slice(),
        Value::Number(_) => std::slice::from_ref(field_value),
        _ => {
            return Err(LagunaTextArtifactError::InvalidField {
                field_name: field_name.to_owned(),
            });
        }
    };
    let mut token_ids = BTreeSet::new();
    for raw_id in raw_ids {
        let unsigned_id =
            raw_id
                .as_u64()
                .ok_or_else(|| LagunaTextArtifactError::InvalidNumericField {
                    field_name: field_name.to_owned(),
                })?;
        let token_id = u32::try_from(unsigned_id).map_err(|_| {
            LagunaTextArtifactError::InvalidNumericField {
                field_name: field_name.to_owned(),
            }
        })?;
        token_ids.insert(token_id);
    }
    Ok(token_ids)
}

pub(super) fn parse_configured_tokens(
    tokenizer_config_fields: &Map<String, Value>,
) -> Result<BTreeMap<u32, String>, LagunaTextArtifactError> {
    let decoder_fields = tokenizer_config_fields
        .get("added_tokens_decoder")
        .and_then(Value::as_object)
        .ok_or_else(|| LagunaTextArtifactError::InvalidField {
            field_name: "added_tokens_decoder".to_owned(),
        })?;
    let mut configured_tokens = BTreeMap::new();
    for (token_id_text, token_descriptor) in decoder_fields {
        let token_id =
            token_id_text
                .parse::<u32>()
                .map_err(|_| LagunaTextArtifactError::InvalidField {
                    field_name: "added_tokens_decoder token ID".to_owned(),
                })?;
        let token_content = token_descriptor
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| LagunaTextArtifactError::InvalidField {
                field_name: format!("added_tokens_decoder.{token_id}.content"),
            })?;
        if configured_tokens
            .insert(token_id, token_content.to_owned())
            .is_some()
        {
            return Err(LagunaTextArtifactError::DuplicateTokenIdentity { token_id });
        }
    }
    Ok(configured_tokens)
}

pub(super) fn parse_tokenizer_added_tokens(
    tokenizer_json_fields: &Map<String, Value>,
) -> Result<BTreeMap<u32, String>, LagunaTextArtifactError> {
    let added_tokens = tokenizer_json_fields
        .get("added_tokens")
        .and_then(Value::as_array)
        .ok_or_else(|| LagunaTextArtifactError::InvalidField {
            field_name: "tokenizer.added_tokens".to_owned(),
        })?;
    let mut tokenizer_tokens = BTreeMap::new();
    for token_descriptor in added_tokens {
        let token_id = token_descriptor
            .get("id")
            .and_then(Value::as_u64)
            .and_then(|raw_id| u32::try_from(raw_id).ok())
            .ok_or_else(|| LagunaTextArtifactError::InvalidField {
                field_name: "tokenizer.added_tokens.id".to_owned(),
            })?;
        let token_content = token_descriptor
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| LagunaTextArtifactError::InvalidField {
                field_name: "tokenizer.added_tokens.content".to_owned(),
            })?;
        if tokenizer_tokens
            .insert(token_id, token_content.to_owned())
            .is_some()
        {
            return Err(LagunaTextArtifactError::DuplicateTokenIdentity { token_id });
        }
    }
    Ok(tokenizer_tokens)
}

pub(super) fn validate_bidirectional_added_tokens(
    configured_tokens: &BTreeMap<u32, String>,
    tokenizer_tokens: &BTreeMap<u32, String>,
) -> Result<(), LagunaTextArtifactError> {
    for (token_id, token_content) in configured_tokens {
        validate_identity(tokenizer_tokens, *token_id, token_content)?;
    }
    for (token_id, token_content) in tokenizer_tokens {
        validate_identity(configured_tokens, *token_id, token_content)?;
    }
    Ok(())
}

fn validate_identity(
    tokens: &BTreeMap<u32, String>,
    configured_token_id: u32,
    token_content: &str,
) -> Result<(), LagunaTextArtifactError> {
    if tokens.get(&configured_token_id).map(String::as_str) != Some(token_content) {
        return Err(LagunaTextArtifactError::SpecialTokenMismatch {
            configured_token_id,
            token_content: bounded_artifact_text(token_content),
            tokenizer_token_id: tokens
                .iter()
                .find_map(|(token_id, content)| (content == token_content).then_some(*token_id)),
        });
    }
    Ok(())
}

pub(super) fn validate_tokenizer_vocabulary(
    tokenizer: &Tokenizer,
    model_vocabulary_size: u32,
    configured_tokens: &BTreeMap<u32, String>,
) -> Result<(), LagunaTextArtifactError> {
    let tokenizer_vocabulary_size =
        u32::try_from(tokenizer.get_vocab_size(true)).map_err(|_| {
            LagunaTextArtifactError::InvalidNumericField {
                field_name: "tokenizer vocabulary size".to_owned(),
            }
        })?;
    if tokenizer_vocabulary_size != model_vocabulary_size {
        return Err(LagunaTextArtifactError::ModelContractMismatch {
            field_name: "vocab_size",
        });
    }
    let mut observed_token_ids = BTreeSet::new();
    for tokenizer_token_id in tokenizer.get_vocab(true).into_values() {
        if tokenizer_token_id >= model_vocabulary_size
            || !observed_token_ids.insert(tokenizer_token_id)
        {
            return Err(LagunaTextArtifactError::ModelContractMismatch {
                field_name: "vocab_size",
            });
        }
    }
    for (configured_token_id, token_content) in configured_tokens {
        let tokenizer_token_id = tokenizer.token_to_id(token_content);
        if tokenizer_token_id != Some(*configured_token_id)
            || tokenizer.id_to_token(*configured_token_id).as_deref() != Some(token_content)
        {
            return Err(LagunaTextArtifactError::SpecialTokenMismatch {
                configured_token_id: *configured_token_id,
                token_content: bounded_artifact_text(token_content),
                tokenizer_token_id,
            });
        }
    }
    Ok(())
}

pub(super) fn validate_configured_control_ids(
    tokenizer_config_fields: &Map<String, Value>,
    configured_tokens: &BTreeMap<u32, String>,
    bos_token_id: u32,
    pad_token_id: u32,
    end_token_ids: &BTreeSet<u32>,
) -> Result<(), LagunaTextArtifactError> {
    for (field_name, token_id) in [("bos_token", bos_token_id), ("pad_token", pad_token_id)] {
        let token_content = configured_token_content(tokenizer_config_fields, field_name)?;
        validate_identity(configured_tokens, token_id, token_content)?;
    }
    for end_token_id in end_token_ids {
        if !configured_tokens.contains_key(end_token_id) {
            return Err(LagunaTextArtifactError::SpecialTokenMismatch {
                configured_token_id: *end_token_id,
                token_content: "configured end token".to_owned(),
                tokenizer_token_id: None,
            });
        }
    }
    Ok(())
}

fn configured_token_content<'a>(
    tokenizer_config_fields: &'a Map<String, Value>,
    field_name: &str,
) -> Result<&'a str, LagunaTextArtifactError> {
    let token_value = tokenizer_config_fields.get(field_name).ok_or_else(|| {
        LagunaTextArtifactError::InvalidField {
            field_name: field_name.to_owned(),
        }
    })?;
    token_value
        .as_str()
        .or_else(|| token_value.get("content").and_then(Value::as_str))
        .ok_or_else(|| LagunaTextArtifactError::InvalidField {
            field_name: field_name.to_owned(),
        })
}
