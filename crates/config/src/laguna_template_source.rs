//! Shared Laguna root-template selection keeps public discovery and worker startup aligned.

use std::fmt;

use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::Value;
use thiserror::Error;

const CHAT_TEMPLATE_FIELD_NAME: &str = "chat_template";

/// Bounded filesystem evidence supplied by discovery or authoritative validation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LagunaStandaloneChatTemplateState {
    Missing,
    Empty,
    NonEmpty,
}

/// The one root source selected before include resolution or template compilation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LagunaRootChatTemplateSource {
    Embedded {
        template_source: String,
        standalone_template_requires_include: bool,
    },
    Standalone,
}

/// A deterministic root-source selection failure shared by discovery and model startup.
#[derive(Debug, Error)]
pub enum LagunaRootChatTemplateSelectionError {
    #[error("Laguna tokenizer configuration is not valid JSON")]
    MalformedTokenizerConfig(#[source] serde_json::Error),
    #[error("Laguna tokenizer configuration root must be a JSON object")]
    TokenizerConfigMustBeObject,
    #[error("Laguna tokenizer configuration contains duplicate chat_template fields")]
    DuplicateChatTemplateField,
    #[error("Laguna embedded chat_template must not be empty")]
    EmptyEmbeddedChatTemplate,
    #[error("Laguna embedded chat_template must be a string or null")]
    UnsupportedEmbeddedChatTemplateType,
    #[error("Laguna artifact does not provide a root chat template")]
    MissingRootChatTemplate,
    #[error("Laguna standalone chat_template.jinja must not be empty")]
    EmptyStandaloneChatTemplate,
    #[error("Laguna artifact provides conflicting embedded and standalone chat templates")]
    ConflictingRootChatTemplates,
}

/// Selects one authority without opening files so every caller can preserve its own I/O policy.
pub fn select_laguna_root_chat_template(
    tokenizer_config_bytes: &[u8],
    standalone_template_state: LagunaStandaloneChatTemplateState,
) -> Result<LagunaRootChatTemplateSource, LagunaRootChatTemplateSelectionError> {
    let tokenizer_config: Value = serde_json::from_slice(tokenizer_config_bytes)
        .map_err(LagunaRootChatTemplateSelectionError::MalformedTokenizerConfig)?;
    if !tokenizer_config.is_object() {
        return Err(LagunaRootChatTemplateSelectionError::TokenizerConfigMustBeObject);
    }

    let template_projection = parse_template_projection(tokenizer_config_bytes)?;
    if template_projection.chat_template_occurrence_count > 1 {
        return Err(LagunaRootChatTemplateSelectionError::DuplicateChatTemplateField);
    }

    match template_projection.chat_template {
        Some(Value::String(template_source)) if template_source.is_empty() => {
            Err(LagunaRootChatTemplateSelectionError::EmptyEmbeddedChatTemplate)
        }
        Some(Value::String(template_source)) => Ok(LagunaRootChatTemplateSource::Embedded {
            template_source,
            // A physical chat_template.jinja may be an explicitly selected include.
            // Callers resolve the graph before deciding that it is a second root authority.
            standalone_template_requires_include: matches!(
                standalone_template_state,
                LagunaStandaloneChatTemplateState::NonEmpty
            ),
        }),
        Some(Value::Null) | None => match standalone_template_state {
            LagunaStandaloneChatTemplateState::NonEmpty => {
                Ok(LagunaRootChatTemplateSource::Standalone)
            }
            LagunaStandaloneChatTemplateState::Empty => {
                Err(LagunaRootChatTemplateSelectionError::EmptyStandaloneChatTemplate)
            }
            LagunaStandaloneChatTemplateState::Missing => {
                Err(LagunaRootChatTemplateSelectionError::MissingRootChatTemplate)
            }
        },
        Some(_) => Err(LagunaRootChatTemplateSelectionError::UnsupportedEmbeddedChatTemplateType),
    }
}

/// Rejects a nonempty standalone file unless the embedded root selected it as an include.
pub fn validate_laguna_standalone_chat_template_role(
    root_template_source: &LagunaRootChatTemplateSource,
    standalone_template_is_selected_include: bool,
) -> Result<(), LagunaRootChatTemplateSelectionError> {
    if matches!(
        root_template_source,
        LagunaRootChatTemplateSource::Embedded {
            standalone_template_requires_include: true,
            ..
        }
    ) && !standalone_template_is_selected_include
    {
        return Err(LagunaRootChatTemplateSelectionError::ConflictingRootChatTemplates);
    }
    Ok(())
}

fn parse_template_projection(
    tokenizer_config_bytes: &[u8],
) -> Result<TokenizerTemplateProjection, LagunaRootChatTemplateSelectionError> {
    let mut deserializer = serde_json::Deserializer::from_slice(tokenizer_config_bytes);
    TokenizerTemplateProjection::deserialize(&mut deserializer)
        .map_err(LagunaRootChatTemplateSelectionError::MalformedTokenizerConfig)
}

struct TokenizerTemplateProjection {
    chat_template: Option<Value>,
    chat_template_occurrence_count: usize,
}

impl<'de> Deserialize<'de> for TokenizerTemplateProjection {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(TokenizerTemplateProjectionVisitor)
    }
}

struct TokenizerTemplateProjectionVisitor;

impl<'de> Visitor<'de> for TokenizerTemplateProjectionVisitor {
    type Value = TokenizerTemplateProjection;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a Laguna tokenizer configuration object")
    }

    fn visit_map<M>(self, mut fields: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut chat_template = None;
        let mut chat_template_occurrence_count = 0usize;
        while let Some(field_name) = fields.next_key::<String>()? {
            let field_value = fields.next_value::<Value>()?;
            if field_name == CHAT_TEMPLATE_FIELD_NAME {
                chat_template_occurrence_count = chat_template_occurrence_count.saturating_add(1);
                if chat_template.is_none() {
                    chat_template = Some(field_value);
                }
            }
        }
        Ok(TokenizerTemplateProjection {
            chat_template,
            chat_template_occurrence_count,
        })
    }
}
