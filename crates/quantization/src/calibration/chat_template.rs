//! Renders calibration chat samples through the model's own chat template.
//!
//! OptiQ calibrates on the token sequences the model will actually see: a
//! chat sample is rendered with `tokenizer.apply_chat_template` before it is
//! encoded, so the sweep's activations include the role markers, the tool
//! preamble and the reasoning tags. Rendering the same sample as plain
//! role-prefixed text gives different tokens at different offsets, and a
//! sweep on those measures a different calibration set from the reference.
//!
//! The environment mirrors the one transformers builds for a chat template
//! (`transformers/utils/chat_template_utils.py`): block trimming on, loop
//! controls, a `raise_exception` global, a `tojson` filter backed by
//! `json.dumps(ensure_ascii=False)`, and the Python string and mapping
//! methods templates call (`startswith`, `split`, `items`, ...).

use std::path::Path;

use minijinja::value::{Kwargs, Value, ValueKind};
use minijinja::{Environment, Error, ErrorKind, State};

use super::python_json::to_python_json;
use super::sample::CalibrationMessage;

#[cfg(test)]
#[path = "chat_template_tests.rs"]
mod chat_template_tests;

/// The name the one template is registered under.
const CHAT_TEMPLATE_NAME: &str = "chat_template";

/// The standalone template file newer releases ship beside the tokenizer.
const CHAT_TEMPLATE_FILE_NAME: &str = "chat_template.jinja";

/// The tokenizer settings file that carries the template in older releases
/// and the special tokens every template may reference.
const TOKENIZER_CONFIG_FILE_NAME: &str = "tokenizer_config.json";

/// The special tokens transformers passes to every template.
const SPECIAL_TOKEN_NAMES: [&str; 4] = ["bos_token", "eos_token", "pad_token", "unk_token"];

/// A model's chat template with the special tokens it may reference.
pub struct ChatTemplate {
    environment: Environment<'static>,
    special_tokens: Vec<(String, String)>,
}

impl ChatTemplate {
    /// Loads the template from a model directory: `chat_template.jinja`
    /// first, as transformers does, then the `chat_template` string of
    /// `tokenizer_config.json`. `None` when the model ships no template or
    /// the template does not parse, in which case calibration falls back to
    /// role-prefixed text, OptiQ's own fallback.
    #[must_use]
    pub fn from_model_directory(model_directory: &Path) -> Option<Self> {
        let tokenizer_config: serde_json::Value =
            match std::fs::read_to_string(model_directory.join(TOKENIZER_CONFIG_FILE_NAME)) {
                Ok(config_text) => serde_json::from_str(&config_text).unwrap_or_default(),
                Err(_) => serde_json::Value::Null,
            };
        let template_source =
            match std::fs::read_to_string(model_directory.join(CHAT_TEMPLATE_FILE_NAME)) {
                Ok(template_source) => template_source,
                Err(_) => tokenizer_config["chat_template"].as_str()?.to_owned(),
            };
        let mut special_tokens = Vec::new();
        for special_token_name in SPECIAL_TOKEN_NAMES {
            let token_entry = &tokenizer_config[special_token_name];
            let token_text = match token_entry.as_str() {
                Some(token_text) => Some(token_text),
                None => token_entry["content"].as_str(),
            };
            if let Some(token_text) = token_text {
                special_tokens.push((special_token_name.to_owned(), token_text.to_owned()));
            }
        }
        Self::from_source(template_source, special_tokens)
    }

    /// Builds the template from its source text; `None` when it does not
    /// parse.
    #[must_use]
    pub fn from_source(
        template_source: String,
        special_tokens: Vec<(String, String)>,
    ) -> Option<Self> {
        let mut environment = Environment::new();
        environment.set_trim_blocks(true);
        environment.set_lstrip_blocks(true);
        environment.add_function("raise_exception", raise_exception);
        environment.add_filter("tojson", python_tojson);
        environment.set_unknown_method_callback(python_method);
        match environment.add_template_owned(CHAT_TEMPLATE_NAME, template_source) {
            Ok(()) => Some(Self {
                environment,
                special_tokens,
            }),
            Err(parse_error) => {
                tracing::warn!(%parse_error, "the chat template does not parse; calibration falls back to role-prefixed text");
                None
            }
        }
    }

    /// The conversation as the model's prompt text, with no generation
    /// prompt appended, or the template's own error text.
    pub fn render(
        &self,
        messages: &[CalibrationMessage],
        tool_definitions: Option<&Value>,
    ) -> Result<String, String> {
        let mut context_entries: Vec<(String, Value)> = Vec::new();
        context_entries.push(("messages".to_owned(), Value::from_serialize(messages)));
        context_entries.push(("add_generation_prompt".to_owned(), Value::from(false)));
        if let Some(tool_definitions) = tool_definitions {
            context_entries.push(("tools".to_owned(), tool_definitions.clone()));
        }
        for (special_token_name, special_token_text) in &self.special_tokens {
            let token_value = Value::from(special_token_text.as_str());
            context_entries.push((special_token_name.clone(), token_value));
        }
        let template = match self.environment.get_template(CHAT_TEMPLATE_NAME) {
            Ok(template) => template,
            Err(lookup_error) => return Err(lookup_error.to_string()),
        };
        match template.render(Value::from_iter(context_entries)) {
            Ok(rendered_text) => Ok(rendered_text),
            Err(render_error) => Err(render_error.to_string()),
        }
    }
}

/// The template's way of refusing a conversation it cannot render.
fn raise_exception(message: String) -> Result<Value, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, message))
}

/// `tojson` as transformers defines it; `indent` and the other keyword
/// arguments are accepted and ignored, which no shipped template relies on.
fn python_tojson(value: Value, _keyword_arguments: Kwargs) -> String {
    to_python_json(&value)
}

/// The Python methods chat templates call on strings and mappings, which
/// Jinja inherits from Python and this engine does not define.
fn python_method(
    _state: &State,
    receiver: &Value,
    method_name: &str,
    arguments: &[Value],
) -> Result<Value, Error> {
    if let Some(text) = receiver.as_str() {
        return python_string_method(text, method_name, arguments);
    }
    if receiver.kind() == ValueKind::Map {
        return python_mapping_method(receiver, method_name, arguments);
    }
    Err(unknown_method(method_name))
}

fn python_string_method(
    text: &str,
    method_name: &str,
    arguments: &[Value],
) -> Result<Value, Error> {
    let first_argument = arguments.first().and_then(Value::as_str);
    let stripped_characters: Vec<char> = first_argument.unwrap_or_default().chars().collect();
    match (method_name, first_argument) {
        ("startswith", Some(prefix)) => Ok(Value::from(text.starts_with(prefix))),
        ("endswith", Some(suffix)) => Ok(Value::from(text.ends_with(suffix))),
        ("split", Some(separator)) => Ok(owned_pieces(text.split(separator))),
        ("split", None) => Ok(owned_pieces(text.split_whitespace())),
        ("strip", None) => Ok(Value::from(text.trim())),
        ("lstrip", None) => Ok(Value::from(text.trim_start())),
        ("rstrip", None) => Ok(Value::from(text.trim_end())),
        ("strip", Some(_)) => Ok(Value::from(
            text.trim_matches(stripped_characters.as_slice()),
        )),
        ("lstrip", Some(_)) => Ok(Value::from(
            text.trim_start_matches(stripped_characters.as_slice()),
        )),
        ("rstrip", Some(_)) => Ok(Value::from(
            text.trim_end_matches(stripped_characters.as_slice()),
        )),
        ("upper", _) => Ok(Value::from(text.to_uppercase())),
        ("lower", _) => Ok(Value::from(text.to_lowercase())),
        ("replace", Some(replaced)) => {
            let replacement = arguments.get(1).and_then(Value::as_str).unwrap_or_default();
            Ok(Value::from(text.replace(replaced, replacement)))
        }
        _ => Err(unknown_method(method_name)),
    }
}

/// The pieces of a split as a template list of owned strings.
fn owned_pieces<'text>(pieces: impl Iterator<Item = &'text str>) -> Value {
    let mut owned_pieces = Vec::new();
    for piece in pieces {
        owned_pieces.push(Value::from(piece));
    }
    Value::from(owned_pieces)
}

fn python_mapping_method(
    mapping: &Value,
    method_name: &str,
    arguments: &[Value],
) -> Result<Value, Error> {
    match method_name {
        "keys" => Ok(Value::from(mapping.try_iter()?.collect::<Vec<Value>>())),
        "items" => {
            let mut key_value_pairs = Vec::new();
            for key in mapping.try_iter()? {
                let entry_value = mapping.get_item(&key)?;
                key_value_pairs.push(Value::from(vec![key, entry_value]));
            }
            Ok(Value::from(key_value_pairs))
        }
        "get" => {
            let Some(key) = arguments.first() else {
                return Err(unknown_method(method_name));
            };
            let entry_value = mapping.get_item(key)?;
            if entry_value.is_undefined() {
                return Ok(arguments.get(1).cloned().unwrap_or(Value::from(())));
            }
            Ok(entry_value)
        }
        _ => Err(unknown_method(method_name)),
    }
}

fn unknown_method(method_name: &str) -> Error {
    Error::new(
        ErrorKind::UnknownMethod,
        format!("the chat template calls an unsupported method `{method_name}`"),
    )
}
