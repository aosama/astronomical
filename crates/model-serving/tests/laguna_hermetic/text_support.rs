use std::collections::BTreeMap;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, ChatToolDefinition,
    RequestId,
};
use astronomical_model_serving::{
    LagunaTargetNormalizer, LagunaTextArtifactDescriptor, LagunaTextArtifactNormalizer,
    LagunaTextArtifactSources,
};
use serde_json::{Map, Value, json};

use super::support::{config_bytes, config_value};

pub(super) const SYNTHETIC_LAGUNA_MODEL_ID: &str = "synthetic-laguna";
pub(super) const MODEL_EOS_TOKEN_ID: u32 = 2;
pub(super) const ASSISTANT_END_TOKEN_ID: u32 = 31;
pub(super) const GENERATION_ONLY_EOS_TOKEN_ID: u32 = 47;
pub(super) const SYNTHETIC_VOCABULARY_SIZE: u32 = 128;
pub(super) const LARGE_MODEL_CONTEXT_TOKEN_COUNT: u32 = 32_768;
pub(super) const DEFAULT_SYSTEM_MESSAGE: &str =
    "Use the supplied play as the only source for literary analysis.";
pub(super) const ROMEO_AND_JULIET_SOURCE: &str = include_str!(
    "../../../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);

/// A compact Poolside template keeps tests readable while exercising the real supported semantics.
pub(super) const POOLSIDE_TEMPLATE: &str = r#"
{{- bos_token -}}
{%- set enable_thinking = enable_thinking | default(false) -%}
{%- set preserve_thinking = preserve_thinking | default(false) -%}
{%- set add_generation_prompt = add_generation_prompt | default(false) -%}
{%- set system_message = "Use the supplied play as the only source for literary analysis." -%}
{%- if messages and messages[0].role == "system" -%}
  {%- set system_message = messages[0].content -%}
  {%- set messages = messages[1:] -%}
{%- endif -%}
{%- if (system_message and system_message.strip()) or tools -%}
  {{- "<system>" -}}
  {{- system_message.rstrip() if system_message and system_message.strip() else "" -}}
  {%- if tools -%}
    {{- "\n\n<available_tools>\n" -}}
    {%- for tool in tools -%}
      {{- (tool | tojson) + "\n" -}}
    {%- endfor -%}
    {{- "</available_tools>" -}}
  {%- endif -%}
  {{- "</system>\n" -}}
{%- endif -%}
{%- for message in messages -%}
  {%- if message.role == "user" -%}
    {{- "<user>" + message.content + "</user>\n" -}}
  {%- elif message.role == "assistant" -%}
    {{- "<assistant>" -}}
    {%- if enable_thinking or preserve_thinking -%}
      {{- "<think>" + message.reasoning_content + "</think>" -}}
    {%- else -%}
      {{- "</think>" -}}
    {%- endif -%}
    {{- message.content if message.content else "" -}}
    {%- for tool_call in message.tool_calls -%}
      {{- "<tool_call>" + tool_call.function.name -}}
      {%- for argument_name, argument_value in tool_call.function.arguments.items() -%}
        {{- "<arg_key>" + argument_name + "</arg_key>" -}}
        {{- "<arg_value>" + argument_value + "</arg_value>" -}}
      {%- endfor -%}
      {{- "</tool_call>" -}}
    {%- endfor -%}
    {{- "</assistant>\n" -}}
  {%- elif message.role == "tool" -%}
    {{- "<tool_response>" + message.content + "</tool_response>\n" -}}
  {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
  {{- "<assistant>" -}}
  {{- "<think>" if enable_thinking else "</think>" -}}
{%- endif -%}
"#;

/// All text-side artifact inputs are mutable so one test can change one contract at a time.
pub(super) struct SyntheticLagunaTextArtifact {
    pub(super) model_config: Value,
    pub(super) tokenizer: Value,
    pub(super) tokenizer_config: Value,
    pub(super) generation_config: Option<Value>,
    pub(super) root_chat_template_source: String,
    pub(super) included_templates: BTreeMap<String, Vec<u8>>,
}

impl SyntheticLagunaTextArtifact {
    /// Builds the XS-like inline artifact: no top-k or repetition-penalty policy is implied.
    pub(super) fn extra_small_inline() -> Self {
        Self {
            model_config: model_config_value(LARGE_MODEL_CONTEXT_TOKEN_COUNT),
            tokenizer: tokenizer_value(),
            tokenizer_config: tokenizer_config_value(POOLSIDE_TEMPLATE),
            generation_config: Some(generation_config_value(false)),
            root_chat_template_source: POOLSIDE_TEMPLATE.to_owned(),
            included_templates: BTreeMap::new(),
        }
    }

    /// Builds the S-like artifact whose tokenizer selects one artifact-local included template.
    pub(super) fn small_included() -> Self {
        let mut artifact = Self::extra_small_inline();
        artifact.set_embedded_chat_template("{% include 'chat_template.jinja' %}");
        artifact.included_templates.insert(
            "chat_template.jinja".to_owned(),
            template_with_defaults(true, true).into_bytes(),
        );
        artifact.generation_config = Some(generation_config_value(true));
        artifact
    }

    pub(super) fn set_embedded_chat_template(&mut self, template_source: impl Into<String>) {
        let template_source = template_source.into();
        self.tokenizer_config["chat_template"] = json!(template_source.clone());
        self.root_chat_template_source = template_source;
    }

    /// Normalizes bytes only through the proposed Laguna owner after canonical model validation.
    pub(super) fn normalize(&self) -> LagunaTextArtifactDescriptor {
        self.try_normalize()
            .expect("the synthetic Laguna text artifact should normalize")
    }

    pub(super) fn try_normalize(
        &self,
    ) -> Result<LagunaTextArtifactDescriptor, astronomical_model_serving::LagunaTextArtifactError>
    {
        let model_config_bytes = config_bytes(&self.model_config);
        let tokenizer_bytes =
            serde_json::to_vec(&self.tokenizer).expect("the synthetic tokenizer should serialize");
        let tokenizer_config_bytes = serde_json::to_vec(&self.tokenizer_config)
            .expect("the synthetic tokenizer config should serialize");
        let generation_config_bytes = self.generation_config.as_ref().map(|generation_config| {
            serde_json::to_vec(generation_config)
                .expect("the synthetic generation config should serialize")
        });
        let target_contract = LagunaTargetNormalizer::normalize(&model_config_bytes)
            .expect("model-side Laguna normalization must precede text normalization");

        LagunaTextArtifactNormalizer::normalize(
            &target_contract,
            LagunaTextArtifactSources {
                model_config_bytes: &model_config_bytes,
                tokenizer_bytes: &tokenizer_bytes,
                tokenizer_config_bytes: &tokenizer_config_bytes,
                generation_config_bytes: generation_config_bytes.as_deref(),
                root_chat_template_source: &self.root_chat_template_source,
                included_template_bytes_by_name: &self.included_templates,
            },
        )
    }
}

/// Produces a request whose human input always comes from the repository literary fixture.
pub(super) fn romeo_and_juliet_command(
    request_number: u64,
    thinking_budget: Option<u16>,
) -> ChatGenerationCommand {
    ChatGenerationCommand {
        request_id: RequestId::new(request_number),
        model: SYNTHETIC_LAGUNA_MODEL_ID.to_owned(),
        messages: vec![ChatMessage::User {
            content: ROMEO_AND_JULIET_SOURCE.to_owned(),
            images: Vec::new(),
        }],
        tools: declared_literary_tools(),
        tool_choice: ChatToolChoice::Auto,
        settings: ChatGenerationSettings {
            max_output_tokens: 512,
            temperature_thousandths: None,
            top_p_thousandths: None,
            seed: Some(98),
            thinking_budget,
        },
    }
}

/// Declares two tools so the parser journey can prove ordered, repeated tool-call streaming.
pub(super) fn declared_literary_tools() -> Vec<ChatToolDefinition> {
    vec![
        ChatToolDefinition {
            name: "find_character".to_owned(),
            description: Some("Find one character in the supplied play.".to_owned()),
            parameters_json:
                r#"{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}"#
                    .to_owned(),
        },
        ChatToolDefinition {
            name: "summarize_scene".to_owned(),
            description: Some("Summarize one scene from the supplied play.".to_owned()),
            parameters_json:
                r#"{"type":"object","properties":{"scene":{"type":"string"}},"required":["scene"]}"#
                    .to_owned(),
        },
    ]
}

pub(super) fn template_with_defaults(
    enables_thinking_by_default: bool,
    preserves_prior_reasoning: bool,
) -> String {
    POOLSIDE_TEMPLATE
        .replace(
            "enable_thinking | default(false)",
            &format!("enable_thinking | default({enables_thinking_by_default})"),
        )
        .replace(
            "preserve_thinking | default(false)",
            &format!("preserve_thinking | default({preserves_prior_reasoning})"),
        )
}

fn model_config_value(maximum_position_count: u32) -> Value {
    let mut model_config = config_value(1);
    model_config["vocab_size"] = json!(SYNTHETIC_VOCABULARY_SIZE);
    model_config["hidden_size"] = json!(16);
    model_config["intermediate_size"] = json!(32);
    model_config["num_attention_heads"] = json!(2);
    model_config["num_key_value_heads"] = json!(1);
    model_config["head_dim"] = json!(8);
    model_config["max_position_embeddings"] = json!(maximum_position_count);
    model_config["eos_token_id"] = json!([MODEL_EOS_TOKEN_ID, ASSISTANT_END_TOKEN_ID]);
    model_config["bos_token_id"] = json!(MODEL_EOS_TOKEN_ID);
    model_config["pad_token_id"] = json!(9);
    model_config
}

fn generation_config_value(has_small_model_sampling_policy: bool) -> Value {
    let mut generation_config = json!({
        "do_sample": true,
        "eos_token_id": [ASSISTANT_END_TOKEN_ID, GENERATION_ONLY_EOS_TOKEN_ID],
        "temperature": 1.0,
        "top_p": 1.0,
        "tool_call_parser": "poolside_v1",
        "reasoning_parser": "poolside_v1",
        "default_chat_template_kwargs": {"enable_thinking": true}
    });
    if has_small_model_sampling_policy {
        generation_config["top_k"] = json!(20);
        generation_config["repetition_penalty"] = json!(1.05);
    }
    generation_config
}

fn tokenizer_config_value(chat_template: &str) -> Value {
    let mut added_tokens_decoder = Map::new();
    for (token_id, token_text, is_special) in control_tokens() {
        added_tokens_decoder.insert(
            token_id.to_string(),
            json!({
                "content": token_text,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": is_special
            }),
        );
    }
    json!({
        "added_tokens_decoder": added_tokens_decoder,
        "bos_token": eos_token_text(),
        "eos_token": eos_token_text(),
        "pad_token": pad_token_text(),
        "model_max_length": 1_000_000_000_000_u64,
        "chat_template": chat_template
    })
}

fn tokenizer_value() -> Value {
    let control_tokens = control_tokens();
    let mut vocabulary = Map::new();
    for token_id in 0..SYNTHETIC_VOCABULARY_SIZE {
        if !control_tokens
            .iter()
            .any(|(control_token_id, _, _)| *control_token_id == token_id)
        {
            vocabulary.insert(format!("token_{token_id}"), json!(token_id));
        }
    }
    for (token_id, token_text, _) in &control_tokens {
        vocabulary.insert((*token_text).to_owned(), json!(token_id));
    }
    let added_tokens = control_tokens
        .iter()
        .map(|(token_id, token_text, is_special)| {
            json!({
                "id": token_id,
                "content": token_text,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": is_special
            })
        })
        .collect::<Vec<_>>();
    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": null,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": vocabulary,
            "unk_token": "token_0"
        }
    })
}

fn control_tokens() -> Vec<(u32, &'static str, bool)> {
    vec![
        (MODEL_EOS_TOKEN_ID, eos_token_text(), true),
        (9, pad_token_text(), true),
        (18, "<think>", false),
        (19, "</think>", false),
        (23, "<assistant>", false),
        (ASSISTANT_END_TOKEN_ID, "</assistant>", true),
        (25, "<tool_call>", false),
        (26, "</tool_call>", false),
        (GENERATION_ONLY_EOS_TOKEN_ID, "<generation_end>", true),
    ]
}

const fn eos_token_text() -> &'static str {
    "\u{3008}|EOS|\u{3009}"
}

const fn pad_token_text() -> &'static str {
    "\u{3008}|PAD|\u{3009}"
}
