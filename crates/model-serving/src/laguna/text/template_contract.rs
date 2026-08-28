use astronomical_ipc_protocol::{
    ChatAssistantToolCall, ChatAssistantToolFunction, ChatMessage, ChatToolDefinition,
};
use serde_json::{Value, json};

use super::LagunaTextArtifactError;
use super::artifact_descriptor::LagunaTemplateContract;
use super::template_context::LagunaTemplateContext;
use super::template_program::{LagunaTemplateProgram, LagunaTemplateProgramError};

const PROBE_USER_CONTENT: &str = "__laguna_probe_user__";
const PROBE_SYSTEM_CONTENT: &str = "__laguna_probe_system__";
const PROBE_ASSISTANT_CONTENT: &str = "__laguna_probe_assistant__";
const PROBE_REASONING_CONTENT: &str = "__laguna_probe_reasoning__";
const PROBE_TOOL_RESPONSE: &str = "__laguna_probe_tool_response__";
const PROBE_TOOL_NAME: &str = "laguna_probe_tool";

/// Derives canonical facts only from bounded executions of the compiled artifact template.
pub(super) fn derive_template_contract(
    template_program: &LagunaTemplateProgram,
    bos_token_content: &str,
) -> Result<LagunaTemplateContract, LagunaTextArtifactError> {
    let default_system_message =
        derive_default_system_message(template_program, bos_token_content)?;
    validate_explicit_and_empty_system_behavior(template_program, bos_token_content)?;
    validate_tool_definition_protocol(template_program, bos_token_content)?;
    let preserves_prior_reasoning = derive_history_protocol(template_program, bos_token_content)?;
    let default_thinking_enabled =
        derive_and_validate_generation_prefixes(template_program, bos_token_content)?;
    Ok(LagunaTemplateContract {
        bos_token_content: bos_token_content.to_owned(),
        default_system_message,
        default_thinking_enabled,
        preserves_prior_reasoning,
    })
}

fn derive_default_system_message(
    template_program: &LagunaTemplateProgram,
    bos_token_content: &str,
) -> Result<String, LagunaTextArtifactError> {
    let rendered_prompt = render_probe(
        template_program,
        vec![user_message()],
        Vec::new(),
        Some(false),
        false,
        bos_token_content,
    )?;
    let after_bos = strip_required_prefix(
        &rendered_prompt,
        bos_token_content,
        "beginning token or user-message protocol does not match Poolside",
    )?;
    let (system_inner, after_system) = take_optional_tagged_block(after_bos, "system");
    let (user_inner, after_user) = take_required_tagged_block(
        after_system,
        "user",
        "beginning token or user-message protocol does not match Poolside",
    )?;
    if user_inner != PROBE_USER_CONTENT || !after_user.trim().is_empty() {
        return Err(unsupported_poolside(
            "beginning token or user-message protocol does not match Poolside",
        ));
    }
    Ok(system_inner.unwrap_or("").to_owned())
}

fn validate_explicit_and_empty_system_behavior(
    template_program: &LagunaTemplateProgram,
    bos_token_content: &str,
) -> Result<(), LagunaTextArtifactError> {
    let explicit_system_prompt = render_probe(
        template_program,
        vec![system_message(PROBE_SYSTEM_CONTENT), user_message()],
        Vec::new(),
        Some(false),
        false,
        bos_token_content,
    )?;
    if !matches_system_then_user(
        &explicit_system_prompt,
        bos_token_content,
        Some(PROBE_SYSTEM_CONTENT),
    ) {
        return Err(unsupported_poolside(
            "explicit system-message protocol does not match Poolside",
        ));
    }

    let empty_system_prompt = render_probe(
        template_program,
        vec![system_message(""), user_message()],
        Vec::new(),
        Some(false),
        false,
        bos_token_content,
    )?;
    if !matches_system_then_user(&empty_system_prompt, bos_token_content, None) {
        return Err(unsupported_poolside(
            "empty system-message behavior does not match Poolside",
        ));
    }
    Ok(())
}

fn validate_tool_definition_protocol(
    template_program: &LagunaTemplateProgram,
    bos_token_content: &str,
) -> Result<(), LagunaTextArtifactError> {
    let tool_definition = ChatToolDefinition {
        name: PROBE_TOOL_NAME.to_owned(),
        description: Some("__laguna_probe_description__".to_owned()),
        parameters_json: r#"{"type":"object","properties":{}}"#.to_owned(),
    };
    let rendered_prompt = render_probe(
        template_program,
        vec![system_message(""), user_message()],
        vec![tool_definition],
        Some(false),
        false,
        bos_token_content,
    )?;
    let after_bos = strip_required_prefix(
        &rendered_prompt,
        bos_token_content,
        "tool-definition system block does not match Poolside",
    )?;
    let (system_block, after_system) = take_required_tagged_block(
        after_bos,
        "system",
        "tool-definition system block does not match Poolside",
    )?;
    let (user_inner, after_user) = take_required_tagged_block(
        after_system,
        "user",
        "tool-definition system block does not match Poolside",
    )?;
    if user_inner != PROBE_USER_CONTENT || !after_user.trim().is_empty() {
        return Err(unsupported_poolside(
            "tool-definition system block does not match Poolside",
        ));
    }
    let serialized_tool = extract_available_tools_json(system_block).ok_or(
        unsupported_poolside("available-tools markers do not match Poolside"),
    )?;
    let rendered_tool: Value = serde_json::from_str(serialized_tool).map_err(|_| {
        LagunaTextArtifactError::UnsupportedTemplateContract {
            description: "artifact template does not render an OpenAI-style tool definition",
        }
    })?;
    let expected_tool = json!({
        "type": "function",
        "function": {
            "name": PROBE_TOOL_NAME,
            "description": "__laguna_probe_description__",
            "parameters": {"type": "object", "properties": {}}
        }
    });
    if rendered_tool != expected_tool {
        return Err(LagunaTextArtifactError::UnsupportedTemplateContract {
            description: "artifact template changes OpenAI-style tool semantics",
        });
    }
    Ok(())
}

fn derive_history_protocol(
    template_program: &LagunaTemplateProgram,
    bos_token_content: &str,
) -> Result<bool, LagunaTextArtifactError> {
    let assistant_message = ChatMessage::Assistant {
        content: Some(PROBE_ASSISTANT_CONTENT.to_owned()),
        reasoning_content: Some(PROBE_REASONING_CONTENT.to_owned()),
        tool_calls: vec![ChatAssistantToolCall {
            id: "laguna-probe-call".to_owned(),
            function: ChatAssistantToolFunction {
                name: PROBE_TOOL_NAME.to_owned(),
                arguments_json: r#"{"count":"7","label":"café"}"#.to_owned(),
            },
        }],
    };
    let messages = vec![
        system_message(PROBE_SYSTEM_CONTENT),
        assistant_message,
        ChatMessage::Tool {
            tool_call_id: "laguna-probe-call".to_owned(),
            content: PROBE_TOOL_RESPONSE.to_owned(),
        },
    ];
    let rendered_disabled = render_probe(
        template_program,
        messages.clone(),
        Vec::new(),
        Some(false),
        false,
        bos_token_content,
    )?;
    let preserves_prior_reasoning = history_preserves_reasoning(&rendered_disabled)?;
    if !contains_poolside_history_markers(&rendered_disabled, bos_token_content)? {
        return Err(unsupported_poolside(
            "assistant history, tool-call, or tool-response protocol does not match Poolside",
        ));
    }

    let rendered_enabled = render_probe(
        template_program,
        messages,
        Vec::new(),
        Some(true),
        false,
        bos_token_content,
    )?;
    if !contains_poolside_history_markers(&rendered_enabled, bos_token_content)?
        || !rendered_enabled.contains(PROBE_REASONING_CONTENT)
    {
        return Err(unsupported_poolside(
            "thinking-enabled assistant history does not match Poolside",
        ));
    }
    Ok(preserves_prior_reasoning)
}

fn derive_and_validate_generation_prefixes(
    template_program: &LagunaTemplateProgram,
    bos_token_content: &str,
) -> Result<bool, LagunaTextArtifactError> {
    for (enable_thinking, thinking_enabled) in [(false, false), (true, true)] {
        let rendered_prompt = render_probe(
            template_program,
            vec![system_message(""), user_message()],
            Vec::new(),
            Some(enable_thinking),
            true,
            bos_token_content,
        )?;
        if assistant_thinking_prefix(&rendered_prompt, bos_token_content) != Some(thinking_enabled)
        {
            return Err(unsupported_poolside(
                "assistant thinking prefix does not match Poolside",
            ));
        }
    }

    let rendered_default = render_probe(
        template_program,
        vec![system_message(""), user_message()],
        Vec::new(),
        None,
        true,
        bos_token_content,
    )?;
    assistant_thinking_prefix(&rendered_default, bos_token_content).ok_or(unsupported_poolside(
        "template-default assistant prefix does not match Poolside",
    ))
}

fn render_probe(
    template_program: &LagunaTemplateProgram,
    messages: Vec<ChatMessage>,
    tools: Vec<ChatToolDefinition>,
    enable_thinking: Option<bool>,
    add_generation_prompt: bool,
    bos_token_content: &str,
) -> Result<String, LagunaTextArtifactError> {
    let mut context =
        LagunaTemplateContext::from_chat(&messages, &tools, enable_thinking, bos_token_content)
            .map_err(|_| LagunaTextArtifactError::UnsupportedTemplateContract {
                description: "internal template semantic probe context is invalid",
            })?;
    if !add_generation_prompt {
        context = context.without_generation_prompt();
    }
    template_program
        .render(&context)
        .map_err(|program_error| match program_error {
            LagunaTemplateProgramError::Template(source) => {
                LagunaTextArtifactError::TemplateProbeRendering(source)
            }
            LagunaTemplateProgramError::OutputTooLarge { .. }
            | LagunaTemplateProgramError::OutputNotUtf8(_) => {
                LagunaTextArtifactError::UnsupportedTemplateContract {
                    description: "template semantic probe output is invalid",
                }
            }
        })
}

fn system_message(content: &str) -> ChatMessage {
    ChatMessage::System {
        content: content.to_owned(),
    }
}

fn user_message() -> ChatMessage {
    ChatMessage::User {
        content: PROBE_USER_CONTENT.to_owned(),
        images: Vec::new(),
    }
}

fn unsupported_poolside(description: &'static str) -> LagunaTextArtifactError {
    LagunaTextArtifactError::UnsupportedTemplateContract { description }
}

fn strip_required_prefix<'a>(
    source: &'a str,
    prefix: &str,
    description: &'static str,
) -> Result<&'a str, LagunaTextArtifactError> {
    source
        .strip_prefix(prefix)
        .ok_or(unsupported_poolside(description))
}

fn matches_system_then_user(
    rendered_prompt: &str,
    bos_token_content: &str,
    expected_system_content: Option<&str>,
) -> bool {
    let Some(after_bos) = rendered_prompt.strip_prefix(bos_token_content) else {
        return false;
    };
    let (system_inner, after_system) = take_optional_tagged_block(after_bos, "system");
    let Some((user_inner, after_user)) = take_tagged_block(after_system, "user") else {
        return false;
    };
    system_inner == expected_system_content
        && user_inner == PROBE_USER_CONTENT
        && after_user.trim().is_empty()
}

fn take_optional_tagged_block<'a>(source: &'a str, tag: &str) -> (Option<&'a str>, &'a str) {
    take_tagged_block(source, tag).map_or((None, source), |(inner, rest)| (Some(inner), rest))
}

fn take_required_tagged_block<'a>(
    source: &'a str,
    tag: &str,
    description: &'static str,
) -> Result<(&'a str, &'a str), LagunaTextArtifactError> {
    take_tagged_block(source, tag).ok_or(unsupported_poolside(description))
}

fn take_tagged_block<'a>(source: &'a str, tag: &str) -> Option<(&'a str, &'a str)> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let trimmed_source = source.trim_start();
    let after_open = trimmed_source.strip_prefix(&open)?;
    let after_open = after_open.strip_prefix('\n').unwrap_or(after_open);
    let close_offset = after_open.find(&close)?;
    let inner = after_open[..close_offset].trim();
    let after_close = &after_open[close_offset + close.len()..];
    let after_close = after_close.strip_prefix('\n').unwrap_or(after_close);
    Some((inner, after_close))
}

fn extract_available_tools_json(system_inner: &str) -> Option<&str> {
    let start_marker = "<available_tools>";
    let end_marker = "</available_tools>";
    let start_offset = system_inner.find(start_marker)? + start_marker.len();
    let after_start = system_inner[start_offset..]
        .strip_prefix('\n')
        .unwrap_or(&system_inner[start_offset..]);
    let end_offset = after_start.find(end_marker)?;
    let json_payload = after_start[..end_offset].trim();
    if json_payload.is_empty() || json_payload.contains('\n') {
        return None;
    }
    Some(json_payload)
}

fn contains_poolside_history_markers(
    rendered_prompt: &str,
    bos_token_content: &str,
) -> Result<bool, LagunaTextArtifactError> {
    if !matches_system_then_remainder(rendered_prompt, bos_token_content, PROBE_SYSTEM_CONTENT) {
        return Ok(false);
    }
    Ok(rendered_prompt.contains("<assistant>")
        && rendered_prompt.contains(PROBE_ASSISTANT_CONTENT)
        && rendered_prompt.contains(&format!("<tool_call>{PROBE_TOOL_NAME}"))
        && rendered_prompt.contains("<arg_key>count</arg_key>")
        && rendered_prompt.contains("<arg_key>label</arg_key>")
        && rendered_prompt.contains("café")
        && rendered_prompt.contains(PROBE_TOOL_RESPONSE)
        && rendered_prompt.contains("<tool_response>"))
}

fn matches_system_then_remainder(
    rendered_prompt: &str,
    bos_token_content: &str,
    expected_system_content: &str,
) -> bool {
    let Some(after_bos) = rendered_prompt.strip_prefix(bos_token_content) else {
        return false;
    };
    take_tagged_block(after_bos, "system")
        .is_some_and(|(system_inner, _)| system_inner == expected_system_content)
}

fn history_preserves_reasoning(rendered_disabled: &str) -> Result<bool, LagunaTextArtifactError> {
    if rendered_disabled.contains(PROBE_REASONING_CONTENT) {
        return Ok(true);
    }
    if rendered_disabled.contains("</think>") {
        return Ok(false);
    }
    Err(unsupported_poolside(
        "assistant history, tool-call, or tool-response protocol does not match Poolside",
    ))
}

fn assistant_thinking_prefix(rendered_prompt: &str, bos_token_content: &str) -> Option<bool> {
    if !rendered_prompt.starts_with(bos_token_content) {
        return None;
    }
    let assistant_offset = rendered_prompt.rfind("<assistant>")?;
    let suffix = rendered_prompt[assistant_offset + "<assistant>".len()..].trim();
    match suffix {
        "<think>" => Some(true),
        "</think>" => Some(false),
        _ => None,
    }
}
