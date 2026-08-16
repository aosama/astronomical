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
    let user_block = format!("<user>{PROBE_USER_CONTENT}</user>\n");
    let header = rendered_prompt
        .strip_prefix(bos_token_content)
        .and_then(|without_bos| without_bos.strip_suffix(&user_block))
        .ok_or(LagunaTextArtifactError::UnsupportedTemplateContract {
            description: "beginning token or user-message protocol does not match Poolside",
        })?;
    if header.is_empty() {
        return Ok(String::new());
    }
    header
        .strip_prefix("<system>")
        .and_then(|system_block| system_block.strip_suffix("</system>\n"))
        .map(str::to_owned)
        .ok_or(LagunaTextArtifactError::UnsupportedTemplateContract {
            description: "default system-message protocol does not match Poolside",
        })
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
    let expected_explicit = format!(
        "{bos_token_content}<system>{PROBE_SYSTEM_CONTENT}</system>\n\
         <user>{PROBE_USER_CONTENT}</user>\n"
    );
    if explicit_system_prompt != expected_explicit {
        return Err(LagunaTextArtifactError::UnsupportedTemplateContract {
            description: "explicit system-message protocol does not match Poolside",
        });
    }

    let empty_system_prompt = render_probe(
        template_program,
        vec![system_message(""), user_message()],
        Vec::new(),
        Some(false),
        false,
        bos_token_content,
    )?;
    let expected_empty = format!("{bos_token_content}<user>{PROBE_USER_CONTENT}</user>\n");
    if empty_system_prompt != expected_empty {
        return Err(LagunaTextArtifactError::UnsupportedTemplateContract {
            description: "empty system-message behavior does not match Poolside",
        });
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
    let user_block = format!("<user>{PROBE_USER_CONTENT}</user>\n");
    let system_block = rendered_prompt
        .strip_prefix(bos_token_content)
        .and_then(|without_bos| without_bos.strip_suffix(&user_block))
        .and_then(|header| header.strip_prefix("<system>"))
        .and_then(|header| header.strip_suffix("</system>\n"))
        .ok_or(LagunaTextArtifactError::UnsupportedTemplateContract {
            description: "tool-definition system block does not match Poolside",
        })?;
    let serialized_tool = system_block
        .split_once("<available_tools>\n")
        .and_then(|(_, available_tools)| available_tools.strip_suffix("</available_tools>"))
        .and_then(|available_tools| available_tools.strip_suffix('\n'))
        .filter(|available_tools| !available_tools.contains('\n'))
        .ok_or(LagunaTextArtifactError::UnsupportedTemplateContract {
            description: "available-tools markers do not match Poolside",
        })?;
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
    let history_suffix_without_reasoning = history_suffix(false);
    let history_suffix_with_reasoning = history_suffix(true);
    let expected_header = format!("{bos_token_content}<system>{PROBE_SYSTEM_CONTENT}</system>\n");
    let preserves_prior_reasoning = if rendered_disabled
        == format!("{expected_header}{history_suffix_without_reasoning}")
    {
        false
    } else if rendered_disabled == format!("{expected_header}{history_suffix_with_reasoning}") {
        true
    } else {
        return Err(LagunaTextArtifactError::UnsupportedTemplateContract {
            description: "assistant history, tool-call, or tool-response protocol does not match Poolside",
        });
    };

    let rendered_enabled = render_probe(
        template_program,
        messages,
        Vec::new(),
        Some(true),
        false,
        bos_token_content,
    )?;
    if rendered_enabled != format!("{expected_header}{history_suffix_with_reasoning}") {
        return Err(LagunaTextArtifactError::UnsupportedTemplateContract {
            description: "thinking-enabled assistant history does not match Poolside",
        });
    }
    Ok(preserves_prior_reasoning)
}

fn derive_and_validate_generation_prefixes(
    template_program: &LagunaTemplateProgram,
    bos_token_content: &str,
) -> Result<bool, LagunaTextArtifactError> {
    for (enable_thinking, expected_suffix) in
        [(false, "<assistant></think>"), (true, "<assistant><think>")]
    {
        let rendered_prompt = render_probe(
            template_program,
            vec![system_message(""), user_message()],
            Vec::new(),
            Some(enable_thinking),
            true,
            bos_token_content,
        )?;
        if !rendered_prompt.starts_with(bos_token_content)
            || !rendered_prompt.ends_with(expected_suffix)
        {
            return Err(LagunaTextArtifactError::UnsupportedTemplateContract {
                description: "assistant thinking prefix does not match Poolside",
            });
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
    if rendered_default.ends_with("<assistant><think>") {
        Ok(true)
    } else if rendered_default.ends_with("<assistant></think>") {
        Ok(false)
    } else {
        Err(LagunaTextArtifactError::UnsupportedTemplateContract {
            description: "template-default assistant prefix does not match Poolside",
        })
    }
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

fn history_suffix(includes_reasoning: bool) -> String {
    let reasoning_block = if includes_reasoning {
        format!("<think>{PROBE_REASONING_CONTENT}</think>")
    } else {
        "</think>".to_owned()
    };
    format!(
        "<assistant>{reasoning_block}{PROBE_ASSISTANT_CONTENT}\
         <tool_call>{PROBE_TOOL_NAME}<arg_key>count</arg_key><arg_value>7</arg_value>\
         <arg_key>label</arg_key><arg_value>café</arg_value></tool_call></assistant>\n\
         <tool_response>{PROBE_TOOL_RESPONSE}</tool_response>\n"
    )
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
