use std::path::{Path, PathBuf};

use astronomical_ipc_protocol::{
    ChatAssistantToolCall, ChatAssistantToolFunction, ChatGenerationCommand,
    ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use astronomical_model_serving::Qwen3_5Tokenizer;

use super::speculative_prefill::RepresentativePrompt;
use super::speculative_prefill_tool_process_restart::SpeculativePrefillProcessPassReport;

const PROCESS_RESTART_MINIMUM_PROMPT_TOKEN_COUNT: usize = 8_192;
const PROCESS_RESTART_OUTPUT_TOKEN_COUNT: u16 = 256;
const ROMEO_AND_JULIET_SOURCE: &str = include_str!(
    "../../../../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);
const BASELINE_SYSTEM_INSTRUCTION: &str = "Use the declared tool and return its required fields.";

pub(super) fn prepare_natural_tool_prompt(
    tokenizer: &Qwen3_5Tokenizer,
    target_model_id: &str,
    declared_tools: &[astronomical_ipc_protocol::ChatToolDefinition],
) -> RepresentativePrompt {
    prepare_natural_tool_prompt_with_system_instruction(
        tokenizer,
        target_model_id,
        declared_tools,
        BASELINE_SYSTEM_INSTRUCTION,
    )
}

pub(super) fn prepare_natural_tool_prompt_with_system_instruction(
    tokenizer: &Qwen3_5Tokenizer,
    target_model_id: &str,
    declared_tools: &[astronomical_ipc_protocol::ChatToolDefinition],
    system_instruction: &str,
) -> RepresentativePrompt {
    let source_material = repeated_source_material_for_minimum_tokens(
        tokenizer,
        target_model_id,
        declared_tools,
        system_instruction,
    );
    prepared_prompt_from_messages(
        tokenizer,
        target_model_id,
        declared_tools,
        initial_tool_messages_with_system_instruction(&source_material, system_instruction),
    )
}

pub(super) fn prepare_natural_tool_follow_up_prompt(
    tokenizer: &Qwen3_5Tokenizer,
    target_model_id: &str,
    declared_tools: &[astronomical_ipc_protocol::ChatToolDefinition],
    function_name: &str,
    arguments_json: &str,
) -> RepresentativePrompt {
    let initial_source_material = repeated_source_material_for_minimum_tokens(
        tokenizer,
        target_model_id,
        declared_tools,
        BASELINE_SYSTEM_INSTRUCTION,
    );
    let initial_prompt_token_count = prepared_prompt_from_messages(
        tokenizer,
        target_model_id,
        declared_tools,
        initial_tool_messages(&initial_source_material),
    )
    .prompt_token_ids
    .len();
    let mut follow_up_source_material = String::new();
    loop {
        if !follow_up_source_material.is_empty() {
            follow_up_source_material.push_str("\n\n");
        }
        follow_up_source_material.push_str(ROMEO_AND_JULIET_SOURCE);
        let prepared_follow_up_prompt = prepared_prompt_from_messages(
            tokenizer,
            target_model_id,
            declared_tools,
            tool_follow_up_messages(
                &initial_source_material,
                &follow_up_source_material,
                function_name,
                arguments_json,
            ),
        );
        if prepared_follow_up_prompt.prompt_token_ids.len()
            >= initial_prompt_token_count + PROCESS_RESTART_MINIMUM_PROMPT_TOKEN_COUNT
        {
            return prepared_follow_up_prompt;
        }
    }
}

fn repeated_source_material_for_minimum_tokens(
    tokenizer: &Qwen3_5Tokenizer,
    target_model_id: &str,
    declared_tools: &[astronomical_ipc_protocol::ChatToolDefinition],
    system_instruction: &str,
) -> String {
    let mut source_material = String::new();
    loop {
        if !source_material.is_empty() {
            source_material.push_str("\n\n");
        }
        source_material.push_str(ROMEO_AND_JULIET_SOURCE);
        let prepared_request = tokenizer
            .prepare_chat(
                &ChatGenerationCommand {
                    request_id: RequestId::new(95_399),
                    model: target_model_id.to_owned(),
                    messages: initial_tool_messages_with_system_instruction(
                        &source_material,
                        system_instruction,
                    ),
                    tools: declared_tools.to_vec(),
                    tool_choice: ChatToolChoice::Auto,
                    settings: process_pass_generation_settings(),
                },
                false,
            )
            .expect("the natural tool process prompt should prepare");
        if prepared_request.input_token_ids().len() >= PROCESS_RESTART_MINIMUM_PROMPT_TOKEN_COUNT {
            return source_material;
        }
    }
}

fn initial_tool_messages(source_material: &str) -> Vec<ChatMessage> {
    initial_tool_messages_with_system_instruction(source_material, BASELINE_SYSTEM_INSTRUCTION)
}

fn initial_tool_messages_with_system_instruction(
    source_material: &str,
    system_instruction: &str,
) -> Vec<ChatMessage> {
    vec![
        ChatMessage::System {
            content: system_instruction.to_owned(),
        },
        ChatMessage::User {
            content: format!(
                "Call record_literary_analysis now. Record the play's central conflict and classify its outcome as tragic. Use this source material as evidence.\n\n{source_material}"
            ),
            images: Vec::new(),
        },
    ]
}

fn tool_follow_up_messages(
    initial_source_material: &str,
    follow_up_source_material: &str,
    function_name: &str,
    arguments_json: &str,
) -> Vec<ChatMessage> {
    let mut messages = initial_tool_messages(initial_source_material);
    messages.push(ChatMessage::Assistant {
        content: None,
        reasoning_content: None,
        tool_calls: vec![ChatAssistantToolCall {
            id: "call_issue_50_process_restart".to_owned(),
            function: ChatAssistantToolFunction {
                name: function_name.to_owned(),
                arguments_json: arguments_json.to_owned(),
            },
        }],
    });
    messages.push(ChatMessage::Tool {
        tool_call_id: "call_issue_50_process_restart".to_owned(),
        content: "The literary analysis was recorded successfully.".to_owned(),
    });
    messages.push(ChatMessage::User {
        content: format!(
            "Now call the same tool again with a refined central-conflict summary grounded in this additional Romeo and Juliet material. Keep the outcome tragic.\n\n{follow_up_source_material}"
        ),
        images: Vec::new(),
    });
    messages
}

fn prepared_prompt_from_messages(
    tokenizer: &Qwen3_5Tokenizer,
    target_model_id: &str,
    declared_tools: &[astronomical_ipc_protocol::ChatToolDefinition],
    messages: Vec<ChatMessage>,
) -> RepresentativePrompt {
    let prepared_request = tokenizer
        .prepare_chat(
            &ChatGenerationCommand {
                request_id: RequestId::new(95_398),
                model: target_model_id.to_owned(),
                messages,
                tools: declared_tools.to_vec(),
                tool_choice: ChatToolChoice::Auto,
                settings: process_pass_generation_settings(),
            },
            false,
        )
        .expect("the natural process-restart prompt should prepare");
    RepresentativePrompt {
        prompt_token_ids: prepared_request.input_token_ids().to_vec(),
        image_pad_token_id: tokenizer.image_pad_token_id(),
        processed_visual_images: Vec::new(),
        ordinary_target_prefill_control_span_token_count: prepared_request
            .ordinary_target_prefill_control_span_token_count(),
        sampling_temperature_thousandths: 0,
        sampling_top_p_thousandths: 1_000,
        sampling_seed: None,
    }
}

fn process_pass_generation_settings() -> ChatGenerationSettings {
    ChatGenerationSettings {
        max_output_tokens: PROCESS_RESTART_OUTPUT_TOKEN_COUNT,
        temperature_thousandths: Some(0),
        top_p_thousandths: Some(1_000),
        seed: None,
        thinking_budget: Some(0),
    }
}

pub(super) fn required_environment_path(
    environment_variable_name: &str,
    failure_message: &str,
) -> PathBuf {
    std::env::var_os(environment_variable_name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{failure_message}"))
}

pub(super) fn read_process_pass_report(
    process_report_path: &Path,
) -> SpeculativePrefillProcessPassReport {
    serde_json::from_slice(
        &std::fs::read(process_report_path)
            .expect("the isolated process pass report should be readable"),
    )
    .expect("the isolated process pass report should contain valid JSON")
}

pub(super) fn file_count_in_directory(directory_path: &Path) -> usize {
    match std::fs::read_dir(directory_path) {
        Ok(directory_entries) => directory_entries
            .filter_map(Result::ok)
            .filter(|directory_entry| directory_entry.path().is_file())
            .count(),
        Err(directory_read_error)
            if directory_read_error.kind() == std::io::ErrorKind::NotFound =>
        {
            0
        }
        Err(directory_read_error) => {
            panic!(
                "failed to read {}: {directory_read_error}",
                directory_path.display()
            )
        }
    }
}
