use std::path::{Path, PathBuf};

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use astronomical_model_serving::Qwen3_5Tokenizer;

use super::RepresentativePrompt;
use super::tool_process_restart::SpeculativePrefillProcessPassReport;

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
        let prepared_prompt = prepared_prompt_from_messages(
            tokenizer,
            target_model_id,
            declared_tools,
            initial_tool_messages_with_system_instruction(&source_material, system_instruction),
        );
        if prepared_prompt.prompt_token_ids.len() >= PROCESS_RESTART_MINIMUM_PROMPT_TOKEN_COUNT {
            return source_material;
        }
    }
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
                qwen_thinking_channel_seed: None,
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
        sampling_temperature_thousandths: 1_000,
        sampling_top_p_thousandths: 1_000,
        sampling_seed: None,
    }
}

fn process_pass_generation_settings() -> ChatGenerationSettings {
    ChatGenerationSettings {
        max_output_tokens: PROCESS_RESTART_OUTPUT_TOKEN_COUNT,
        temperature_thousandths: None,
        top_p_thousandths: None,
        seed: None,
        thinking_budget: Some(256),
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

pub(super) fn file_count_under_directory(directory_path: &Path) -> usize {
    match std::fs::read_dir(directory_path) {
        Ok(directory_entries) => directory_entries.filter_map(Result::ok).fold(
            0,
            |published_file_count, directory_entry| {
                let entry_path = directory_entry.path();
                if entry_path.is_dir() {
                    published_file_count + file_count_under_directory(&entry_path)
                } else if entry_path.is_file() {
                    published_file_count + 1
                } else {
                    published_file_count
                }
            },
        ),
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
