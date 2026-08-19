//! Same-process SSD restore journey for the reference Laguna extra-small artifact.

use std::process::{Command, Stdio};
use std::thread;
use std::time::Instant;

use astronomical_config::PromptCacheConfig;
use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
    WorkerChunkingConfiguration,
};
use astronomical_model_serving::{
    GeneratedToken, LagunaServingSettings, MlxInferenceExecution,
    initialize_laguna_execution_with_serving_settings,
};

use super::artifact::{
    LAGUNA_XS_PUBLIC_MODEL_ID, QUALIFICATION_POLL_INTERVAL, QUALIFICATION_PROGRESS_INTERVAL,
    QUALIFICATION_TIMEOUT, bounded_romeo_and_juliet_source, resolve_reference_model_directory,
};
use super::generate::resolve_machine_mlx_memory_limits;

const QUALIFICATION_CHILD_MODEL_ID: &str = "ASTRONOMICAL_LAGUNA_RESTORE_CHILD_MODEL_ID";
const GENERATED_TOKEN_LIMIT: usize = 8;
const PROMPT_CACHE_BLOCK_TOKEN_COUNT: u32 = 256;

#[test]
#[ignore = "loads the reference Laguna XS artifact and proves same-process SSD prompt-cache restore"]
fn should_restore_romeo_and_juliet_prompt_prefix_from_the_ssd_cache() {
    if std::env::var(QUALIFICATION_CHILD_MODEL_ID).as_deref() == Ok(LAGUNA_XS_PUBLIC_MODEL_ID) {
        restore_from_reference_artifact();
        return;
    }
    eprintln!("[laguna-restore] starting GPU restore journey model={LAGUNA_XS_PUBLIC_MODEL_ID}");
    let test_executable =
        std::env::current_exe().expect("the qualification test executable path should resolve");
    let mut child_process = Command::new(test_executable)
        .args([
            "model_artifact_qualification::laguna::restore::should_restore_romeo_and_juliet_prompt_prefix_from_the_ssd_cache",
            "--exact",
            "--ignored",
            "--nocapture",
            "--test-threads",
            "1",
        ])
        .env(QUALIFICATION_CHILD_MODEL_ID, LAGUNA_XS_PUBLIC_MODEL_ID)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("the isolated Laguna restore process should start");
    let start_time = Instant::now();
    let mut next_progress_time = QUALIFICATION_PROGRESS_INTERVAL;
    loop {
        if let Some(exit_status) = child_process
            .try_wait()
            .expect("the Laguna restore process status should be readable")
        {
            assert!(
                exit_status.success(),
                "isolated Laguna restore journey failed for {LAGUNA_XS_PUBLIC_MODEL_ID}"
            );
            eprintln!("[laguna-restore] completed model={LAGUNA_XS_PUBLIC_MODEL_ID}");
            return;
        }
        let elapsed_time = start_time.elapsed();
        if elapsed_time >= QUALIFICATION_TIMEOUT {
            let _kill_outcome = child_process.kill();
            let _wait_outcome = child_process.wait();
            panic!(
                "Laguna restore journey exceeded {} seconds for {LAGUNA_XS_PUBLIC_MODEL_ID}",
                QUALIFICATION_TIMEOUT.as_secs(),
            );
        }
        if elapsed_time >= next_progress_time {
            eprintln!(
                "[laguna-restore] loading model={LAGUNA_XS_PUBLIC_MODEL_ID} elapsed_seconds={}",
                elapsed_time.as_secs()
            );
            next_progress_time += QUALIFICATION_PROGRESS_INTERVAL;
        }
        thread::sleep(QUALIFICATION_POLL_INTERVAL);
    }
}

fn restore_from_reference_artifact() {
    let model_directory = resolve_reference_model_directory();
    let (active_memory_limit_bytes, allocator_cache_memory_limit_bytes) =
        resolve_machine_mlx_memory_limits();
    let prompt_cache_home =
        tempfile::tempdir().expect("a temporary prompt-cache root should exist");
    let serving_settings = LagunaServingSettings {
        chunking: Some(WorkerChunkingConfiguration {
            fixed_prompt_processing_chunk_size_tokens: 8_192,
            fixed_ssd_streaming_prompt_processing_chunk_size_tokens: None,
            full_attention_key_value_growth_tokens: 256,
            speculative_prefill_draft_forward_tokens: 2_048,
            prefill_graph_submission_layer_interval: 1,
            experimental_ssd_paging_generation_graph_submission_layer_interval: 0,
            prompt_cache_block_tokens: Some(PROMPT_CACHE_BLOCK_TOKEN_COUNT),
            prompt_cache_common_prefix_stride_blocks: 1,
        }),
        persistent_prompt_cache_enabled: true,
        prompt_cache_config: Some(PromptCacheConfig::new(
            prompt_cache_home.path().to_path_buf(),
            80_000_000_000,
        )),
        performance_attribution_log_path: None,
    };
    eprintln!(
        "[laguna-restore] phase=load model={LAGUNA_XS_PUBLIC_MODEL_ID} cache_root={}",
        prompt_cache_home.path().display()
    );
    let load_started_at = Instant::now();
    let (generation_processor, mut execution) = initialize_laguna_execution_with_serving_settings(
        &model_directory,
        active_memory_limit_bytes,
        allocator_cache_memory_limit_bytes,
        true,
        serving_settings,
    )
    .expect("Laguna startup should construct cache-enabled execution");
    execution
        .load()
        .expect("Laguna weights should load on the GPU");
    eprintln!(
        "[laguna-restore] phase=loaded elapsed_seconds={}",
        load_started_at.elapsed().as_secs()
    );

    let first_generation =
        generate_once(&generation_processor, &mut execution, RequestId::new(105));
    assert_eq!(
        first_generation.restored_token_count, 0,
        "the first Romeo and Juliet request must miss the empty cache"
    );
    assert!(
        !first_generation.generated_token_ids.is_empty(),
        "the first request must emit Romeo and Juliet tokens"
    );
    eprintln!(
        "[laguna-restore] phase=first-miss token_count={} text={:?}",
        first_generation.generated_token_ids.len(),
        first_generation.generated_text
    );

    let second_generation =
        generate_once(&generation_processor, &mut execution, RequestId::new(205));
    eprintln!(
        "[laguna-restore] phase=second-hit restored_tokens={} token_count={} text={:?}",
        second_generation.restored_token_count,
        second_generation.generated_token_ids.len(),
        second_generation.generated_text
    );
    assert!(
        second_generation.restored_token_count >= PROMPT_CACHE_BLOCK_TOKEN_COUNT,
        "the second Romeo and Juliet request must restore at least one SSD cache block"
    );
    assert_eq!(
        first_generation.generated_token_ids, second_generation.generated_token_ids,
        "restored greedy tokens must match the cold first request"
    );
}

struct RestoreGenerationOutcome {
    restored_token_count: u32,
    generated_token_ids: Vec<u32>,
    generated_text: String,
}

fn generate_once(
    generation_processor: &astronomical_model_serving::LagunaGenerationProcessor,
    execution: &mut astronomical_model_serving::LagunaInferenceExecution,
    request_id: RequestId,
) -> RestoreGenerationOutcome {
    let source_excerpt = bounded_romeo_and_juliet_source();
    let chat_command = ChatGenerationCommand {
        request_id,
        model: LAGUNA_XS_PUBLIC_MODEL_ID.to_owned(),
        messages: vec![ChatMessage::User {
            content: format!(
                "Use the supplied Romeo and Juliet source as the only evidence. In two short sentences name the two households and the tragic ending.\n\n{source_excerpt}"
            ),
            images: Vec::new(),
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: GENERATED_TOKEN_LIMIT as u16,
            temperature_thousandths: None,
            top_p_thousandths: None,
            seed: Some(105),
            thinking_budget: Some(0),
        },
    };
    let prepared_generation = generation_processor
        .prepare_chat(&chat_command)
        .expect("the Romeo and Juliet prompt should prepare");
    eprintln!(
        "[laguna-restore] phase=prompt request={} prompt_token_count={}",
        request_id.value(),
        prepared_generation.prompt_token_ids().len()
    );
    let generation_start = execution
        .start_generation(prepared_generation.into_inference_request())
        .expect("Laguna prompt processing should start");
    let mut token_decoder = generation_processor.incremental_decoder();
    let mut generated_token_ids = Vec::new();
    let mut generated_text = String::new();
    while generated_token_ids.len() < GENERATED_TOKEN_LIMIT {
        match execution
            .decode_next_token(chat_command.request_id)
            .expect("Laguna should emit a generated token or end")
        {
            GeneratedToken::TokenId { token_id, .. } => {
                let generated_token_index = generated_token_ids.len();
                generated_token_ids.push(token_id);
                if let Some(decoded_piece) = token_decoder
                    .push_token(token_id)
                    .expect("generated tokens should stay inside the certified vocabulary")
                {
                    generated_text.push_str(&decoded_piece);
                }
                eprintln!(
                    "[laguna-restore] phase=token request={} index={} token_id={} text_so_far={:?}",
                    request_id.value(),
                    generated_token_index,
                    token_id,
                    generated_text
                );
            }
            GeneratedToken::PrefillProgress {
                processed_token_count,
                elapsed_millis,
                ..
            } => {
                eprintln!(
                    "[laguna-restore] phase=prefill request={} processed_tokens={} elapsed_millis={}",
                    request_id.value(),
                    processed_token_count,
                    elapsed_millis
                );
            }
            GeneratedToken::EndOfSequence => break,
            other => panic!("Laguna restore produced an unexpected boundary: {other:?}"),
        }
    }
    execution
        .cancel_generation(chat_command.request_id)
        .expect("the loaded Laguna engine should remain reusable after cancel");
    RestoreGenerationOutcome {
        restored_token_count: generation_start.cached_token_count(),
        generated_token_ids,
        generated_text,
    }
}
