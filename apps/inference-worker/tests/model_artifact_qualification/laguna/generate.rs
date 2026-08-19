//! Isolated GPU generate journey for the reference Laguna XS artifact.

use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use astronomical_model_serving::{
    GeneratedToken, MlxInferenceExecution, initialize_laguna_execution,
};
use astronomical_runtime_integration::maximum_recommended_gpu_working_set_size_bytes;

use super::artifact::{
    LAGUNA_XS_PUBLIC_MODEL_ID, bounded_romeo_and_juliet_source, resolve_reference_model_directory,
};

const QUALIFICATION_TIMEOUT: Duration = Duration::from_secs(115);
const QUALIFICATION_POLL_INTERVAL: Duration = Duration::from_millis(100);
const QUALIFICATION_PROGRESS_INTERVAL: Duration = Duration::from_secs(5);
const QUALIFICATION_CHILD_MODEL_ID: &str = "ASTRONOMICAL_LAGUNA_GENERATE_CHILD_MODEL_ID";
const GENERATED_TOKEN_LIMIT: usize = 8;
const IOGPU_WIRED_LIMIT_SYSCTL_KEY: &str = "iogpu.wired_limit_mb";

#[test]
#[ignore = "loads the reference Laguna XS artifact onto the GPU and generates Romeo and Juliet tokens"]
fn should_generate_romeo_and_juliet_tokens_from_the_reference_laguna_xs_artifact() {
    run_bounded_generate(
        "model_artifact_qualification::laguna::generate::should_generate_romeo_and_juliet_tokens_from_the_reference_laguna_xs_artifact",
    );
}

fn run_bounded_generate(test_name: &str) {
    if std::env::var(QUALIFICATION_CHILD_MODEL_ID).as_deref() == Ok(LAGUNA_XS_PUBLIC_MODEL_ID) {
        generate_from_reference_artifact();
        return;
    }
    eprintln!("[laguna-generate] starting GPU journey model={LAGUNA_XS_PUBLIC_MODEL_ID}");
    let test_executable =
        std::env::current_exe().expect("the qualification test executable path should resolve");
    let mut child_process = Command::new(test_executable)
        .args([
            test_name,
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
        .expect("the isolated Laguna generate process should start");
    let start_time = Instant::now();
    let mut next_progress_time = QUALIFICATION_PROGRESS_INTERVAL;
    loop {
        if let Some(exit_status) = child_process
            .try_wait()
            .expect("the Laguna generate process status should be readable")
        {
            assert!(
                exit_status.success(),
                "isolated Laguna generate journey failed for {LAGUNA_XS_PUBLIC_MODEL_ID}"
            );
            eprintln!("[laguna-generate] completed model={LAGUNA_XS_PUBLIC_MODEL_ID}");
            return;
        }
        let elapsed_time = start_time.elapsed();
        if elapsed_time >= QUALIFICATION_TIMEOUT {
            let _kill_outcome = child_process.kill();
            let _wait_outcome = child_process.wait();
            panic!(
                "Laguna generate journey exceeded {} seconds for {LAGUNA_XS_PUBLIC_MODEL_ID}",
                QUALIFICATION_TIMEOUT.as_secs(),
            );
        }
        if elapsed_time >= next_progress_time {
            eprintln!(
                "[laguna-generate] loading model={LAGUNA_XS_PUBLIC_MODEL_ID} elapsed_seconds={}",
                elapsed_time.as_secs()
            );
            next_progress_time += QUALIFICATION_PROGRESS_INTERVAL;
        }
        thread::sleep(QUALIFICATION_POLL_INTERVAL);
    }
}

fn generate_from_reference_artifact() {
    let model_directory = resolve_reference_model_directory();
    let (active_memory_limit_bytes, allocator_cache_memory_limit_bytes) =
        resolve_machine_mlx_memory_limits();
    eprintln!(
        "[laguna-generate] phase=load model={LAGUNA_XS_PUBLIC_MODEL_ID} active_limit_bytes={} cache_limit_bytes={}",
        active_memory_limit_bytes, allocator_cache_memory_limit_bytes
    );
    let load_started_at = Instant::now();
    let (generation_processor, mut execution) = initialize_laguna_execution(
        &model_directory,
        active_memory_limit_bytes,
        allocator_cache_memory_limit_bytes,
        true,
    )
    .expect("Laguna startup should construct execution");
    let load_result = execution
        .load()
        .expect("Laguna weights should load on the GPU");
    eprintln!(
        "[laguna-generate] phase=loaded elapsed_seconds={} expert_memory_mode={:?}",
        load_started_at.elapsed().as_secs(),
        load_result.expert_memory_mode()
    );

    // Residency depends on the user's machine ceiling, so qualification must
    // accept every executable mode selected by the centralized memory policy.
    assert!(
        load_result.expert_memory_mode().is_some(),
        "expert memory mode must be reported after loading"
    );

    let source_excerpt = bounded_romeo_and_juliet_source();
    let chat_command = ChatGenerationCommand {
        request_id: RequestId::new(106),
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
            seed: Some(106),
            thinking_budget: Some(0),
        },
    };
    let prepared_generation = generation_processor
        .prepare_chat(&chat_command)
        .expect("the Romeo and Juliet prompt should prepare");
    eprintln!(
        "[laguna-generate] phase=prompt prompt_token_count={}",
        prepared_generation.prompt_token_ids().len()
    );
    execution
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
                    .expect("generated tokens should stay inside the vocabulary")
                {
                    generated_text.push_str(&decoded_piece);
                }
                eprintln!(
                    "[laguna-generate] phase=token index={} token_id={} text_so_far={:?}",
                    generated_token_index, token_id, generated_text
                );
            }
            GeneratedToken::PrefillProgress {
                processed_token_count,
                elapsed_millis,
                ..
            } => {
                eprintln!(
                    "[laguna-generate] phase=prefill processed_tokens={} elapsed_millis={}",
                    processed_token_count, elapsed_millis
                );
            }
            GeneratedToken::EndOfSequence => break,
            other => panic!("Laguna generate produced an unexpected boundary: {other:?}"),
        }
    }
    execution
        .cancel_generation(chat_command.request_id)
        .expect("the loaded Laguna engine should remain reusable after cancel");
    assert!(
        !generated_token_ids.is_empty(),
        "Laguna must emit at least one token for the Romeo and Juliet prompt"
    );
    eprintln!(
        "[laguna-generate] phase=done token_count={} text={:?}",
        generated_token_ids.len(),
        generated_text
    );
}

pub(super) fn resolve_machine_mlx_memory_limits() -> (usize, usize) {
    let sysctl_output = Command::new("/usr/sbin/sysctl")
        .args(["-n", IOGPU_WIRED_LIMIT_SYSCTL_KEY])
        .output()
        .expect("iogpu.wired_limit_mb should be readable");
    let wired_limit_text = String::from_utf8_lossy(&sysctl_output.stdout);
    let wired_limit_mebibytes = wired_limit_text
        .trim()
        .parse::<u64>()
        .expect("iogpu.wired_limit_mb should be an unsigned integer");
    let machine_ceiling_bytes = if wired_limit_mebibytes == 0 {
        maximum_recommended_gpu_working_set_size_bytes()
            .expect("the machine GPU working-set size should be readable")
    } else {
        usize::try_from(wired_limit_mebibytes.saturating_mul(1024 * 1024))
            .expect("the wired-memory ceiling should fit the platform integer range")
    };
    (machine_ceiling_bytes, machine_ceiling_bytes)
}
