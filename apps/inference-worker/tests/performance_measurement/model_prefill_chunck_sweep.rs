use std::time::Duration;

use astronomical_inference_worker::worker_startup::run_bootstrapped_worker;
use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice,
    MAX_IPC_FRAME_BYTES, ProtocolReader, ProtocolWriter, RequestId, WorkerCommand, WorkerEvent,
};
use astronomical_supervisor::ResolvedRuntimeConfigResolver;
use tokio::time::{Instant, timeout};

const MAXIMUM_SUMMARY_TOKENS: u16 = 2_000;
const MODEL_ID: &str = crate::common::ORNITH_MODEL_ARTIFACT_QUALIFICATION_MODEL_ID;
const SOURCE_DOCUMENT_FIXTURE: &str =
    include_str!("../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");
const SWEEP_TIMEOUT: Duration = Duration::from_secs(115);

#[derive(Debug)]
#[allow(dead_code)]
struct PrefillChunckMetrics {
    prefill_chunck_tokens: u32,
    prompt_tokens_per_second: f64,
    generation_tokens_per_second: f64,
    prompt_token_count: u64,
    completion_token_count: u64,
    prompt_processing_seconds: f64,
    generation_seconds: f64,
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the Ornith model in-process and benchmarks prefill_chunck_tokens 1024"]
async fn should_measure_model_throughput_with_prefill_chunck_tokens_1024() {
    assert_valid_prefill_chunck_sweep_result(run_prefill_chunck_sweep_with_timeout(1024).await);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the Ornith model in-process and benchmarks prefill_chunck_tokens 2048 (baseline)"]
async fn should_measure_model_throughput_with_prefill_chunck_tokens_2048() {
    assert_valid_prefill_chunck_sweep_result(run_prefill_chunck_sweep_with_timeout(2048).await);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the Ornith model in-process and benchmarks prefill_chunck_tokens 4096"]
async fn should_measure_model_throughput_with_prefill_chunck_tokens_4096() {
    assert_valid_prefill_chunck_sweep_result(run_prefill_chunck_sweep_with_timeout(4096).await);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the Ornith model in-process and benchmarks prefill_chunck_tokens 8192"]
async fn should_measure_model_throughput_with_prefill_chunck_tokens_8192() {
    assert_valid_prefill_chunck_sweep_result(run_prefill_chunck_sweep_with_timeout(8192).await);
}

async fn run_prefill_chunck_sweep_with_timeout(prefill_chunck_tokens: u32) -> PrefillChunckMetrics {
    timeout(
        SWEEP_TIMEOUT,
        run_model_artifact_with_prefill_chunck_tokens(prefill_chunck_tokens),
    )
    .await
    .expect("the prefill_chunck_tokens benchmark must finish within 115 seconds")
}

fn assert_valid_prefill_chunck_sweep_result(prefill_chunck_metrics: PrefillChunckMetrics) {
    assert!(
        prefill_chunck_metrics.prompt_token_count > 0
            && prefill_chunck_metrics.completion_token_count > 1
    );
}

/// Loads the Ornith model over duplex pipes with one neutral fixed chunk policy.
async fn run_model_artifact_with_prefill_chunck_tokens(
    prefill_chunck_tokens: u32,
) -> PrefillChunckMetrics {
    let (test_to_worker, worker_from_test) = tokio::io::duplex(MAX_IPC_FRAME_BYTES * 4);
    let (worker_to_test, test_from_worker) = tokio::io::duplex(MAX_IPC_FRAME_BYTES * 4);
    let worker_task = tokio::spawn(async move {
        run_bootstrapped_worker(worker_from_test, worker_to_test)
            .await
            .expect("the in-process worker should run successfully");
    });

    let mut protocol_writer = ProtocolWriter::new(test_to_worker);
    let mut protocol_reader = ProtocolReader::new(test_from_worker);

    let isolated_development_home = crate::common::isolated_development_home_from_user_config();
    let worker_runtime_config = ResolvedRuntimeConfigResolver::for_development_home_directory(
        isolated_development_home.path().to_path_buf(),
        std::path::PathBuf::new(),
    )
    .load()
    .expect("the prefill sweep worker configuration should resolve");
    let mut worker_startup_configuration = worker_runtime_config.worker_startup_configuration();
    worker_startup_configuration
        .chunking
        .fixed_prompt_processing_chunk_size_tokens = prefill_chunck_tokens;
    protocol_writer
        .send_command(&WorkerCommand::InitializeWorker(
            worker_startup_configuration,
        ))
        .await
        .expect("the benchmark should initialize its worker");

    let startup_event = protocol_reader
        .next_event()
        .await
        .expect("the worker event stream should be readable")
        .expect("the worker should report idle startup before closing");
    if !matches!(startup_event, WorkerEvent::Idle { .. }) {
        panic!("the first worker event should be Idle, got {startup_event:?}");
    }
    let configured_model_directory =
        crate::common::configured_model_artifact_directory_by_id(MODEL_ID);
    protocol_writer
        .send_command(&WorkerCommand::SwapModel {
            model_directory: configured_model_directory.to_string_lossy().into_owned(),
            max_output_tokens: u32::from(MAXIMUM_SUMMARY_TOKENS),
        })
        .await
        .expect("the benchmark should select its configured model");
    let model_loaded_event = protocol_reader
        .next_event()
        .await
        .expect("the worker event stream should remain readable")
        .expect("the worker should report model loading before closing");
    let WorkerEvent::ModelSwapped { model_id, .. } = model_loaded_event else {
        panic!("the worker should load the benchmark model, got {model_loaded_event:?}");
    };
    eprintln!("[prefill-chunck-sweep-{prefill_chunck_tokens}] worker ready: {model_id}");

    let source_document = static_source_document();
    let user_prompt = format!(
        "Summarize the following document in exactly three concise paragraphs. Do not use headings or bullet points.\n\n{source_document}"
    );
    let chat_command = ChatGenerationCommand {
        request_id: RequestId::new(1),
        model: MODEL_ID.to_owned(),
        messages: vec![ChatMessage::User {
            content: user_prompt,
            images: Vec::new(),
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: MAXIMUM_SUMMARY_TOKENS,
            temperature_thousandths: None,
            top_p_thousandths: None,
            seed: None,
            thinking_budget: Some(256),
        },
    };

    eprintln!("[prefill-chunck-sweep-{prefill_chunck_tokens}] sending generate command");
    let request_started_at = Instant::now();
    protocol_writer
        .send_command(&WorkerCommand::Generate(chat_command))
        .await
        .expect("the test should send the generate command");

    let mut first_output_at: Option<Instant> = None;
    let mut prompt_token_count: Option<u64> = None;
    let mut completion_token_count: Option<u64> = None;

    loop {
        let Some(worker_event) = protocol_reader
            .next_event()
            .await
            .expect("the worker event stream should be readable")
        else {
            break;
        };
        match worker_event {
            WorkerEvent::Output { .. } => {
                if first_output_at.is_none() {
                    first_output_at = Some(Instant::now());
                    eprintln!(
                        "[prefill-chunck-sweep-{prefill_chunck_tokens}] first output received"
                    );
                }
            }
            WorkerEvent::Completed {
                prompt_token_count: completed_prompt_tokens,
                generated_token_count,
                ..
            } => {
                prompt_token_count = Some(u64::from(completed_prompt_tokens));
                completion_token_count = Some(u64::from(generated_token_count));
                break;
            }
            WorkerEvent::Failed { reason, .. } => {
                panic!("the worker failed the prefill_chunck_tokens sweep request: {reason:?}");
            }
            WorkerEvent::ModelSwapFailed { .. } => {
                panic!("the worker failed to load the prefill sweep model");
            }
            WorkerEvent::PrefillProgress { .. }
            | WorkerEvent::GenerationPreparationStarted { .. }
            | WorkerEvent::FirstDecodeCompleted { .. }
            | WorkerEvent::GenerationProgress { .. }
            | WorkerEvent::PromptWorkReuse { .. }
            | WorkerEvent::MlxMemorySample { .. }
            | WorkerEvent::MlxMemoryLimitChanged { .. }
            | WorkerEvent::MlxMemoryLimitRejected { .. }
            | WorkerEvent::GenerationFinalized { .. }
            | WorkerEvent::ExpertMemoryModeChanged { .. }
            | WorkerEvent::PersistentPromptCacheStats { .. }
            | WorkerEvent::PromptCacheCleared { .. }
            | WorkerEvent::ModelSwapped { .. }
            | WorkerEvent::Idle { .. }
            | WorkerEvent::RuntimeFeatureConfigurationApplied { .. }
            | WorkerEvent::Ready { .. } => {}
        }
    }
    let response_completed_at = Instant::now();

    // Best-effort cleanup: measurement already captured, close/join errors irrelevant.
    let _close_outcome = protocol_writer.close().await;
    let _worker_join = worker_task.await;

    let first_output_at = first_output_at.unwrap_or(response_completed_at);
    let prompt_processing_duration = first_output_at.duration_since(request_started_at);
    let generation_duration = response_completed_at.duration_since(first_output_at);
    let prompt_tokens_per_second =
        prompt_token_count.unwrap_or(0) as f64 / prompt_processing_duration.as_secs_f64();
    let generation_tokens_per_second = (completion_token_count.unwrap_or(0).saturating_sub(1))
        as f64
        / generation_duration.as_secs_f64();
    eprintln!(
        "[prefill-chunck-sweep-{prefill_chunck_tokens}] done: prompt={prompt_tokens_per_second:.1} tok/s, generation={generation_tokens_per_second:.1} tok/s"
    );

    PrefillChunckMetrics {
        prefill_chunck_tokens,
        prompt_tokens_per_second,
        generation_tokens_per_second,
        prompt_token_count: prompt_token_count.unwrap_or(0),
        completion_token_count: completion_token_count.unwrap_or(0),
        prompt_processing_seconds: prompt_processing_duration.as_secs_f64(),
        generation_seconds: generation_duration.as_secs_f64(),
    }
}

fn static_source_document() -> String {
    SOURCE_DOCUMENT_FIXTURE
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}
