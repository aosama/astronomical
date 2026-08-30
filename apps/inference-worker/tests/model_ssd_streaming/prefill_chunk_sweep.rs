//! Compares fixed prompt-processing chunk sizes on the same SSD-streamed model workload.

use std::time::Duration;

use astronomical_inference_worker::worker_startup::run_bootstrapped_worker;
use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice,
    MAX_IPC_FRAME_BYTES, ProtocolReader, ProtocolWriter, RequestId, WorkerCommand, WorkerEvent,
};
use astronomical_supervisor::ResolvedRuntimeConfigResolver;
use tokio::time::{Instant, timeout};

const MAXIMUM_SUMMARY_TOKENS: u16 = 2_000;
fn model_id() -> &'static str {
    crate::support::large_sparse_moe_model_id()
}
const SOURCE_DOCUMENT_FIXTURE: &str =
    include_str!("../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");
const SWEEP_TIMEOUT: Duration = Duration::from_secs(115);

#[derive(Debug)]
#[allow(dead_code)]
struct PrefillChunkMetrics {
    prefill_chunk_tokens: u32,
    prompt_tokens_per_second: f64,
    generation_tokens_per_second: f64,
    prompt_token_count: u64,
    completion_token_count: u64,
    prompt_processing_seconds: f64,
    generation_seconds: f64,
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the Ornith model in-process and benchmarks 1,024-token model SSD streaming prefill chunks"]
async fn should_measure_model_ssd_streaming_with_1024_token_prefill_chunks() {
    assert_valid_prefill_chunk_sweep_result(run_prefill_chunk_sweep_with_timeout(1024).await);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the Ornith model in-process and benchmarks 2,048-token model SSD streaming prefill chunks"]
async fn should_measure_model_ssd_streaming_with_2048_token_prefill_chunks() {
    assert_valid_prefill_chunk_sweep_result(run_prefill_chunk_sweep_with_timeout(2048).await);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the Ornith model in-process and benchmarks 4,096-token model SSD streaming prefill chunks"]
async fn should_measure_model_ssd_streaming_with_4096_token_prefill_chunks() {
    assert_valid_prefill_chunk_sweep_result(run_prefill_chunk_sweep_with_timeout(4096).await);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the Ornith model in-process and benchmarks 8,192-token model SSD streaming prefill chunks"]
async fn should_measure_model_ssd_streaming_with_8192_token_prefill_chunks() {
    assert_valid_prefill_chunk_sweep_result(run_prefill_chunk_sweep_with_timeout(8192).await);
}

async fn run_prefill_chunk_sweep_with_timeout(prefill_chunk_tokens: u32) -> PrefillChunkMetrics {
    let local_task_set = tokio::task::LocalSet::new();
    timeout(
        SWEEP_TIMEOUT,
        local_task_set.run_until(run_model_with_prefill_chunk_tokens(prefill_chunk_tokens)),
    )
    .await
    .expect("the model SSD streaming prefill-chunk benchmark must finish within 115 seconds")
}

fn assert_valid_prefill_chunk_sweep_result(prefill_chunk_metrics: PrefillChunkMetrics) {
    assert!(
        prefill_chunk_metrics.prompt_token_count > 0
            && prefill_chunk_metrics.completion_token_count > 1
    );
}

/// Loads the Ornith model over duplex pipes with one neutral fixed chunk policy.
async fn run_model_with_prefill_chunk_tokens(prefill_chunk_tokens: u32) -> PrefillChunkMetrics {
    let (test_to_worker, worker_from_test) = tokio::io::duplex(MAX_IPC_FRAME_BYTES * 4);
    let (worker_to_test, test_from_worker) = tokio::io::duplex(MAX_IPC_FRAME_BYTES * 4);
    let worker_task = tokio::task::spawn_local(async move {
        run_bootstrapped_worker(worker_from_test, worker_to_test)
            .await
            .expect("the in-process worker should run successfully");
    });

    let mut protocol_writer = ProtocolWriter::new(test_to_worker);
    let mut protocol_reader = ProtocolReader::new(test_from_worker);

    let isolated_development_home = crate::support::isolated_development_home_from_user_config();
    let worker_runtime_config = ResolvedRuntimeConfigResolver::for_development_home_directory(
        isolated_development_home.path().to_path_buf(),
        std::path::PathBuf::new(),
    )
    .load()
    .expect("the prefill sweep worker configuration should resolve");
    let worker_startup_configuration = worker_runtime_config.worker_startup_configuration();
    let model_policy = worker_runtime_config
        .model_policy_catalog
        .get(model_id())
        .expect("the configured benchmark model should have a resolved policy");
    let mut worker_model_configuration = model_policy.worker_model_configuration.clone();
    worker_model_configuration
        .autoregressive_mut()
        .expect("the model SSD streaming benchmark requires an autoregressive model")
        .chunking
        .fixed_prompt_processing_chunk_size_tokens = prefill_chunk_tokens;
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
    let configured_model_directory = &model_policy.model_directory;
    protocol_writer
        .send_command(&WorkerCommand::SwapModel {
            model_directory: configured_model_directory.to_string_lossy().into_owned(),
            model_configuration: worker_model_configuration,
        })
        .await
        .expect("the benchmark should select its configured model");
    // Idle startup can leave its process-scoped configuration event queued;
    // model-swap acknowledgement remains the boundary that permits generation.
    let model_id = loop {
        let model_loaded_event = protocol_reader
            .next_event()
            .await
            .expect("the worker event stream should remain readable")
            .expect("the worker should report model loading before closing");
        match model_loaded_event {
            WorkerEvent::ModelSwapped { model_id, .. } => break model_id,
            WorkerEvent::RuntimeFeatureConfigurationApplied { .. } => {}
            unexpected_event => {
                panic!("the worker should load the benchmark model, got {unexpected_event:?}");
            }
        }
    };
    eprintln!("[model-ssd-streaming-prefill-{prefill_chunk_tokens}] worker ready: {model_id}");

    let source_document = static_source_document();
    let user_prompt = format!(
        "Summarize the following document in exactly three concise paragraphs. Do not use headings or bullet points.\n\n{source_document}"
    );
    let chat_command = ChatGenerationCommand {
        request_id: RequestId::new(1),
        model: model_id,
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
        qwen_thinking_channel_seed: None,
    };

    eprintln!("[model-ssd-streaming-prefill-{prefill_chunk_tokens}] sending generate command");
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
                        "[model-ssd-streaming-prefill-{prefill_chunk_tokens}] first output received"
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
                panic!("the worker failed the model SSD streaming prefill sweep: {reason:?}");
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
            WorkerEvent::ImageGenerationProgress { .. }
            | WorkerEvent::ImageGenerationCompleted { .. }
            | WorkerEvent::ImageGenerationFailed { .. }
            | WorkerEvent::ImageGenerationFinalized { .. } => {
                panic!("an autoregressive model SSD streaming benchmark received an image event")
            }
        }
    }
    let response_completed_at = Instant::now();

    // Each sweep cell must release its MLX process state before another fixed-size
    // cell can establish an independent measurement.
    protocol_writer
        .close()
        .await
        .expect("the benchmark protocol should close cleanly");
    worker_task
        .await
        .expect("the in-process benchmark worker should not panic");

    let first_output_at = first_output_at.expect("the benchmark should receive generated output");
    let prompt_token_count =
        prompt_token_count.expect("the terminal event should report prompt tokens");
    let completion_token_count =
        completion_token_count.expect("the terminal event should report generated tokens");
    let prompt_processing_duration = first_output_at.duration_since(request_started_at);
    let generation_duration = response_completed_at.duration_since(first_output_at);
    let prompt_tokens_per_second =
        prompt_token_count as f64 / prompt_processing_duration.as_secs_f64();
    let generation_tokens_per_second =
        completion_token_count.saturating_sub(1) as f64 / generation_duration.as_secs_f64();
    eprintln!(
        "[model-ssd-streaming-prefill-{prefill_chunk_tokens}] done: prompt={prompt_tokens_per_second:.1} tok/s, generation={generation_tokens_per_second:.1} tok/s"
    );

    PrefillChunkMetrics {
        prefill_chunk_tokens,
        prompt_tokens_per_second,
        generation_tokens_per_second,
        prompt_token_count,
        completion_token_count,
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
