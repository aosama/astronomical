use std::{path::Path, time::Instant};

use model_serving::{
    AstroSsdPromptCacheDiskStoreConfig, InferenceEngine, PerformanceAttribution,
    PerformanceAttributionLog, PromptProcessingChunkSizer, Qwen3_5ArtifactValidator,
    Qwen3_5StreamingEngine, TextMlxEngineConstructors,
};
use tokio::time::{MissedTickBehavior, interval};

use super::{IMAGE_PAD_TOKEN_ID, PROGRESS_INTERVAL};

pub(crate) fn create_attributed_engine(
    model_directory: &Path,
    performance_attribution_log_path: &Path,
    mlx_memory_limits: &runtime_integration::MlxMemoryLimits,
    fixed_prompt_processing_chunk_size_tokens: u32,
) -> (Qwen3_5StreamingEngine, Vec<u32>) {
    create_attributed_engine_with_ssd_streaming_prefill(
        model_directory,
        performance_attribution_log_path,
        mlx_memory_limits,
        fixed_prompt_processing_chunk_size_tokens,
    )
}

pub(crate) fn create_attributed_engine_with_ssd_streaming_prefill(
    model_directory: &Path,
    performance_attribution_log_path: &Path,
    mlx_memory_limits: &runtime_integration::MlxMemoryLimits,
    fixed_ssd_streaming_prompt_processing_chunk_size_tokens: u32,
) -> (Qwen3_5StreamingEngine, Vec<u32>) {
    create_attributed_engine_keeping_prompts(
        model_directory,
        performance_attribution_log_path,
        mlx_memory_limits,
        (
            fixed_ssd_streaming_prompt_processing_chunk_size_tokens,
            StreamingJourneyOptions::default(),
        ),
    )
}

/// What a journey may ask of the streaming engine beyond the test run's
/// configuration; the default asks nothing.
#[derive(Default)]
pub(crate) struct StreamingJourneyOptions {
    /// An `astro_ssd_prompt_cache` store, so a conversation's next turn can
    /// restore what the last one read.
    pub(crate) astro_ssd_prompt_cache_store: Option<AstroSsdPromptCacheDiskStoreConfig>,
    /// Turns on the opt-in Neural Engine prompt reading (issue #2585).
    pub(crate) reads_prompts_on_the_neural_engine: bool,
}

/// The streaming engine with a journey's options.
fn create_attributed_engine_keeping_prompts(
    model_directory: &Path,
    performance_attribution_log_path: &Path,
    mlx_memory_limits: &runtime_integration::MlxMemoryLimits,
    (fixed_ssd_streaming_prompt_processing_chunk_size_tokens, journey_options): (
        u32,
        StreamingJourneyOptions,
    ),
) -> (Qwen3_5StreamingEngine, Vec<u32>) {
    let mut worker_chunking = crate::common::standard_worker_chunking_configuration();
    worker_chunking.experimental_neural_engine_prefill_enabled =
        journey_options.reads_prompts_on_the_neural_engine;
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, 20_480)
        .expect("the configured Ornith artifact should validate before benchmark loading");
    let end_of_sequence_token_ids = validated_artifact
        .config()
        .end_of_sequence_token_ids()
        .to_vec();
    let performance_attribution_log =
        PerformanceAttributionLog::open(performance_attribution_log_path, true)
            .expect("the benchmark should open its JSON Lines log");
    let qwen3_5_engine =
        Qwen3_5StreamingEngine::new_with_runtime_chunking_and_performance_attribution(
            validated_artifact,
            mlx_memory_limits.active_memory_limit_bytes(),
            mlx_memory_limits.mlx_buffer_cache_limit_bytes(),
            journey_options.astro_ssd_prompt_cache_store,
            PromptProcessingChunkSizer::for_chunk_size_tokens(
                fixed_ssd_streaming_prompt_processing_chunk_size_tokens,
            )
            .expect("the benchmark streaming prefill chunk size should be valid"),
            IMAGE_PAD_TOKEN_ID,
            model_directory.to_path_buf(),
            worker_chunking,
            PerformanceAttribution::enabled(),
            performance_attribution_log,
        )
        .expect("the benchmark engine settings should be valid");
    (qwen3_5_engine, end_of_sequence_token_ids)
}

pub(crate) async fn load_engine_with_progress(
    qwen3_5_engine: &mut impl InferenceEngine,
    phase_name: &str,
) -> Option<ipc_protocol::WorkerEvent> {
    let phase_started_at = Instant::now();
    let mut progress_interval = interval(PROGRESS_INTERVAL);
    progress_interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    progress_interval.tick().await;
    let load_future = qwen3_5_engine.load();
    tokio::pin!(load_future);
    loop {
        tokio::select! {
            load_outcome = &mut load_future => {
                // What the load says about Neural Engine prompt reading,
                // when the setting is on.
                return load_outcome
                    .expect("the benchmark engine should load")
                    .neural_engine_prefill_event();
            }
            _ = progress_interval.tick() => eprintln!("[performance-attribution] status=progress phase={phase_name} elapsed_seconds={:.1} ETA_seconds=unknown", phase_started_at.elapsed().as_secs_f64()),
        }
    }
}

/// A streaming engine loaded as the test run's configuration names it:
/// `model_directory` and `paging_experiment`'s MLX limit and prompt chunk.
pub(crate) struct PagingExperimentEngine {
    /// The load's Neural Engine prompt-reading event; `None` with it off.
    pub(crate) neural_engine_prefill_at_load: Option<ipc_protocol::WorkerEvent>,
    pub(crate) engine: Qwen3_5StreamingEngine,
    pub(crate) end_of_sequence_token_ids: Vec<u32>,
    pub(crate) model_directory: std::path::PathBuf,
    /// Holds `attribution.jsonl`, where attributed requests are reported.
    pub(crate) attribution_directory: tempfile::TempDir,
}

/// Loads the configured model on the streaming engine, reporting progress
/// under `load_phase_name`.
pub(crate) async fn load_paging_experiment_engine(load_phase_name: &str) -> PagingExperimentEngine {
    load_paging_experiment_engine_with(load_phase_name, StreamingJourneyOptions::default()).await
}

/// The same engine with a journey's options.
pub(crate) async fn load_paging_experiment_engine_with(
    load_phase_name: &str,
    journey_options: StreamingJourneyOptions,
) -> PagingExperimentEngine {
    let test_run_configuration = test_harness::test_run::TestRunContext::current().configuration;
    let model_directory = test_run_configuration
        .model_directory
        .expect("model_directory names the sparse model");
    let paging_experiment = test_run_configuration.paging_experiment;
    let mlx_limit_bytes = usize::try_from(
        paging_experiment
            .maximum_mlx_memory_bytes()
            .expect("paging_experiment sets a valid MLX limit"),
    )
    .expect("the MLX limit should fit usize");
    let attribution_directory =
        tempfile::tempdir().expect("the journey should create an attribution directory");
    let mlx_memory_limits =
        runtime_integration::MlxMemoryLimits::new(mlx_limit_bytes, mlx_limit_bytes)
            .expect("the configured limit should be a valid MLX limit");
    let (mut engine, end_of_sequence_token_ids) = create_attributed_engine_keeping_prompts(
        &model_directory,
        &attribution_directory.path().join("attribution.jsonl"),
        &mlx_memory_limits,
        (
            paging_experiment.streaming_prompt_processing_chunk_tokens,
            journey_options,
        ),
    );
    let neural_engine_prefill_at_load =
        load_engine_with_progress(&mut engine, load_phase_name).await;
    PagingExperimentEngine {
        neural_engine_prefill_at_load,
        engine,
        end_of_sequence_token_ids,
        model_directory,
        attribution_directory,
    }
}
