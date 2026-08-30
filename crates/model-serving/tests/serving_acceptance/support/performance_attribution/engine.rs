use std::{path::Path, time::Instant};

use astronomical_model_serving::{
    InferenceEngine, PerformanceAttribution, PerformanceAttributionLog, Qwen3_5ArtifactValidator,
    Qwen3_5Engine, Qwen3_5PromptProcessingChunkSizer,
};
use tokio::time::{MissedTickBehavior, interval};

use super::{IMAGE_PAD_TOKEN_ID, PROGRESS_INTERVAL};

pub(crate) fn create_attributed_engine(
    model_directory: &Path,
    performance_attribution_log_path: &Path,
    mlx_memory_limits: &astronomical_runtime_integration::MlxMemoryLimits,
    fixed_prompt_processing_chunk_size_tokens: u32,
) -> (Qwen3_5Engine, Vec<u32>) {
    create_attributed_engine_with_ssd_streaming_prefill(
        model_directory,
        performance_attribution_log_path,
        mlx_memory_limits,
        fixed_prompt_processing_chunk_size_tokens,
        fixed_prompt_processing_chunk_size_tokens,
    )
}

pub(crate) fn create_attributed_engine_with_ssd_streaming_prefill(
    model_directory: &Path,
    performance_attribution_log_path: &Path,
    mlx_memory_limits: &astronomical_runtime_integration::MlxMemoryLimits,
    fixed_prompt_processing_chunk_size_tokens: u32,
    fixed_ssd_streaming_prompt_processing_chunk_size_tokens: u32,
) -> (Qwen3_5Engine, Vec<u32>) {
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
    let qwen3_5_engine = Qwen3_5Engine::new_with_runtime_chunking_and_speculative_prefill_and_performance_attribution(
        validated_artifact,
        mlx_memory_limits.active_memory_limit_bytes(),
        mlx_memory_limits.allocator_cache_memory_limit_bytes(),
        None,
        Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens_with_ssd_streaming(
            fixed_prompt_processing_chunk_size_tokens,
            fixed_ssd_streaming_prompt_processing_chunk_size_tokens,
        )
        .expect("the benchmark prefill chunck size should be valid"),
        IMAGE_PAD_TOKEN_ID,
        model_directory.to_path_buf(),
        crate::common::standard_worker_chunking_configuration(),
        true,
        false,
        crate::common::disabled_worker_speculative_prefill_configuration(),
        PerformanceAttribution::enabled(),
        performance_attribution_log,
    )
    .expect("the benchmark engine settings should be valid");
    (qwen3_5_engine, end_of_sequence_token_ids)
}

pub(crate) async fn load_engine_with_progress(
    qwen3_5_engine: &mut Qwen3_5Engine,
    phase_name: &str,
) {
    let phase_started_at = Instant::now();
    let mut progress_interval = interval(PROGRESS_INTERVAL);
    progress_interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    progress_interval.tick().await;
    let load_future = qwen3_5_engine.load();
    tokio::pin!(load_future);
    loop {
        tokio::select! {
            load_outcome = &mut load_future => { load_outcome.expect("the benchmark engine should load"); return; }
            _ = progress_interval.tick() => eprintln!("[performance-attribution] status=progress phase={phase_name} elapsed_seconds={:.1} ETA_seconds=unknown", phase_started_at.elapsed().as_secs_f64()),
        }
    }
}
