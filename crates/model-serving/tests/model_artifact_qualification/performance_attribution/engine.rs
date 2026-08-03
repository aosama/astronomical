use std::{path::Path, time::Instant};

use astronomical_model_serving::{
    DEFAULT_FULL_ATTENTION_KV_STATE_GROWTH_TOKENS, InferenceEngine, PerformanceAttribution,
    PerformanceAttributionLog, Qwen3_5MoEArtifactValidator, Qwen3_5MoEEngine,
    Qwen3_5MoEPrefillChunckSizer,
};
use tokio::time::{MissedTickBehavior, interval};

use super::{IMAGE_PAD_TOKEN_ID, PROGRESS_INTERVAL};

pub(crate) fn create_attributed_engine(
    model_directory: &Path,
    performance_attribution_log_path: &Path,
    mlx_memory_limits: &astronomical_runtime_integration::MlxMemoryLimits,
    fixed_prefill_chunck_tokens: u32,
) -> (Qwen3_5MoEEngine, Vec<u32>) {
    let validated_artifact = Qwen3_5MoEArtifactValidator::new()
        .validate(model_directory, 20_480)
        .expect("the Qwen3.6 OptiQ artifact should validate before benchmark loading");
    let end_of_sequence_token_ids = validated_artifact
        .config()
        .end_of_sequence_token_ids()
        .to_vec();
    let performance_attribution_log =
        PerformanceAttributionLog::open(performance_attribution_log_path, true)
            .expect("the benchmark should open its JSON Lines log");
    let qwen3_5_moe_engine =
        Qwen3_5MoEEngine::new_with_prefill_chunck_sizer_and_performance_attribution(
            validated_artifact,
            mlx_memory_limits.active_memory_limit_bytes(),
            mlx_memory_limits.allocator_cache_memory_limit_bytes(),
            None,
            Qwen3_5MoEPrefillChunckSizer::for_fixed_prefill_chunck_tokens(
                fixed_prefill_chunck_tokens,
            )
            .expect("the benchmark prefill chunck size should be valid"),
            IMAGE_PAD_TOKEN_ID,
            model_directory.to_path_buf(),
            DEFAULT_FULL_ATTENTION_KV_STATE_GROWTH_TOKENS,
            true,
            false,
            PerformanceAttribution::enabled(),
            performance_attribution_log,
        )
        .expect("the benchmark engine settings should be valid");
    (qwen3_5_moe_engine, end_of_sequence_token_ids)
}

pub(crate) async fn load_engine_with_progress(
    qwen3_5_moe_engine: &mut Qwen3_5MoEEngine,
    phase_name: &str,
) {
    let phase_started_at = Instant::now();
    let mut progress_interval = interval(PROGRESS_INTERVAL);
    progress_interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    progress_interval.tick().await;
    let load_future = qwen3_5_moe_engine.load();
    tokio::pin!(load_future);
    loop {
        tokio::select! {
            load_outcome = &mut load_future => { load_outcome.expect("the benchmark engine should load"); return; }
            _ = progress_interval.tick() => eprintln!("[performance-attribution] status=progress phase={phase_name} elapsed_seconds={:.1} ETA_seconds=unknown", phase_started_at.elapsed().as_secs_f64()),
        }
    }
}
