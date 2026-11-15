//! What it costs the streaming engine to run one prompt span again: the
//! parity journey fails a span on purpose, the way MLX reports a span it
//! could not fit, and these helpers count what the engine did about it
//! (issue #2221).

use astronomical_model_serving::Qwen3_5StreamingEngine;

/// What the engine had counted and kept before a span was failed on purpose.
pub(super) struct RedoBaseline {
    overflow_count: u64,
    redo_attempt_count: u64,
    kept_expert_bytes: u64,
}

/// What failing one span cost: steps that hit the limit, redo attempts, and the
/// experts kept before and after.
pub(super) struct RedoCost {
    pub(super) overflow_count: u64,
    pub(super) redo_attempt_count: u64,
    pub(super) kept_expert_bytes_before: u64,
    pub(super) kept_expert_bytes_after: u64,
}

async fn kept_expert_bytes(qwen3_5_engine: &Qwen3_5StreamingEngine) -> u64 {
    qwen3_5_engine
        .expert_weight_memory_cache_statistics_for_tests()
        .await
        .expect("the loaded model should report its astro expert cache")
        .resident_payload_byte_count
}

/// Has the next layer-at-a-time prompt fail `span_to_fail_once` once.
pub(super) async fn arm_span_failure(
    qwen3_5_engine: &Qwen3_5StreamingEngine,
    span_to_fail_once: (usize, usize),
) -> RedoBaseline {
    let (overflow_count, redo_attempt_count) = qwen3_5_engine
        .memory_redo_counts_for_tests()
        .await
        .expect("the loaded model should count its redos");
    let kept_expert_bytes = kept_expert_bytes(qwen3_5_engine).await;
    qwen3_5_engine
        .fail_prompt_span_once_for_tests(span_to_fail_once.0, span_to_fail_once.1)
        .await
        .expect("the engine should accept the span to fail");
    RedoBaseline {
        overflow_count,
        redo_attempt_count,
        kept_expert_bytes,
    }
}

pub(super) async fn redo_cost_since(
    qwen3_5_engine: &Qwen3_5StreamingEngine,
    redo_baseline: RedoBaseline,
) -> RedoCost {
    let (overflow_count, redo_attempt_count) = qwen3_5_engine
        .memory_redo_counts_for_tests()
        .await
        .expect("the loaded model should count its redos");
    RedoCost {
        overflow_count: overflow_count - redo_baseline.overflow_count,
        redo_attempt_count: redo_attempt_count - redo_baseline.redo_attempt_count,
        kept_expert_bytes_before: redo_baseline.kept_expert_bytes,
        kept_expert_bytes_after: kept_expert_bytes(qwen3_5_engine).await,
    }
}

/// One failed span is one step over the limit, redone once, and the cache gives back
/// only the room the redo leased: at least four fifths of what it kept stays.
pub(super) fn assert_one_span_was_redone_cheaply(redo_cost: &RedoCost, depth_name: &str) {
    assert_eq!(
        (redo_cost.overflow_count, redo_cost.redo_attempt_count),
        (1, 1),
        "one failed span must be one step over the limit and one redo at depth {depth_name}"
    );
    assert!(
        redo_cost.kept_expert_bytes_after.saturating_mul(5)
            >= redo_cost.kept_expert_bytes_before.saturating_mul(4),
        "a redone span must leave at least 80% of the kept experts at depth {depth_name}: before={} after={}",
        redo_cost.kept_expert_bytes_before,
        redo_cost.kept_expert_bytes_after
    );
}
