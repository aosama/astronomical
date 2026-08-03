use super::*;

#[test]
fn should_ignore_partial_prefill_chuncks_when_learning_candidate_quality() {
    let mut prefill_chunck_size_optimizer = three_candidate_optimizer();
    let prompt_processing_context = PrefillChunckSizeOptimizerContext::new(47);

    // Trust all candidates so exploitation can pick the best.
    // 256 at 2000ms is slow (throughput ~128 t/s).
    record_full_observations(
        &mut prefill_chunck_size_optimizer,
        prompt_processing_context,
        256,
        &[2_000, 2_000, 2_000],
    );
    // 512 at 600ms is fast (throughput ~853 t/s).
    record_full_observations(
        &mut prefill_chunck_size_optimizer,
        prompt_processing_context,
        512,
        &[600, 600, 600],
    );
    // 1024 at 11_000ms is very slow (throughput ~93 t/s) — but this is before the partial.
    record_full_observations(
        &mut prefill_chunck_size_optimizer,
        prompt_processing_context,
        1_024,
        &[11_000, 11_000, 11_000],
    );

    // Partial chunks must not count toward trust or quality.
    prefill_chunck_size_optimizer
        .tell(
            prompt_processing_context,
            1_024,
            PrefillChunckSizeOptimizerObservation::partial_prefill_chunck(4, 100),
        )
        .expect("partial observation should be accepted but ignored");

    assert_eq!(
        ask_candidate_prefill_chunck_tokens(
            &mut prefill_chunck_size_optimizer,
            prompt_processing_context
        ),
        512,
        "partial chunks should not pollute candidate quality learning"
    );
}
