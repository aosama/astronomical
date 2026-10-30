//! Beside-source tests for the pure chunked-prefill units.

use super::*;

#[test]
fn should_clamp_the_first_chunk_at_the_first_block_boundary_it_would_cross() {
    let chunk_length = super::prefill_chunk_length(0, 100, 100, Some(32));
    assert_eq!(chunk_length, 32);
}

#[test]
fn should_anchor_a_later_chunk_on_the_block_boundary_after_its_start() {
    // One block (32 tokens) is already processed; the next forward may run
    // to the following boundary at 64 (32 tokens), never past it.
    assert_eq!(super::prefill_chunk_length(32, 100, 100, Some(32)), 32);
    // A start one token before that boundary may only consume that token.
    assert_eq!(super::prefill_chunk_length(31, 100, 100, Some(32)), 1);
}

#[test]
fn should_keep_the_requested_chunk_end_without_a_persistent_prompt_cache() {
    assert_eq!(super::prefill_chunk_length(0, 100, 100, None), 100);
}

#[test]
fn should_stop_the_chunk_short_when_the_remaining_prompt_runs_out_first() {
    // Ten tokens are requested but only three remain, so the chunk ends at
    // the prompt end even though the block boundary is far beyond it.
    assert_eq!(super::prefill_chunk_length(7, 3, 100, Some(32)), 3);
}

#[test]
fn should_produce_an_empty_chunk_when_no_prompt_tokens_remain() {
    assert_eq!(super::prefill_chunk_length(64, 0, 100, Some(32)), 0);
}

#[test]
fn should_produce_an_empty_chunk_when_the_chunk_budget_is_zero() {
    assert_eq!(super::prefill_chunk_length(7, 100, 0, None), 0);
}

#[test]
fn should_report_only_the_chunk_own_token_count_as_progress() {
    let generated_token = super::prefill_progress_token(
        32,
        7,
        None,
        WorkerPromptWorkReuse {
            target_eligible_token_count: 128,
            target_restored_token_count: 64,
        },
    );
    assert_eq!(
        generated_token,
        GeneratedToken::PrefillProgress {
            processed_token_count: 32,
            elapsed_millis: 7,
            forward_prefill_chunk_elapsed_millis: 7,
            completed_prefill_chunk_tokens: 32,
            mlx_memory_telemetry: None,
            expert_residency_telemetry: None,
            expert_memory_mode: Some(ExpertMemoryMode::Resident),
            prompt_work_reuse: WorkerPromptWorkReuse {
                target_eligible_token_count: 128,
                target_restored_token_count: 64,
            },
            persistent_prompt_cache_diagnostics: None,
        }
    );
}

#[test]
fn should_forward_the_sampled_memory_telemetry_into_the_progress_token() {
    // The telemetry is opaque here, so the caller's exact value — including
    // `None` — must reach the progress event untouched.
    let generated_token = super::prefill_progress_token(
        1,
        0,
        None,
        WorkerPromptWorkReuse {
            target_eligible_token_count: 0,
            target_restored_token_count: 0,
        },
    );
    let GeneratedToken::PrefillProgress {
        mlx_memory_telemetry,
        prompt_work_reuse,
        ..
    } = generated_token
    else {
        panic!("a prefill chunk must report PrefillProgress");
    };
    assert_eq!(mlx_memory_telemetry, None);
    assert_eq!(prompt_work_reuse.target_eligible_token_count, 0);
    assert_eq!(prompt_work_reuse.target_restored_token_count, 0);
}

#[test]
fn should_build_a_zero_length_progress_token_at_the_prompt_end() {
    let generated_token = super::prefill_progress_token(
        0,
        0,
        None,
        WorkerPromptWorkReuse {
            target_eligible_token_count: 0,
            target_restored_token_count: 0,
        },
    );
    let GeneratedToken::PrefillProgress {
        processed_token_count,
        elapsed_millis,
        forward_prefill_chunk_elapsed_millis,
        completed_prefill_chunk_tokens,
        ..
    } = generated_token
    else {
        panic!("a prefill chunk must report PrefillProgress");
    };
    assert_eq!(processed_token_count, 0);
    assert_eq!(elapsed_millis, 0);
    assert_eq!(forward_prefill_chunk_elapsed_millis, 0);
    assert_eq!(completed_prefill_chunk_tokens, 0);
}
