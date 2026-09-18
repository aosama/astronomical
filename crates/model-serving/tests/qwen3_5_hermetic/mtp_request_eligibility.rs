use astronomical_model_serving::qwen3_5_mtp_request_is_eligible;

fn opted_in_resident_text_request_without_prompt_cache() -> bool {
    qwen3_5_mtp_request_is_eligible(true, true, true, false, false, 0, 32, false)
}

#[test]
fn should_allow_opted_in_resident_text_without_prompt_cache() {
    assert!(opted_in_resident_text_request_without_prompt_cache());
}

#[test]
fn should_keep_ssd_paged_sparse_experts_target_only() {
    assert!(!qwen3_5_mtp_request_is_eligible(
        true, true, true, false, false, 0, 32, true,
    ));
}

// Supersedes the former "availability always wins" pin. The defect surfaced
// through the production A/B run: config with `prompt_cache.enabled` plus
// `acceleration.mtp.enabled` never engaged the MTP head because mere cache
// *availability* gated it off. Prefill cache reuse and the decode-side head
// are orthogonal, and the restored-token-count condition below still
// disqualifies a restore that replaced prompt work.
#[test]
fn should_run_the_mtp_head_alongside_an_available_persistent_prompt_cache() {
    assert!(qwen3_5_mtp_request_is_eligible(
        true, true, true, false, false, 0, 32, false,
    ));
}

#[test]
fn should_forfeit_the_mtp_head_when_a_cache_restore_replaced_prompt_work() {
    assert!(!qwen3_5_mtp_request_is_eligible(
        true, true, true, false, false, 31, 32, false,
    ));
}

#[test]
fn should_keep_visual_inputs_target_only() {
    assert!(!qwen3_5_mtp_request_is_eligible(
        true, true, true, false, true, 0, 32, false,
    ));
    assert!(!qwen3_5_mtp_request_is_eligible(
        true, true, true, true, false, 0, 32, false,
    ));
}
