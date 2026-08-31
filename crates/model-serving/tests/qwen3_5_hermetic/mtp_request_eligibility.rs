use astronomical_model_serving::qwen3_5_mtp_request_is_eligible;

fn opted_in_resident_text_request_without_prompt_cache() -> bool {
    qwen3_5_mtp_request_is_eligible(true, true, true, false, false, false, false, 32, 0)
}

#[test]
fn should_allow_opted_in_resident_text_without_prompt_cache() {
    assert!(opted_in_resident_text_request_without_prompt_cache());
}

#[test]
fn should_keep_ssd_paged_sparse_experts_target_only() {
    assert!(!qwen3_5_mtp_request_is_eligible(
        true, true, true, false, false, false, true, 32, 0,
    ));
}

#[test]
fn should_prefer_persistent_prompt_cache_over_multi_token_prediction() {
    assert!(!qwen3_5_mtp_request_is_eligible(
        true, true, true, false, false, true, false, 32, 0,
    ));
}

#[test]
fn should_keep_visual_inputs_target_only() {
    assert!(!qwen3_5_mtp_request_is_eligible(
        true, true, true, false, true, false, false, 32, 0,
    ));
    assert!(!qwen3_5_mtp_request_is_eligible(
        true, true, true, false, false, true, false, 32, 0,
    ));
}
