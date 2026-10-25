use astronomical_ipc_protocol::WorkerPromptProcessingPhase;

use super::{can_use_persistent_prompt_cache, initial_prompt_processing_phase};

#[test]
fn should_enable_the_persistent_prompt_cache_for_a_text_request_when_it_is_available() {
    assert!(can_use_persistent_prompt_cache(true, false));
}

#[test]
fn should_disable_the_persistent_prompt_cache_for_precomputed_visual_embeddings() {
    assert!(!can_use_persistent_prompt_cache(true, true));
}

#[test]
fn should_disable_the_persistent_prompt_cache_when_no_store_is_installed() {
    assert!(!can_use_persistent_prompt_cache(false, false));
}

#[test]
fn should_keep_the_persistent_prompt_cache_disabled_when_both_conditions_block_it() {
    assert!(!can_use_persistent_prompt_cache(false, true));
}

#[test]
fn should_select_the_target_prefill_phase_at_the_cold_start_cursor() {
    assert_eq!(
        initial_prompt_processing_phase(0, 512),
        Some(WorkerPromptProcessingPhase::Target)
    );
}

#[test]
fn should_select_the_target_prefill_phase_for_the_last_unprocessed_prompt_token() {
    assert_eq!(
        initial_prompt_processing_phase(511, 512),
        Some(WorkerPromptProcessingPhase::Target)
    );
}

#[test]
fn should_skip_prefill_when_the_restored_prefix_covers_the_whole_prompt() {
    assert_eq!(initial_prompt_processing_phase(512, 512), None);
}

#[test]
fn should_skip_prefill_for_an_empty_prompt_at_a_zero_cursor() {
    assert_eq!(initial_prompt_processing_phase(0, 0), None);
}
