use super::{PromptStretch, stop_saving_after_a_failed_save};
use crate::InferenceEngineError;
use crate::engine_request_test_support::test_engine_request;

fn token_count_from(prefill_start: usize, prefill_end: usize) -> usize {
    PromptStretch {
        prefill_start,
        prefill_end,
    }
    .token_count()
}

#[test]
fn should_count_the_tokens_a_stretch_covers() {
    assert_eq!(token_count_from(2_048, 4_096), 2_048);
}

#[test]
fn should_count_one_token_for_a_single_token_tail() {
    assert_eq!(token_count_from(10_010, 10_011), 1);
}

#[test]
fn should_count_no_tokens_for_an_empty_stretch() {
    assert_eq!(token_count_from(512, 512), 0);
}

#[test]
fn should_stop_saving_but_keep_the_request_when_a_save_is_refused() {
    let mut active_request = test_engine_request(());
    active_request.can_use_astro_ssd_prompt_cache = true;

    let outcome = stop_saving_after_a_failed_save(
        &mut active_request,
        Err(InferenceEngineError::InvalidRequest {
            reason: "astro SSD prompt cache failed during required persistent prompt-state capture"
                .to_owned(),
        }),
    );

    assert!(outcome.is_ok());
    assert!(!active_request.can_use_astro_ssd_prompt_cache);
}

#[test]
fn should_keep_saving_after_a_save_that_succeeded() {
    let mut active_request = test_engine_request(());
    active_request.can_use_astro_ssd_prompt_cache = true;

    let outcome = stop_saving_after_a_failed_save(&mut active_request, Ok(()));

    assert!(outcome.is_ok());
    assert!(active_request.can_use_astro_ssd_prompt_cache);
}

#[test]
fn should_still_stop_the_request_when_the_failure_is_not_a_refused_save() {
    let mut active_request = test_engine_request(());
    active_request.can_use_astro_ssd_prompt_cache = true;

    let outcome = stop_saving_after_a_failed_save(
        &mut active_request,
        Err(InferenceEngineError::Fatal {
            reason: "the GPU failed".to_owned(),
        }),
    );

    assert!(matches!(outcome, Err(InferenceEngineError::Fatal { .. })));
    assert!(active_request.can_use_astro_ssd_prompt_cache);
}
