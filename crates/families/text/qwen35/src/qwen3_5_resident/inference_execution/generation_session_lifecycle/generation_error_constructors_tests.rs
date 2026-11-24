use super::*;

#[test]
fn should_carry_the_exact_missing_image_pad_token_id_failure_text() {
    let engine_error = missing_image_pad_token_id_error();

    assert!(matches!(
        engine_error,
        crate::InferenceEngineError::InvalidRequest { reason } if reason == "generation request is missing the image-pad token ID"
    ));
}

#[test]
fn should_carry_the_exact_restored_prefix_range_failure_text() {
    let try_from_int_error = u32::try_from(usize::MAX).expect_err("usize maximum must exceed u32");

    let restored_prefix_error = restored_prompt_prefix_exceeds_u32_range_error(try_from_int_error);

    assert!(matches!(
        restored_prefix_error,
        crate::InferenceEngineError::Fatal { reason } if reason == "restored prompt prefix exceeds the u32 range"
    ));
}

#[test]
fn should_carry_the_exact_prefill_workspace_overflow_failure_texts() {
    let eligible_capture_error =
        eligible_astro_ssd_prompt_cache_capture_has_no_astro_ssd_prompt_cache_owner_error();
    let boundary_workspace_error = boundary_checkpoint_workspace_bytes_overflowed_error();
    let publication_workspace_error =
        astro_ssd_prompt_cache_publication_workspace_bytes_overflowed_error();

    assert!(matches!(
        eligible_capture_error,
        crate::InferenceEngineError::Fatal { reason } if reason == "eligible astro-ssd-prompt-cache capture has no astro-ssd-prompt-cache owner"
    ));
    assert!(matches!(
        boundary_workspace_error,
        crate::InferenceEngineError::Fatal { reason } if reason == "boundary checkpoint workspace bytes overflowed"
    ));
    assert!(matches!(
        publication_workspace_error,
        crate::InferenceEngineError::Fatal { reason } if reason == "astro-ssd-prompt-cache publication workspace bytes overflowed"
    ));
}

#[test]
fn should_carry_the_exact_memory_governance_failure_texts() {
    let context_reservation_error = generation_context_memory_reservation_overflowed_error();

    assert!(matches!(
        context_reservation_error,
        crate::InferenceEngineError::InvalidRequest { reason } if reason == "generation context memory reservation overflowed"
    ));
}
