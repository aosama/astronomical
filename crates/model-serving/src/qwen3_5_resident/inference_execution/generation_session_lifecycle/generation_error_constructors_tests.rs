use super::*;

#[test]
fn should_carry_the_exact_engine_lost_loaded_model_failure_text() {
    let engine_error = engine_lost_loaded_model_error();

    assert!(matches!(
        engine_error,
        crate::InferenceEngineError::Fatal { reason } if reason == "Qwen3.5 engine lost its loaded model"
    ));
}

#[test]
fn should_carry_the_exact_missing_image_pad_token_id_failure_text() {
    let engine_error = missing_image_pad_token_id_error();

    assert!(matches!(
        engine_error,
        crate::InferenceEngineError::InvalidRequest { reason } if reason == "generation request is missing the image-pad token ID"
    ));
}

#[test]
fn should_carry_the_exact_visual_embedding_model_contract_not_loaded_failure_text() {
    let engine_error = visual_embedding_model_contract_not_loaded_error();

    assert!(matches!(
        engine_error,
        crate::InferenceEngineError::Fatal { reason } if reason == "Qwen3.5 persistent visual embedding model contract is not loaded"
    ));
}

#[test]
fn should_carry_the_exact_thinking_budget_reservation_overflow_failure_text() {
    let engine_error = thinking_budget_output_reservation_overflow_error();

    assert!(matches!(
        engine_error,
        crate::InferenceEngineError::InvalidRequest { reason } if reason == "thinking-budget output reservation overflowed"
    ));
}

#[test]
fn should_carry_the_exact_generation_context_overflow_failure_text() {
    let engine_error = generation_context_token_count_overflow_error();

    assert!(matches!(
        engine_error,
        crate::InferenceEngineError::InvalidRequest { reason } if reason == "generation context token count overflowed"
    ));
}

#[test]
fn should_carry_the_exact_mlx_memory_range_failure_texts() {
    // Any genuinely failing integer conversion yields the one TryFromIntError.
    let try_from_int_error = u32::try_from(usize::MAX).expect_err("usize maximum must exceed u32");
    let active_memory_error = mlx_active_memory_bytes_exceed_u64_range_error(try_from_int_error);
    let mlx_buffer_cache_error = mlx_buffer_cache_bytes_exceed_u64_range_error(try_from_int_error);
    let peak_memory_error = mlx_peak_memory_bytes_exceed_u64_range_error(try_from_int_error);

    assert!(matches!(
        active_memory_error,
        crate::InferenceEngineError::Fatal { reason } if reason == "MLX active memory bytes exceed the u64 range"
    ));
    assert!(matches!(
        mlx_buffer_cache_error,
        crate::InferenceEngineError::Fatal { reason } if reason == "MLX buffer cache memory bytes exceed the u64 range"
    ));
    assert!(matches!(
        peak_memory_error,
        crate::InferenceEngineError::Fatal { reason } if reason == "MLX peak memory bytes exceed the u64 range"
    ));
}

#[test]
fn should_carry_the_exact_visual_embedding_and_restored_prefix_range_failure_texts() {
    let try_from_int_error = u32::try_from(usize::MAX).expect_err("usize maximum must exceed u32");

    let row_count_error = visual_embedding_row_count_exceeds_i32_range_error(try_from_int_error);
    let hidden_size_error =
        visual_embedding_hidden_size_exceeds_i32_range_error(try_from_int_error);
    let restored_prefix_error = restored_prompt_prefix_exceeds_u32_range_error(try_from_int_error);

    assert!(matches!(
        row_count_error,
        crate::InferenceEngineError::Fatal { reason } if reason == "visual embedding row count exceeds the i32 range"
    ));
    assert!(matches!(
        hidden_size_error,
        crate::InferenceEngineError::Fatal { reason } if reason == "visual embedding hidden size exceeds the i32 range"
    ));
    assert!(matches!(
        restored_prefix_error,
        crate::InferenceEngineError::Fatal { reason } if reason == "restored prompt prefix exceeds the u32 range"
    ));
}

#[test]
fn should_carry_the_exact_prefill_workspace_overflow_failure_texts() {
    let eligible_capture_error =
        eligible_astro_ssd_prompt_cache_capture_has_no_persistent_cache_owner_error();
    let boundary_workspace_error = boundary_checkpoint_workspace_bytes_overflowed_error();
    let publication_workspace_error =
        astro_ssd_prompt_cache_publication_workspace_bytes_overflowed_error();
    let completed_chunk_tokens_error = completed_prefill_chunk_tokens_exceed_u32_range_error(
        u32::try_from(usize::MAX).expect_err("usize maximum must exceed u32"),
    );

    assert!(matches!(
        eligible_capture_error,
        crate::InferenceEngineError::Fatal { reason } if reason == "eligible astro-ssd-prompt-cache capture has no persistent cache owner"
    ));
    assert!(matches!(
        boundary_workspace_error,
        crate::InferenceEngineError::Fatal { reason } if reason == "boundary checkpoint workspace bytes overflowed"
    ));
    assert!(matches!(
        publication_workspace_error,
        crate::InferenceEngineError::Fatal { reason } if reason == "astro-ssd-prompt-cache publication workspace bytes overflowed"
    ));
    assert!(matches!(
        completed_chunk_tokens_error,
        crate::InferenceEngineError::Fatal { reason } if reason == "completed_prefill_chunk_tokens exceeds the u32 range"
    ));
}

#[test]
fn should_carry_the_exact_generated_token_counter_overflow_failure_text() {
    let engine_error = generated_token_counter_overflowed_error();

    assert!(matches!(
        engine_error,
        crate::InferenceEngineError::Fatal { reason } if reason == "generated-token counter overflowed"
    ));
}

#[test]
fn should_carry_the_exact_construction_failure_texts() {
    let try_from_int_error = u32::try_from(usize::MAX).expect_err("usize maximum must exceed u32");
    let growth_tokens_error =
        full_attention_growth_tokens_exceed_i32_range_error(try_from_int_error);
    let context_reservation_error = context_memory_reservation_overflowed_error();

    assert!(matches!(
        growth_tokens_error,
        crate::InferenceEngineError::Fatal { reason } if reason == "full-attention growth tokens exceed Int32 range"
    ));
    assert!(matches!(
        context_reservation_error,
        crate::InferenceEngineError::Fatal { reason } if reason == "Qwen3.5 context memory reservation overflowed"
    ));
}

#[test]
fn should_carry_the_exact_memory_governance_failure_texts() {
    let try_from_int_error = u32::try_from(usize::MAX).expect_err("usize maximum must exceed u32");

    let context_reservation_error = generation_context_memory_reservation_overflowed_error();
    let prefill_workspace_error =
        prefill_activation_workspace_exceeds_platform_range_error(try_from_int_error);
    let context_workspace_error = generation_context_workspace_reservation_overflowed_error();
    let ceiling_range_error =
        current_mlx_memory_ceiling_exceeds_u64_range_error(try_from_int_error);
    let active_memory_error = mlx_active_memory_exceeds_u64_range_error(try_from_int_error);
    let mlx_buffer_cache_error = mlx_buffer_cache_exceeds_u64_range_error(try_from_int_error);
    let peak_memory_error = mlx_peak_memory_exceeds_u64_range_error(try_from_int_error);

    assert!(matches!(
        context_reservation_error,
        crate::InferenceEngineError::InvalidRequest { reason } if reason == "generation context memory reservation overflowed"
    ));
    assert!(matches!(
        prefill_workspace_error,
        crate::InferenceEngineError::InvalidRequest { reason } if reason == "prefill activation workspace exceeds the platform range"
    ));
    assert!(matches!(
        context_workspace_error,
        crate::InferenceEngineError::InvalidRequest { reason } if reason == "generation context workspace reservation overflowed"
    ));
    assert!(matches!(
        ceiling_range_error,
        crate::InferenceEngineError::Fatal { reason } if reason == "current MLX memory ceiling exceeds the u64 range"
    ));
    assert!(matches!(
        active_memory_error,
        crate::InferenceEngineError::Fatal { reason } if reason == "MLX active memory exceeds the u64 range"
    ));
    assert!(matches!(
        mlx_buffer_cache_error,
        crate::InferenceEngineError::Fatal { reason } if reason == "MLX buffer cache memory exceeds the u64 range"
    ));
    assert!(matches!(
        peak_memory_error,
        crate::InferenceEngineError::Fatal { reason } if reason == "MLX peak memory exceeds the u64 range"
    ));
}
