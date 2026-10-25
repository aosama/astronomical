use super::*;

#[test]
fn should_carry_the_exact_engine_lost_loaded_model_invalid_input_description() {
    let execution_error = engine_lost_loaded_model_invalid_input();

    assert!(matches!(
        execution_error,
        crate::Qwen3_5ExecutionError::InvalidInput { description } if description == "Qwen3.5 engine lost its loaded model"
    ));
}

#[test]
fn should_carry_the_exact_visual_block_contract_invalid_input_description() {
    let execution_error = visual_prompt_cache_checkpoints_have_no_block_contract_invalid_input();

    assert!(matches!(
        execution_error,
        crate::Qwen3_5ExecutionError::InvalidInput { description } if description == "planned visual prompt-cache checkpoints have no block contract"
    ));
}

#[test]
fn should_carry_the_exact_text_block_contract_invalid_input_description() {
    let execution_error = text_prompt_cache_checkpoints_have_no_block_contract_invalid_input();

    assert!(matches!(
        execution_error,
        crate::Qwen3_5ExecutionError::InvalidInput { description } if description == "planned text prompt-cache checkpoints have no block contract"
    ));
}

#[test]
fn should_carry_the_exact_terminal_seed_sampling_invalid_input_description() {
    let execution_error = terminal_prefill_first_generated_token_sampling_failed_invalid_input(
        "simulated sampler failure",
    );

    assert!(matches!(
        execution_error,
        crate::Qwen3_5ExecutionError::InvalidInput { description } if description == "failed to sample the first generated token from terminal prefill logits"
    ));
}
