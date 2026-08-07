#[allow(dead_code)]
#[path = "../../src/qwen3_5/inference_execution/prefill_execution_context.rs"]
mod prefill_execution_context;
#[path = "../../src/qwen3_5/inference_execution/speculative_prefill.rs"]
mod speculative_prefill_policy;

use prefill_execution_context::{
    CAPACITY_REDUCED_CONTEXT_FLAG, Qwen3_5PrefillExecutionContext,
    SPECULATIVE_PREFILL_TARGET_ONLY_MTP_PREFIX_CONTEXT_FLAG,
};
use speculative_prefill_policy::{
    Qwen3_5SpeculativePrefillChunckMode, qwen3_5_speculative_prefill_chunck_mode,
};

#[test]
fn should_use_ordinary_target_prefill_when_mtp_is_disabled_for_a_nonterminal_chunck() {
    assert_eq!(
        qwen3_5_speculative_prefill_chunck_mode(false, 3, 5),
        Qwen3_5SpeculativePrefillChunckMode::OrdinaryTarget,
    );
}

#[test]
fn should_use_ordinary_target_prefill_when_mtp_is_disabled_for_a_terminal_chunck() {
    assert_eq!(
        qwen3_5_speculative_prefill_chunck_mode(false, 5, 5),
        Qwen3_5SpeculativePrefillChunckMode::OrdinaryTarget,
    );
}

#[test]
fn should_use_target_only_mtp_prefix_prefill_for_an_active_nonterminal_chunck() {
    assert_eq!(
        qwen3_5_speculative_prefill_chunck_mode(true, 3, 5),
        Qwen3_5SpeculativePrefillChunckMode::TargetOnlyMtpPrefix,
    );
}

#[test]
fn should_use_terminal_mtp_capture_for_an_active_terminal_chunck() {
    assert_eq!(
        qwen3_5_speculative_prefill_chunck_mode(true, 5, 5),
        Qwen3_5SpeculativePrefillChunckMode::TerminalMtpCapture,
    );
}

#[test]
fn should_keep_target_only_mtp_prefix_and_capacity_reduced_context_flags_distinct() {
    let target_only_mtp_prefix_context =
        Qwen3_5PrefillExecutionContext::new(false, true, false, false)
            .with_target_only_mtp_prefix(true);

    assert_ne!(
        SPECULATIVE_PREFILL_TARGET_ONLY_MTP_PREFIX_CONTEXT_FLAG,
        CAPACITY_REDUCED_CONTEXT_FLAG,
    );
    assert_eq!(
        target_only_mtp_prefix_context.context_identifier_flags()
            & SPECULATIVE_PREFILL_TARGET_ONLY_MTP_PREFIX_CONTEXT_FLAG,
        SPECULATIVE_PREFILL_TARGET_ONLY_MTP_PREFIX_CONTEXT_FLAG,
    );
    assert_eq!(
        target_only_mtp_prefix_context.context_identifier_flags() & CAPACITY_REDUCED_CONTEXT_FLAG,
        0,
    );
}
