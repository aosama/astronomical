use crate::PerformanceOperation;
use crate::qwen3_5_core::engine_request_test_support::test_engine_request;

#[test]
fn should_keep_penalized_token_ids_sorted_and_unique() {
    let mut engine_request = test_engine_request(());
    for generated_token_id in [42, 7, 42, 1_601, 7] {
        engine_request.observe_penalized_token_id(generated_token_id);
    }
    assert_eq!(engine_request.penalized_token_ids, vec![7, 42, 1_601]);
}

#[test]
fn should_advance_the_position_by_the_forwarded_tokens() {
    let mut engine_request = test_engine_request(());
    engine_request
        .advance_position(512)
        .expect("a prompt chunk");
    engine_request
        .advance_position(1)
        .expect("one decode token");
    assert_eq!(engine_request.next_position_tokens, 513);
}

#[test]
fn should_reach_the_last_position_and_refuse_one_past_it() {
    let mut engine_request = test_engine_request(());
    engine_request.next_position_tokens = u32::MAX - 1;
    engine_request
        .advance_position(1)
        .expect("the last representable position");
    assert_eq!(engine_request.next_position_tokens, u32::MAX);
    assert!(engine_request.advance_position(1).is_err());
    assert_eq!(engine_request.next_position_tokens, u32::MAX);
}

#[test]
fn should_refuse_a_forwarded_token_count_beyond_the_position_range() {
    let mut engine_request = test_engine_request(());
    let beyond_position_range = usize::try_from(u64::from(u32::MAX) + 1)
        .expect("this test targets 64-bit hosts, as the project does");
    assert!(
        engine_request
            .advance_position(beyond_position_range)
            .is_err()
    );
    assert_eq!(engine_request.next_position_tokens, 0);
}

#[test]
fn should_run_a_measured_operation_directly_while_attribution_is_disabled() {
    let mut engine_request = test_engine_request(());
    let observed_position = engine_request.measure_operation_with_request(
        PerformanceOperation::PersistentPromptCacheStateExtraction,
        observe_next_position,
    );
    assert_eq!(observed_position, 0);
    assert!(!engine_request.performance_attribution.is_enabled());
}

fn observe_next_position(
    engine_request: &mut crate::qwen3_5_core::engine_request::Qwen3_5EngineRequest<()>,
) -> u32 {
    engine_request.next_position_tokens
}
