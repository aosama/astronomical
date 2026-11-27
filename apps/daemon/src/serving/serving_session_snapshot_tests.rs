use super::{CompletedRequestWork, ServingSessionSnapshot};

fn completed_work(
    computed_prompt_tokens: u32,
    prompt_millis: u64,
    generated_tokens: u16,
    writing_millis: u64,
) -> CompletedRequestWork {
    CompletedRequestWork {
        prompt_token_count: computed_prompt_tokens,
        astro_ssd_prompt_cache_restored_token_count: 0,
        computed_prompt_token_count: computed_prompt_tokens,
        generated_token_count: generated_tokens,
        prefill_elapsed_millis: prompt_millis,
        generation_elapsed_millis: writing_millis,
    }
}

#[test]
fn should_have_no_speed_before_any_work() {
    let session = ServingSessionSnapshot::empty();
    assert_eq!(session.average_prefill_tok_per_second, 0.0);
    assert_eq!(session.average_generation_tok_per_second, 0.0);
}

#[test]
fn should_show_the_running_requests_speed_before_any_request_completes() {
    let mut session = ServingSessionSnapshot::empty();
    session.record_running_prefill(2_000, 1_000);
    assert_eq!(session.average_prefill_tok_per_second, 2_000.0);
    session.record_running_generation(50, 1_000);
    assert_eq!(session.average_generation_tok_per_second, 50.0);
    assert_eq!(session.completed_request_count, 0);
}

#[test]
fn should_average_the_running_request_with_completed_ones_by_total_tokens_over_total_time() {
    let mut session = ServingSessionSnapshot::empty();
    session.record_completed_request(completed_work(1_000, 1_000, 40, 2_000));
    session.record_running_generation(10, 1_000);
    // (40 + 10) tokens over (2 + 1) seconds, not the mean of 20 and 10.
    assert!((session.average_generation_tok_per_second - 50.0 / 3.0).abs() < 1e-9);
    assert_eq!(session.average_prefill_tok_per_second, 1_000.0);
}

#[test]
fn should_keep_the_averages_continuous_when_the_running_request_completes() {
    let mut session = ServingSessionSnapshot::empty();
    session.record_completed_request(completed_work(1_000, 1_000, 40, 2_000));
    session.record_running_prefill(3_000, 2_000);
    session.record_running_generation(60, 3_000);
    let averages_while_running = (
        session.average_prefill_tok_per_second,
        session.average_generation_tok_per_second,
    );
    session.record_completed_request(completed_work(3_000, 2_000, 60, 3_000));
    assert_eq!(
        (
            session.average_prefill_tok_per_second,
            session.average_generation_tok_per_second
        ),
        averages_while_running,
        "folding the request in must neither drop nor double its work"
    );
    assert_eq!(session.completed_request_count, 2);
}

#[test]
fn should_drop_a_request_that_ends_without_completing() {
    let mut session = ServingSessionSnapshot::empty();
    session.record_completed_request(completed_work(1_000, 1_000, 40, 2_000));
    session.record_running_generation(500, 1_000);
    session.clear_running_request();
    assert_eq!(session.average_generation_tok_per_second, 20.0);
    assert_eq!(session.completed_request_count, 1);
}

#[test]
fn should_leave_restored_prompt_tokens_out_of_the_reading_speed() {
    let mut session = ServingSessionSnapshot::empty();
    session.record_completed_request(CompletedRequestWork {
        prompt_token_count: 4_000,
        astro_ssd_prompt_cache_restored_token_count: 3_000,
        computed_prompt_token_count: 1_000,
        generated_token_count: 10,
        prefill_elapsed_millis: 500,
        generation_elapsed_millis: 500,
    });
    assert_eq!(session.average_prefill_tok_per_second, 2_000.0);
    assert_eq!(session.total_prompt_token_count, 4_000);
    assert_eq!(session.total_reused_prompt_token_count, 3_000);
}

#[test]
fn should_ignore_a_phase_with_tokens_but_no_time_or_time_but_no_tokens() {
    let mut session = ServingSessionSnapshot::empty();
    session.record_completed_request(completed_work(1_000, 0, 0, 1_000));
    assert_eq!(session.average_prefill_tok_per_second, 0.0);
    assert_eq!(session.average_generation_tok_per_second, 0.0);
    session.record_running_generation(5, 0);
    assert_eq!(session.average_generation_tok_per_second, 0.0);
}
