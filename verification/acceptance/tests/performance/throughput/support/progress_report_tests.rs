use astronomical_daemon::ActiveRequestProgress;

use super::{progress_line, progress_line_for};

#[test]
fn should_report_percent_and_time_left_from_the_rate_so_far() {
    assert_eq!(
        progress_line("prompt", 4_096, 10_240, 12_000),
        "progress stage=prompt 4096/10240 tokens 40% elapsed 12s left 18s"
    );
}

#[test]
fn should_report_an_unknown_time_left_before_the_first_token_completes() {
    assert_eq!(
        progress_line("reply", 0, 500, 3_000),
        "progress stage=reply 0/500 tokens 0% elapsed 3s left unknown"
    );
}

#[test]
fn should_report_nothing_left_once_every_token_completed() {
    assert_eq!(
        progress_line("reply", 500, 500, 20_500),
        "progress stage=reply 500/500 tokens 100% elapsed 20s left 0s"
    );
}

#[test]
fn should_not_divide_by_a_zero_total() {
    assert_eq!(
        progress_line("prompt", 0, 0, 0),
        "progress stage=prompt 0/0 tokens 0% elapsed 0s left unknown"
    );
}

#[test]
fn should_not_underflow_when_more_tokens_completed_than_the_reported_total() {
    assert_eq!(
        progress_line("reply", 12, 10, 1_000),
        "progress stage=reply 12/10 tokens 120% elapsed 1s left 0s"
    );
}

#[test]
fn should_describe_a_reply_in_progress_by_its_generated_tokens() {
    let reply_progress = ActiveRequestProgress::Generation {
        generated_token_count: 125,
        maximum_output_tokens: 500,
        elapsed_millis: 5_000,
    };
    assert_eq!(
        progress_line_for(&reply_progress).as_deref(),
        Some("progress stage=reply 125/500 tokens 25% elapsed 5s left 15s")
    );
}

#[test]
fn should_print_nothing_for_a_stage_without_token_counts() {
    let image_progress = ActiveRequestProgress::ImageGeneration {
        phase: ipc_protocol::ImageGenerationPhase::Denoising,
        completed_steps: 2,
        total_steps: 4,
        elapsed_millis: 900,
    };
    assert_eq!(progress_line_for(&image_progress), None);
}
