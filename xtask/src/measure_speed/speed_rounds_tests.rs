use std::collections::BTreeMap;

use super::ModelSpeedRounds;
use crate::measure_speed::speed_prompts::SpeedTask;
use crate::measure_speed::stream_timing::ReplyMeasurement;
use crate::release_speed::speed_ledger::{FigureSpread, TaskFigures};

fn reply(prompt_tokens: u64, seconds_to_first_token: f64, reply_rate: f64) -> ReplyMeasurement {
    ReplyMeasurement {
        prompt_tokens,
        completion_tokens: 500,
        seconds_to_first_token,
        reply_tokens_per_second: Some(reply_rate),
        drafting: None,
        peak_mlx_memory_bytes: None,
    }
}

fn rounds_at_rates(round_rates: &[f64]) -> ModelSpeedRounds {
    let mut speed_rounds = ModelSpeedRounds::default();
    for speed_task in SpeedTask::ALL {
        for round_rate in round_rates {
            speed_rounds.record(speed_task, reply(10_000, 4.0, *round_rate));
        }
    }
    speed_rounds
}

fn medians(rate: u64) -> BTreeMap<&'static str, u64> {
    BTreeMap::from([
        ("essay_short_prompt", rate),
        ("essay_long_prompt", rate),
        ("code_short_prompt", rate),
        ("code_long_prompt", rate),
    ])
}

#[test]
fn should_take_whole_token_medians_only_once_every_task_has_a_round() {
    let mut partial_rounds = ModelSpeedRounds::default();
    partial_rounds.record(SpeedTask::ShortPromptEssay, reply(40, 0.3, 70.0));

    assert_eq!(
        rounds_at_rates(&[70.4, 71.6, 90.0]).medians(),
        Some(medians(72)),
        "the middle of three, rounded"
    );
    assert_eq!(
        partial_rounds.medians(),
        None,
        "a partial set is never written"
    );
}

#[test]
fn should_take_the_midpoint_of_an_even_number_of_rounds() {
    assert_eq!(
        rounds_at_rates(&[60.0, 70.0, 71.0, 90.0]).medians(),
        Some(medians(71)),
        "70.5 rounds to 71"
    );
}

#[test]
fn should_draw_every_figure_of_a_task_from_its_rounds() {
    let mut speed_rounds = ModelSpeedRounds::default();
    for (prompt_tokens, first_token_seconds, reply_rate) in [
        (10_100, 4.0, 60.0),
        (10_300, 5.0, 64.0),
        (10_200, 4.5, 62.0),
    ] {
        speed_rounds.record(
            SpeedTask::LongPromptCode,
            reply(prompt_tokens, first_token_seconds, reply_rate),
        );
    }

    assert_eq!(
        speed_rounds.task_figures(SpeedTask::LongPromptCode),
        Some(TaskFigures {
            round_count: 3,
            prompt_tokens: 10_200,
            prefill_tokens_per_second: FigureSpread {
                median: 2_266.67,
                lowest: 2_060.0,
                highest: 2_525.0
            },
            seconds_to_first_token: FigureSpread {
                median: 4.5,
                lowest: 4.0,
                highest: 5.0
            },
            reply_tokens_per_second: FigureSpread {
                median: 62.0,
                lowest: 60.0,
                highest: 64.0
            },
            drafting: None,
            peak_mlx_memory_bytes: None,
        })
    );
    assert_eq!(speed_rounds.task_figures(SpeedTask::LongPromptEssay), None);
}

#[test]
fn should_not_count_a_reply_that_gave_no_rate_as_a_round() {
    let mut speed_rounds = ModelSpeedRounds::default();
    for _ in 0..3 {
        speed_rounds.record(
            SpeedTask::ShortPromptCode,
            ReplyMeasurement {
                prompt_tokens: 40,
                completion_tokens: 1,
                seconds_to_first_token: 0.2,
                reply_tokens_per_second: None,
                drafting: None,
                peak_mlx_memory_bytes: None,
            },
        );
    }

    assert_eq!(speed_rounds.task_figures(SpeedTask::ShortPromptCode), None);
}

#[test]
fn should_read_no_prompt_speed_from_a_reply_with_no_first_token_time() {
    assert_eq!(reply(10_000, 0.0, 50.0).prefill_tokens_per_second(), 0.0);
    assert_eq!(
        reply(10_000, 4.0, 50.0).prefill_tokens_per_second(),
        2_500.0
    );
}

#[test]
fn should_give_peak_memory_over_the_replies_that_reported_one() {
    let mut speed_rounds = ModelSpeedRounds::default();
    for peak_mlx_memory_bytes in [Some(18_000_000_000), None, Some(20_000_000_000)] {
        let mut task_reply = reply(10_000, 4.0, 60.0);
        task_reply.peak_mlx_memory_bytes = peak_mlx_memory_bytes;
        speed_rounds.record(SpeedTask::LongPromptCode, task_reply);
    }

    let peak_figures = speed_rounds
        .task_figures(SpeedTask::LongPromptCode)
        .expect("three rounds give figures")
        .peak_mlx_memory_bytes
        .expect("two replies reported a peak");

    assert!((peak_figures.lowest - 18_000_000_000.0).abs() < 1.0);
    assert!((peak_figures.highest - 20_000_000_000.0).abs() < 1.0);
}
