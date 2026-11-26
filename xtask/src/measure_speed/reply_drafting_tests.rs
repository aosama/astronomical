use super::{ReplyDrafting, drafting_figures};
use crate::measure_speed::stream_timing::ReplyMeasurement;

fn reply(completion_tokens: u64, drafting: Option<ReplyDrafting>) -> ReplyMeasurement {
    ReplyMeasurement {
        prompt_tokens: 40,
        completion_tokens,
        seconds_to_first_token: 0.3,
        reply_tokens_per_second: Some(60.0),
        drafting,
        peak_mlx_memory_bytes: None,
    }
}

#[test]
fn should_read_rounds_and_kept_drafts_from_a_drafted_replys_usage() {
    let usage_counts = serde_json::json!({
        "prompt_tokens": 40,
        "completion_tokens": 200,
        "completion_tokens_details": {
            "accepted_prediction_tokens": 150,
            "rejected_prediction_tokens": 250,
            "draft_rounds": 50
        }
    });

    assert_eq!(
        ReplyDrafting::from_usage(&usage_counts),
        Some(ReplyDrafting {
            round_count: 50,
            kept_draft_count: 150
        })
    );
}

#[test]
fn should_read_no_drafting_from_a_usage_without_completion_details() {
    let usage_counts = serde_json::json!({"prompt_tokens": 40, "completion_tokens": 200});

    assert_eq!(ReplyDrafting::from_usage(&usage_counts), None);
    assert_eq!(ReplyDrafting::from_usage(&serde_json::Value::Null), None);
}

#[test]
fn should_give_no_drafting_figures_when_no_reply_drafted() {
    assert_eq!(
        drafting_figures(&[reply(200, None), reply(200, None)]),
        None
    );
    assert_eq!(drafting_figures(&[]), None);
}

#[test]
fn should_give_rounds_kept_drafts_a_round_and_the_drafted_share() {
    let task_figures = drafting_figures(&[
        reply(
            200,
            Some(ReplyDrafting {
                round_count: 50,
                kept_draft_count: 150,
            }),
        ),
        reply(
            200,
            Some(ReplyDrafting {
                round_count: 40,
                kept_draft_count: 160,
            }),
        ),
        reply(
            200,
            Some(ReplyDrafting {
                round_count: 100,
                kept_draft_count: 100,
            }),
        ),
    ])
    .expect("drafted replies give figures");

    assert!((task_figures.rounds_a_reply.median - 50.0).abs() < 1e-9);
    assert!((task_figures.kept_drafts_a_round.median - 3.0).abs() < 1e-9);
    assert!((task_figures.kept_drafts_a_round.highest - 4.0).abs() < 1e-9);
    assert!((task_figures.drafted_share_of_reply.median - 0.75).abs() < 1e-9);
    assert!((task_figures.drafted_share_of_reply.lowest - 0.5).abs() < 1e-9);
}

#[test]
fn should_show_a_reply_that_gave_drafting_up_as_the_lowest_figure() {
    // Edge: one reply drafted and one did not; a reply with no tokens and a
    // round count of zero must not divide by zero.
    let task_figures = drafting_figures(&[
        reply(
            200,
            Some(ReplyDrafting {
                round_count: 50,
                kept_draft_count: 150,
            }),
        ),
        reply(200, None),
        reply(
            0,
            Some(ReplyDrafting {
                round_count: 0,
                kept_draft_count: 0,
            }),
        ),
    ])
    .expect("one drafted reply is enough for figures");

    assert!(task_figures.rounds_a_reply.lowest.abs() < 1e-9);
    assert!(task_figures.kept_drafts_a_round.lowest.abs() < 1e-9);
    assert!(task_figures.drafted_share_of_reply.lowest.abs() < 1e-9);
    assert!((task_figures.drafted_share_of_reply.highest - 0.75).abs() < 1e-9);
}
