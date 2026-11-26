//! What drafting did for the replies of a measured task, read from each
//! reply's own usage: the rounds it ran and the drafts the model kept. A
//! drafted speed is only explained with these beside it, and they show when
//! a reply stopped drafting (issue #2582).

use serde_json::Value;

use super::stream_timing::ReplyMeasurement;
use crate::release_speed::speed_ledger::{DraftingFigures, FigureSpread};

/// One reply's drafting, as its usage reports it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplyDrafting {
    pub round_count: u64,
    pub kept_draft_count: u64,
}

impl ReplyDrafting {
    /// The drafting a reply's usage reports; `None` for a reply that never
    /// drafted, whose usage carries no completion details.
    #[must_use]
    pub fn from_usage(usage_counts: &Value) -> Option<Self> {
        let completion_details = usage_counts
            .get("completion_tokens_details")
            .filter(|details| details.is_object())?;
        Some(Self {
            round_count: completion_details["draft_rounds"].as_u64().unwrap_or(0),
            kept_draft_count: completion_details["accepted_prediction_tokens"]
                .as_u64()
                .unwrap_or(0),
        })
    }

    fn kept_drafts_a_round(self) -> f64 {
        if self.round_count == 0 {
            return 0.0;
        }
        self.kept_draft_count as f64 / self.round_count as f64
    }
}

/// A task's drafting over its replies; `None` when none of them drafted. A
/// reply that did not draft among ones that did counts as no rounds, so a
/// reply that gave drafting up shows in the lowest figure.
#[must_use]
pub fn drafting_figures(task_replies: &[ReplyMeasurement]) -> Option<DraftingFigures> {
    if task_replies
        .iter()
        .all(|task_reply| task_reply.drafting.is_none())
    {
        return None;
    }
    let mut round_counts = Vec::new();
    let mut kept_drafts_a_round = Vec::new();
    let mut drafted_shares_of_reply = Vec::new();
    for task_reply in task_replies {
        let reply_drafting = task_reply.drafting.unwrap_or(ReplyDrafting {
            round_count: 0,
            kept_draft_count: 0,
        });
        round_counts.push(reply_drafting.round_count as f64);
        kept_drafts_a_round.push(reply_drafting.kept_drafts_a_round());
        drafted_shares_of_reply.push(drafted_share_of_reply(
            reply_drafting.kept_draft_count,
            task_reply.completion_tokens,
        ));
    }
    Some(DraftingFigures {
        rounds_a_reply: FigureSpread::of_rounds(&round_counts)?,
        kept_drafts_a_round: FigureSpread::of_rounds(&kept_drafts_a_round)?,
        drafted_share_of_reply: FigureSpread::of_rounds(&drafted_shares_of_reply)?,
    })
}

/// Kept drafts as a share of the reply's tokens, between 0 and 1.
fn drafted_share_of_reply(kept_draft_count: u64, reply_token_count: u64) -> f64 {
    if reply_token_count == 0 {
        return 0.0;
    }
    (kept_draft_count as f64 / reply_token_count as f64).min(1.0)
}

#[cfg(test)]
#[path = "reply_drafting_tests.rs"]
mod reply_drafting_tests;
