//! What drafting did for one reply on the streaming engine, as status and
//! the reply's usage report it (issue #2582).

use ipc_protocol::RequestDraftingStatistics;

use super::engine_request::Qwen3_5EngineRequest;

/// The reply's rounds, drafts and kept drafts when it ends.
///
/// A reply that ends partway through a kept chain leaves decided tokens in
/// its drafter's queue. Every one but the last was counted as a kept draft
/// and never reached the reply; the last is the target's own next choice,
/// not a draft. Those are withdrawn, so the kept count never exceeds the
/// tokens the reply holds.
pub(super) fn reply_drafting_statistics(
    active_request: &Qwen3_5EngineRequest,
) -> RequestDraftingStatistics {
    let reply_state = &active_request.engine_extension;
    let draft_tree_undelivered_token_count = reply_state.draft_tree_drafting.as_ref().map_or(
        0,
        super::draft_tree_reply_drafting::DraftTreeReplyDrafting::undelivered_token_count,
    );
    let mtp_undelivered_token_count = reply_state.mtp_reply_drafting.as_ref().map_or(
        0,
        super::mtp_reply_drafting::MtpReplyDrafting::undelivered_token_count,
    );
    statistics_without_undelivered_drafts(
        reply_state.drafting_statistics,
        draft_tree_undelivered_token_count.max(mtp_undelivered_token_count),
    )
}

/// `recorded_statistics` with the kept drafts still waiting among
/// `undelivered_token_count` decided tokens taken back.
fn statistics_without_undelivered_drafts(
    recorded_statistics: RequestDraftingStatistics,
    undelivered_token_count: usize,
) -> RequestDraftingStatistics {
    let mut reply_statistics = recorded_statistics;
    let undelivered_draft_count = undelivered_token_count.saturating_sub(1);
    reply_statistics.withdraw_unsent_accepted_tokens(
        u32::try_from(undelivered_draft_count).unwrap_or(u32::MAX),
    );
    reply_statistics
}

#[cfg(test)]
#[path = "reply_drafting_record_tests.rs"]
mod tests;
