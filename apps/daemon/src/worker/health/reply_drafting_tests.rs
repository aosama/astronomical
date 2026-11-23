//! What drafting did for the last reply and since the model loaded, as the
//! daemon publishes it.

use std::sync::{Arc, RwLock};

use ipc_protocol::RequestDraftingStatistics;

use super::{WorkerHealthSnapshot, WorkerHealthStatus, record_reply_drafting};

fn reply_drafting(
    drafted_token_count: u32,
    accepted_token_count: u32,
) -> RequestDraftingStatistics {
    RequestDraftingStatistics {
        round_count: 1,
        drafted_token_count,
        accepted_token_count,
    }
}

fn health_snapshot() -> Arc<RwLock<WorkerHealthSnapshot>> {
    Arc::new(RwLock::new(WorkerHealthSnapshot::unavailable(
        WorkerHealthStatus::Loading,
    )))
}

#[test]
fn should_keep_the_last_reply_and_add_it_to_the_session() {
    let health_snapshot = health_snapshot();
    record_reply_drafting(&health_snapshot, Some(reply_drafting(3, 2)));
    record_reply_drafting(&health_snapshot, Some(reply_drafting(3, 1)));
    let published_snapshot = health_snapshot.read().expect("the snapshot is readable");
    assert_eq!(
        published_snapshot.last_reply_drafting,
        Some(reply_drafting(3, 1))
    );
    assert_eq!(published_snapshot.session_drafting.drafted_token_count, 6);
    assert_eq!(published_snapshot.session_drafting.accepted_token_count, 3);
    assert_eq!(published_snapshot.session_drafting.round_count, 2);
}

#[test]
fn should_clear_the_last_reply_but_keep_the_session_when_a_reply_did_not_draft() {
    let health_snapshot = health_snapshot();
    record_reply_drafting(&health_snapshot, Some(reply_drafting(3, 2)));
    record_reply_drafting(&health_snapshot, None);
    let published_snapshot = health_snapshot.read().expect("the snapshot is readable");
    assert_eq!(published_snapshot.last_reply_drafting, None);
    assert_eq!(published_snapshot.session_drafting.accepted_token_count, 2);
}
