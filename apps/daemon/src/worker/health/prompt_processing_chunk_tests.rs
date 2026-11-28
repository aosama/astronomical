//! The prompt chunk report a daemon publishes for the loaded model.

use std::sync::{Arc, RwLock};

use ipc_protocol::{PromptProcessingChunkReason, PromptProcessingChunkReport, WorkerEvent};

use super::health_test_support::ready_snapshot;
use super::{WorkerHealthSnapshot, publish_engine_status_event, publish_prompt_processing_chunk};

const EXPERT_MODEL_ON_M5: PromptProcessingChunkReport = PromptProcessingChunkReport {
    chunk_tokens: 4_096,
    reason: PromptProcessingChunkReason::ExpertModelOnM5OrNewer,
};

fn published_report(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
) -> Option<PromptProcessingChunkReport> {
    health_snapshot
        .read()
        .expect("the snapshot reads")
        .prompt_processing_chunk
}

#[test]
fn should_start_with_no_prompt_chunk_until_a_worker_reports_one() {
    assert_eq!(ready_snapshot().prompt_processing_chunk, None);
}

#[test]
fn should_publish_the_prompt_chunk_a_worker_announces() {
    let health_snapshot = Arc::new(RwLock::new(ready_snapshot()));

    publish_engine_status_event(
        &health_snapshot,
        WorkerEvent::PromptProcessingChunkChanged {
            prompt_processing_chunk: EXPERT_MODEL_ON_M5,
        },
    );

    assert_eq!(published_report(&health_snapshot), Some(EXPERT_MODEL_ON_M5));
}

#[test]
fn should_forget_the_prompt_chunk_when_cleared_for_a_new_load() {
    let health_snapshot = Arc::new(RwLock::new(ready_snapshot()));
    publish_prompt_processing_chunk(&health_snapshot, Some(EXPERT_MODEL_ON_M5));

    publish_prompt_processing_chunk(&health_snapshot, None);

    assert_eq!(published_report(&health_snapshot), None);
}
