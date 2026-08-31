use std::collections::HashMap;

use astronomical_model_serving::{
    Qwen3_5PersistentPromptCacheBoundaryCheckpoint, VerifiedEmissionQueue, VerifiedTargetFrontier,
};

fn frontier(position_tokens: u32, completed_verifier_rows: usize) -> VerifiedTargetFrontier {
    VerifiedTargetFrontier {
        position_tokens,
        boundary: Qwen3_5PersistentPromptCacheBoundaryCheckpoint {
            completed_prefill_chunk_tokens: completed_verifier_rows,
            recurrent_snapshot_tensors: HashMap::new(),
        },
    }
}

fn depth_three_queue() -> VerifiedEmissionQueue {
    let mut queue = VerifiedEmissionQueue::new(frontier(10, 1));
    queue.push(101, Some(frontier(11, 2)));
    queue.push(102, Some(frontier(12, 3)));
    queue.push(103, None);
    queue
}

#[test]
fn should_own_the_exact_public_frontier_after_zero_one_or_two_drafts_drain() {
    for (drained_count, expected_position_tokens) in [(0, 10), (1, 11), (2, 12)] {
        let mut queue = depth_three_queue();
        for expected_token_id in [101, 102].into_iter().take(drained_count) {
            assert_eq!(queue.pop_front(), Some(expected_token_id));
        }
        let public_frontier = queue
            .take_public_frontier()
            .expect("a partially drained queue should retain its public frontier");
        assert_eq!(public_frontier.position_tokens, expected_position_tokens);
    }
}

#[test]
fn should_release_frontier_ownership_after_the_complete_queue_drains() {
    let mut queue = depth_three_queue();
    assert_eq!(queue.pop_front(), Some(101));
    assert_eq!(queue.pop_front(), Some(102));
    assert_eq!(queue.pop_front(), Some(103));
    assert!(queue.is_empty());
    assert!(queue.take_public_frontier().is_none());
}
