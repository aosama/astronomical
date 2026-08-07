use astronomical_model_serving::{
    PersistentPromptCacheWriteQueueOutcome,
    persistent_prompt_cache_write_outcome_advances_parent_chain,
};

#[test]
fn should_advance_the_parent_chain_only_after_a_successful_enqueue_outcome() {
    assert!(persistent_prompt_cache_write_outcome_advances_parent_chain(
        PersistentPromptCacheWriteQueueOutcome::Queued,
    ));
    assert!(persistent_prompt_cache_write_outcome_advances_parent_chain(
        PersistentPromptCacheWriteQueueOutcome::AlreadyQueued,
    ));
    assert!(
        !persistent_prompt_cache_write_outcome_advances_parent_chain(
            PersistentPromptCacheWriteQueueOutcome::DroppedBecauseQueueIsFull,
        )
    );
    assert!(
        !persistent_prompt_cache_write_outcome_advances_parent_chain(
            PersistentPromptCacheWriteQueueOutcome::SkipBecauseCacheIsFull,
        )
    );
}
