use astronomical_model_serving::{
    PersistentPromptCachePublicationOutcome,
    persistent_prompt_cache_publication_advances_parent_chain,
};

#[test]
fn should_advance_the_parent_chain_only_after_a_durable_publication_outcome() {
    assert!(persistent_prompt_cache_publication_advances_parent_chain(
        PersistentPromptCachePublicationOutcome::Published,
    ));
    assert!(persistent_prompt_cache_publication_advances_parent_chain(
        PersistentPromptCachePublicationOutcome::AlreadyPublished,
    ));
}
