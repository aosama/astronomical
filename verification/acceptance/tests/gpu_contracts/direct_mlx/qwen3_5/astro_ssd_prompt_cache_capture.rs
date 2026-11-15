use astronomical_model_serving::{
    AstroSsdPromptCachePublicationOutcome, astro_ssd_prompt_cache_publication_advances_parent_chain,
};

#[test]
fn should_advance_the_parent_chain_only_after_a_durable_publication_outcome() {
    assert!(astro_ssd_prompt_cache_publication_advances_parent_chain(
        AstroSsdPromptCachePublicationOutcome::Published,
    ));
    assert!(astro_ssd_prompt_cache_publication_advances_parent_chain(
        AstroSsdPromptCachePublicationOutcome::AlreadyPublished,
    ));
}
