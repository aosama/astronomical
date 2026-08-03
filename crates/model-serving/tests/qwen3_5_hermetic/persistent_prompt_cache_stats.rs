use astronomical_model_serving::PersistentPromptCacheCounters;

#[test]
fn should_start_with_zero_counters() {
    let persistent_prompt_cache_counters = PersistentPromptCacheCounters::default();
    assert_eq!(
        persistent_prompt_cache_counters.persistent_prompt_cache_hits(),
        0
    );
    assert_eq!(
        persistent_prompt_cache_counters.persistent_prompt_cache_misses(),
        0
    );
    assert_eq!(
        persistent_prompt_cache_counters.persistent_prompt_cache_tokens_saved(),
        0
    );
    assert_eq!(
        persistent_prompt_cache_counters.persistent_prompt_cache_visual_embedding_hits(),
        0
    );
    assert_eq!(
        persistent_prompt_cache_counters.persistent_prompt_cache_visual_embedding_misses(),
        0
    );
    assert_eq!(
        persistent_prompt_cache_counters.persistent_prompt_cache_visual_embedding_rows_loaded(),
        0
    );
}

#[test]
fn should_increment_hits_and_tokens_saved_on_a_cache_hit() {
    let mut persistent_prompt_cache_counters = PersistentPromptCacheCounters::default();
    persistent_prompt_cache_counters.record_cache_hit(2_048);
    assert_eq!(
        persistent_prompt_cache_counters.persistent_prompt_cache_hits(),
        1
    );
    assert_eq!(
        persistent_prompt_cache_counters.persistent_prompt_cache_tokens_saved(),
        2_048
    );
}

#[test]
fn should_increment_misses_on_a_cache_miss() {
    let mut persistent_prompt_cache_counters = PersistentPromptCacheCounters::default();
    persistent_prompt_cache_counters.record_cache_miss();
    assert_eq!(
        persistent_prompt_cache_counters.persistent_prompt_cache_misses(),
        1
    );
    assert_eq!(
        persistent_prompt_cache_counters.persistent_prompt_cache_hits(),
        0
    );
    assert_eq!(
        persistent_prompt_cache_counters.persistent_prompt_cache_tokens_saved(),
        0
    );
}

#[test]
fn should_accumulate_tokens_saved_across_multiple_hits() {
    let mut persistent_prompt_cache_counters = PersistentPromptCacheCounters::default();
    persistent_prompt_cache_counters.record_cache_hit(2_048);
    persistent_prompt_cache_counters.record_cache_hit(4_096);
    assert_eq!(
        persistent_prompt_cache_counters.persistent_prompt_cache_hits(),
        2
    );
    assert_eq!(
        persistent_prompt_cache_counters.persistent_prompt_cache_tokens_saved(),
        6_144
    );
}

#[test]
fn should_return_a_hit_rate_of_zero_when_no_cache_queries_have_occurred() {
    let persistent_prompt_cache_counters = PersistentPromptCacheCounters::default();
    assert_eq!(
        persistent_prompt_cache_counters.persistent_prompt_cache_hit_rate(),
        0.0
    );
}

#[test]
fn should_compute_hit_rate_as_hits_over_total_queries() {
    let mut persistent_prompt_cache_counters = PersistentPromptCacheCounters::default();
    persistent_prompt_cache_counters.record_cache_hit(2_048);
    persistent_prompt_cache_counters.record_cache_hit(4_096);
    persistent_prompt_cache_counters.record_cache_miss();
    // 2 hits / 3 total = 0.6667 rounded to 4 decimals
    assert_eq!(
        persistent_prompt_cache_counters.persistent_prompt_cache_hit_rate(),
        0.6667
    );
}

#[test]
fn should_increment_visual_embedding_hits_and_rows_loaded_on_a_visual_embedding_cache_hit() {
    let mut persistent_prompt_cache_counters = PersistentPromptCacheCounters::default();
    persistent_prompt_cache_counters.record_persistent_prompt_cache_visual_embedding_hit(64);
    persistent_prompt_cache_counters.record_persistent_prompt_cache_visual_embedding_hit(128);

    assert_eq!(
        persistent_prompt_cache_counters.persistent_prompt_cache_visual_embedding_hits(),
        2
    );
    assert_eq!(
        persistent_prompt_cache_counters.persistent_prompt_cache_visual_embedding_rows_loaded(),
        192
    );
    assert_eq!(
        persistent_prompt_cache_counters.persistent_prompt_cache_visual_embedding_misses(),
        0
    );
}

#[test]
fn should_increment_visual_embedding_misses_without_rows_loaded() {
    let mut persistent_prompt_cache_counters = PersistentPromptCacheCounters::default();
    persistent_prompt_cache_counters.record_persistent_prompt_cache_visual_embedding_miss();

    assert_eq!(
        persistent_prompt_cache_counters.persistent_prompt_cache_visual_embedding_misses(),
        1
    );
    assert_eq!(
        persistent_prompt_cache_counters.persistent_prompt_cache_visual_embedding_hits(),
        0
    );
    assert_eq!(
        persistent_prompt_cache_counters.persistent_prompt_cache_visual_embedding_rows_loaded(),
        0
    );
}
