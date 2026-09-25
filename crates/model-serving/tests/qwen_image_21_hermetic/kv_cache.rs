//! Hermetic tests for the Qwen-Image-2.1 KV-cache control logic (prefix length, layout validity, and
//! the extract/reuse slice bookkeeping), compared against the diffusers forward slicing arithmetic
//! in `kv_cache_fixture.rs`.

use astronomical_model_serving::{
    CacheMode, cache_is_valid, cache_query_slice, cache_write_slice, prefix_length,
};

use super::kv_cache_fixture::{
    CACHE_IS_VALID, CACHE_LAYOUT_COUNT, CACHE_MASKS, CACHE_PREFIX_LENS, CACHE_SEQ_LEN,
};

/// A realistic cacheable layout from the fixture: leading prefix, target image as a contiguous suffix.
const CACHEABLE_LAYOUT_INDEX: usize = 0;

#[test]
fn should_match_the_oracle_prefix_length_for_every_layout() {
    assert_eq!(CACHE_MASKS.len(), CACHE_LAYOUT_COUNT);
    for layout_index in 0..CACHE_LAYOUT_COUNT {
        let computed = prefix_length(&CACHE_MASKS[layout_index]);
        assert_eq!(
            computed, CACHE_PREFIX_LENS[layout_index],
            "layout {layout_index}: prefix_len {computed} != oracle {}",
            CACHE_PREFIX_LENS[layout_index]
        );
    }
}

#[test]
fn should_match_the_oracle_cache_validity_for_every_layout() {
    for layout_index in 0..CACHE_LAYOUT_COUNT {
        let computed = cache_is_valid(&CACHE_MASKS[layout_index], true);
        assert_eq!(
            computed, CACHE_IS_VALID[layout_index],
            "layout {layout_index}: validity {computed} != oracle {}",
            CACHE_IS_VALID[layout_index]
        );
    }
    // The fixture must exercise both outcomes, otherwise the test above could pass trivially.
    assert!(
        CACHE_IS_VALID.iter().any(|&is_valid| is_valid),
        "fixture needs a cacheable layout"
    );
    assert!(
        CACHE_IS_VALID.iter().any(|&is_valid| !is_valid),
        "fixture needs an uncacheable layout"
    );
}

#[test]
fn should_reject_every_layout_when_causal_condition_is_disabled() {
    for layout_index in 0..CACHE_LAYOUT_COUNT {
        assert!(
            !cache_is_valid(&CACHE_MASKS[layout_index], false),
            "layout {layout_index} must be invalid with causal_condition disabled, even if the layout is a valid suffix"
        );
    }
}

#[test]
fn should_reject_a_target_token_that_appears_after_the_target_block() {
    // Layout 1 has target tokens then a trailing text token: the target is not the suffix, so the
    // single `[prefix_len:]` split would drop the trailing token. This is the invariant the cache needs.
    let interleaved = &CACHE_MASKS[1];
    assert!(!cache_is_valid(interleaved, true));
    assert_eq!(prefix_length(interleaved), CACHE_PREFIX_LENS[1]);
}

#[test]
fn should_reject_a_pure_target_or_pure_text_sequence_under_causal_condition() {
    // Layout 2 is all target (nothing to cache); layout 3 is all text (nothing to decode).
    assert!(
        !cache_is_valid(&CACHE_MASKS[2], true),
        "pure-target sequence has an empty prefix"
    );
    assert!(
        !cache_is_valid(&CACHE_MASKS[3], true),
        "pure-text sequence has no target to decode"
    );
}

#[test]
fn should_compute_the_whole_sequence_for_prefill_and_extract_and_only_targets_for_reuse() {
    let prefix_len = CACHE_PREFIX_LENS[CACHEABLE_LAYOUT_INDEX];
    assert_eq!(
        cache_query_slice(CacheMode::Prefill, CACHE_SEQ_LEN, prefix_len),
        (0, CACHE_SEQ_LEN)
    );
    assert_eq!(
        cache_query_slice(CacheMode::Extract, CACHE_SEQ_LEN, prefix_len),
        (0, CACHE_SEQ_LEN)
    );
    assert_eq!(
        cache_query_slice(CacheMode::Reuse, CACHE_SEQ_LEN, prefix_len),
        (prefix_len, CACHE_SEQ_LEN)
    );
}

#[test]
fn should_write_only_the_prefix_into_the_cache_on_extract() {
    let prefix_len = CACHE_PREFIX_LENS[CACHEABLE_LAYOUT_INDEX];
    assert_eq!(
        cache_write_slice(CacheMode::Extract, prefix_len),
        Some((0, prefix_len))
    );
    assert_eq!(
        cache_write_slice(CacheMode::Prefill, prefix_len),
        None,
        "prefill keeps nothing"
    );
    assert_eq!(
        cache_write_slice(CacheMode::Reuse, prefix_len),
        None,
        "reuse reads, never writes"
    );
}

#[test]
fn should_align_the_cached_prefix_with_the_start_of_the_decoded_targets() {
    // The cached prefix `[0, prefix_len)` and the decode query `[prefix_len, seq_len)` must abut so
    // that together they cover the whole joint sequence exactly once across an extract + reuse pair.
    let prefix_len = CACHE_PREFIX_LENS[CACHEABLE_LAYOUT_INDEX];
    let (write_start, write_end) =
        cache_write_slice(CacheMode::Extract, prefix_len).expect("extract writes");
    let (query_start, query_end) = cache_query_slice(CacheMode::Reuse, CACHE_SEQ_LEN, prefix_len);
    assert_eq!(write_start, 0);
    assert_eq!(
        write_end, query_start,
        "cached prefix must end exactly where the decoded targets begin"
    );
    assert_eq!(query_end, CACHE_SEQ_LEN);
    assert_eq!(
        write_end - write_start + (query_end - query_start),
        CACHE_SEQ_LEN
    );
}

#[test]
fn should_produce_in_bounds_slices_for_every_layout_and_mode() {
    let modes = [CacheMode::Prefill, CacheMode::Extract, CacheMode::Reuse];
    for layout_index in 0..CACHE_LAYOUT_COUNT {
        let prefix_len = CACHE_PREFIX_LENS[layout_index];
        for mode in modes {
            let (start, end) = cache_query_slice(mode, CACHE_SEQ_LEN, prefix_len);
            assert!(
                start <= end && end <= CACHE_SEQ_LEN,
                "layout {layout_index} {mode:?}: query {start}..{end} out of bounds"
            );
            if let Some((write_start, write_end)) = cache_write_slice(mode, prefix_len) {
                assert!(
                    write_start <= write_end && write_end <= prefix_len,
                    "layout {layout_index} {mode:?}: write {write_start}..{write_end} exceeds prefix {prefix_len}"
                );
            }
        }
    }
}
