//! Unit tests for the Qwen-Image-2.1 Astro decoder state astro image prefix key-value cache control logic (prefix length, layout validity, and
//! the extract/reuse slice bookkeeping), compared against the diffusers forward slicing arithmetic
//! in `astro_image_prefix_key_value_cache_fixture.rs`.

use super::*;

/// A realistic cacheable layout from the fixture: leading prefix, target image as a contiguous suffix.
const CACHEABLE_LAYOUT_INDEX: usize = 0;

#[test]
fn should_match_the_oracle_prefix_length_for_every_layout() {
    assert_eq!(
        ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_MASKS.len(),
        ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_LAYOUT_COUNT
    );
    for layout_index in
        0..ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_LAYOUT_COUNT
    {
        let computed = prefix_length(
            &ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_MASKS[layout_index],
        );
        assert_eq!(
            computed,
            ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_PREFIX_LENS
                [layout_index],
            "layout {layout_index}: prefix_len {computed} != oracle {}",
            ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_PREFIX_LENS
                [layout_index]
        );
    }
}

#[test]
fn should_match_the_oracle_astro_image_prefix_key_value_cache_validity_for_every_layout() {
    for layout_index in
        0..ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_LAYOUT_COUNT
    {
        let computed = astro_image_prefix_key_value_cache_is_valid(
            &ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_MASKS[layout_index],
            true,
        );
        assert_eq!(
            computed,
            ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_IS_VALID[layout_index],
            "layout {layout_index}: validity {computed} != oracle {}",
            ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_IS_VALID[layout_index]
        );
    }
    // The fixture must exercise both outcomes, otherwise the test above could pass trivially.
    assert!(
        ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_IS_VALID
            .iter()
            .any(|&is_valid| is_valid),
        "fixture needs a cacheable layout"
    );
    assert!(
        ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_IS_VALID
            .iter()
            .any(|&is_valid| !is_valid),
        "fixture needs an uncacheable layout"
    );
}

#[test]
fn should_reject_every_layout_when_causal_condition_is_disabled() {
    for (layout_index, astro_image_prefix_key_value_cache_mask) in
        ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_MASKS
            .iter()
            .enumerate()
    {
        assert!(
            !astro_image_prefix_key_value_cache_is_valid(
                astro_image_prefix_key_value_cache_mask,
                false
            ),
            "layout {layout_index} must be invalid with causal_condition disabled, even if the layout is a valid suffix"
        );
    }
}

#[test]
fn should_reject_a_target_token_that_appears_after_the_target_block() {
    // Layout 1 has target tokens then a trailing text token: the target is not the suffix, so the
    // single `[prefix_len:]` split would drop the trailing token. This is the invariant the astro image prefix astro decoder state cache needs.
    let interleaved = &ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_MASKS[1];
    assert!(!astro_image_prefix_key_value_cache_is_valid(
        interleaved,
        true
    ));
    assert_eq!(
        prefix_length(interleaved),
        ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_PREFIX_LENS[1]
    );
}

#[test]
fn should_reject_a_pure_target_or_pure_text_sequence_under_causal_condition() {
    // Layout 2 is all target (nothing to astro image prefix astro decoder state cache); layout 3 is all text (nothing to decode).
    assert!(
        !astro_image_prefix_key_value_cache_is_valid(
            &ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_MASKS[2],
            true
        ),
        "pure-target sequence has an empty prefix"
    );
    assert!(
        !astro_image_prefix_key_value_cache_is_valid(
            &ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_MASKS[3],
            true
        ),
        "pure-text sequence has no target to decode"
    );
}

#[test]
fn should_compute_the_whole_sequence_for_prefill_and_extract_and_only_targets_for_reuse() {
    let prefix_len = ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_PREFIX_LENS
        [CACHEABLE_LAYOUT_INDEX];
    assert_eq!(
        astro_image_prefix_key_value_cache_query_slice(
            AstroImagePrefixKeyValueCacheMode::Prefill,
            ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_SEQ_LEN,
            prefix_len
        ),
        (
            0,
            ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_SEQ_LEN
        )
    );
    assert_eq!(
        astro_image_prefix_key_value_cache_query_slice(
            AstroImagePrefixKeyValueCacheMode::Extract,
            ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_SEQ_LEN,
            prefix_len
        ),
        (
            0,
            ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_SEQ_LEN
        )
    );
    assert_eq!(
        astro_image_prefix_key_value_cache_query_slice(
            AstroImagePrefixKeyValueCacheMode::Reuse,
            ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_SEQ_LEN,
            prefix_len
        ),
        (
            prefix_len,
            ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_SEQ_LEN
        )
    );
}

#[test]
fn should_write_only_the_prefix_into_the_astro_image_prefix_key_value_cache_on_extract() {
    let prefix_len = ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_PREFIX_LENS
        [CACHEABLE_LAYOUT_INDEX];
    assert_eq!(
        astro_image_prefix_key_value_cache_write_slice(
            AstroImagePrefixKeyValueCacheMode::Extract,
            prefix_len
        ),
        Some((0, prefix_len))
    );
    assert_eq!(
        astro_image_prefix_key_value_cache_write_slice(
            AstroImagePrefixKeyValueCacheMode::Prefill,
            prefix_len
        ),
        None,
        "prefill keeps nothing"
    );
    assert_eq!(
        astro_image_prefix_key_value_cache_write_slice(
            AstroImagePrefixKeyValueCacheMode::Reuse,
            prefix_len
        ),
        None,
        "reuse reads, never writes"
    );
}

#[test]
fn should_align_the_cached_prefix_with_the_start_of_the_decoded_targets() {
    // The cached prefix `[0, prefix_len)` and the decode query `[prefix_len, seq_len)` must abut so
    // that together they cover the whole joint sequence exactly once across an extract + reuse pair.
    let prefix_len = ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_PREFIX_LENS
        [CACHEABLE_LAYOUT_INDEX];
    let (write_start, write_end) = astro_image_prefix_key_value_cache_write_slice(
        AstroImagePrefixKeyValueCacheMode::Extract,
        prefix_len,
    )
    .expect("extract writes");
    let (query_start, query_end) = astro_image_prefix_key_value_cache_query_slice(
        AstroImagePrefixKeyValueCacheMode::Reuse,
        ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_SEQ_LEN,
        prefix_len,
    );
    assert_eq!(write_start, 0);
    assert_eq!(
        write_end, query_start,
        "cached prefix must end exactly where the decoded targets begin"
    );
    assert_eq!(
        query_end,
        ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_SEQ_LEN
    );
    assert_eq!(
        write_end - write_start + (query_end - query_start),
        ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_SEQ_LEN
    );
}

#[test]
fn should_produce_in_bounds_slices_for_every_layout_and_mode() {
    let modes = [
        AstroImagePrefixKeyValueCacheMode::Prefill,
        AstroImagePrefixKeyValueCacheMode::Extract,
        AstroImagePrefixKeyValueCacheMode::Reuse,
    ];
    for (layout_index, &prefix_len) in
        ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_PREFIX_LENS
            .iter()
            .enumerate()
    {
        for mode in modes {
            let (start, end) = astro_image_prefix_key_value_cache_query_slice(
                mode,
                ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_SEQ_LEN,
                prefix_len,
            );
            assert!(
                start <= end
                    && end
                        <= ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_SEQ_LEN,
                "layout {layout_index} {mode:?}: query {start}..{end} out of bounds"
            );
            if let Some((write_start, write_end)) =
                astro_image_prefix_key_value_cache_write_slice(mode, prefix_len)
            {
                assert!(
                    write_start <= write_end && write_end <= prefix_len,
                    "layout {layout_index} {mode:?}: write {write_start}..{write_end} exceeds prefix {prefix_len}"
                );
            }
        }
    }
}

// Fixed reference values for the Qwen-Image-2.1 Astro decoder state astro image prefix key-value cache prefix slicing, derived from the diffusers
// forward slicing arithmetic. Do not edit by hand; they are cross-checked data.
//
// Pure oracle data consumed by the test module via `use`.
pub const ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_LAYOUT_COUNT: usize = 5;
pub const ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_SEQ_LEN: usize = 10;
pub const ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_MASKS: [[bool; 10]; 5] = [
    [false, false, true, true, true, true, true, true, true, true],
    [
        false, false, false, true, true, true, true, true, true, false,
    ],
    [true, true, true, true, true, true, true, true, true, true],
    [
        false, false, false, false, false, false, false, false, false, false,
    ],
    [false, true, true, true, true, true, true, true, true, true],
];
pub const ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_PREFIX_LENS: [usize; 5] =
    [2usize, 4usize, 0usize, 10usize, 1usize];
pub const ASTRO_IMAGE_PREFIX_KEY_VALUE_ASTRO_IMAGE_PREFIX_KEY_VALUE_CACHE_IS_VALID: [bool; 5] =
    [true, false, false, false, true];
