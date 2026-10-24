use super::{
    MINIMUM_SHRUNK_RESIDENT_PREFILL_CHUNK_TOKEN_COUNT,
    next_shrunk_resident_prefill_chunk_token_count,
};

#[test]
fn should_halve_the_configured_two_thousand_forty_eight_token_chunk() {
    assert_eq!(
        next_shrunk_resident_prefill_chunk_token_count(2048),
        Some(1024)
    );
}

#[test]
fn should_clamp_a_halved_chunk_to_the_minimum() {
    assert_eq!(
        next_shrunk_resident_prefill_chunk_token_count(400),
        Some(MINIMUM_SHRUNK_RESIDENT_PREFILL_CHUNK_TOKEN_COUNT)
    );
}

#[test]
fn should_shrink_one_token_above_the_minimum_to_the_minimum() {
    assert_eq!(
        next_shrunk_resident_prefill_chunk_token_count(
            MINIMUM_SHRUNK_RESIDENT_PREFILL_CHUNK_TOKEN_COUNT + 1
        ),
        Some(MINIMUM_SHRUNK_RESIDENT_PREFILL_CHUNK_TOKEN_COUNT)
    );
}

#[test]
fn should_stop_shrinking_at_the_minimum() {
    assert_eq!(
        next_shrunk_resident_prefill_chunk_token_count(
            MINIMUM_SHRUNK_RESIDENT_PREFILL_CHUNK_TOKEN_COUNT
        ),
        None
    );
}

#[test]
fn should_not_shrink_an_empty_or_single_token_chunk() {
    assert_eq!(next_shrunk_resident_prefill_chunk_token_count(0), None);
    assert_eq!(next_shrunk_resident_prefill_chunk_token_count(1), None);
}

#[test]
fn should_reach_the_minimum_in_three_halvings_from_two_thousand_forty_eight_tokens() {
    let mut shrunk_prefill_chunk_token_counts = Vec::new();
    let mut rejected_prefill_chunk_token_count = 2048;
    while let Some(shrunk_prefill_chunk_token_count) =
        next_shrunk_resident_prefill_chunk_token_count(rejected_prefill_chunk_token_count)
    {
        shrunk_prefill_chunk_token_counts.push(shrunk_prefill_chunk_token_count);
        rejected_prefill_chunk_token_count = shrunk_prefill_chunk_token_count;
    }
    assert_eq!(shrunk_prefill_chunk_token_counts, vec![1024, 512, 256]);
}
