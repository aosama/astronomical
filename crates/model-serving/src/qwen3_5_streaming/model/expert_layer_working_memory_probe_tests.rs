use std::collections::HashSet;

use super::vocabulary_walking_token_ids;

#[test]
fn should_spread_a_chunk_over_many_different_tokens() {
    let token_ids = vocabulary_walking_token_ids(4_096, 248_320);
    let distinct_token_ids: HashSet<i32> = token_ids.iter().copied().collect();
    assert_eq!(token_ids.len(), 4_096);
    assert_eq!(distinct_token_ids.len(), 4_096);
}

#[test]
fn should_keep_every_token_id_inside_the_vocabulary() {
    let vocabulary_size = 1_000;
    for token_id in vocabulary_walking_token_ids(4_096, vocabulary_size) {
        assert!((0..1_000).contains(&token_id));
    }
}

#[test]
fn should_produce_no_tokens_for_an_empty_or_negative_chunk() {
    assert!(vocabulary_walking_token_ids(0, 248_320).is_empty());
    assert!(vocabulary_walking_token_ids(-5, 248_320).is_empty());
}

#[test]
fn should_not_divide_by_an_empty_vocabulary() {
    assert_eq!(vocabulary_walking_token_ids(3, 0), vec![0, 0, 0]);
}
