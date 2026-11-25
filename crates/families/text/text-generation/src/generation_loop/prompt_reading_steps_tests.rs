use super::PromptStretch;

fn token_count_from(prefill_start: usize, prefill_end: usize) -> usize {
    PromptStretch {
        prefill_start,
        prefill_end,
    }
    .token_count()
}

#[test]
fn should_count_the_tokens_a_stretch_covers() {
    assert_eq!(token_count_from(2_048, 4_096), 2_048);
}

#[test]
fn should_count_one_token_for_a_single_token_tail() {
    assert_eq!(token_count_from(10_010, 10_011), 1);
}

#[test]
fn should_count_no_tokens_for_an_empty_stretch() {
    assert_eq!(token_count_from(512, 512), 0);
}
