//! Shared Romeo-and-Juliet source excerpts for model-input journeys.

pub(crate) fn excerpt(maximum_word_count: usize) -> String {
    include_str!("../fixtures/model_metrics_10000_romeo_and_juliet_words.txt")
        .split_whitespace()
        .take(maximum_word_count)
        .collect::<Vec<_>>()
        .join(" ")
}

pub(crate) fn summary_prompt(maximum_word_count: usize) -> String {
    format!(
        "Summarize this Romeo and Juliet excerpt in one short sentence:\n\n{}",
        excerpt(maximum_word_count)
    )
}
