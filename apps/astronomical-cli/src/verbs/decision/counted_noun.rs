//! Counts written the way a person reads them in the decision reports.
//!
//! Any amount of data trains now, so a report can open with one example
//! about one text; "1 examples" reads as a mistake to a newcomer.

/// `count` followed by `singular` for exactly one and `plural` otherwise.
pub(super) fn counted_noun(count: usize, singular: &str, plural: &str) -> String {
    if count == 1 {
        format!("1 {singular}")
    } else {
        format!("{count} {plural}")
    }
}

#[cfg(test)]
#[path = "counted_noun_tests.rs"]
mod tests;
