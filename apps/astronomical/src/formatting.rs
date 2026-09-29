//! Display formatting shared by the CLI verbs.

/// Decimal SI gigabytes (1 GB = 1,000,000,000 bytes): two decimals with
/// trailing zeros and the trailing dot trimmed, so `1500000000` renders as
/// `1.5` and `2000000000` as `2`.
pub(crate) fn format_gigabytes(size_bytes: u64) -> String {
    let gigabytes = size_bytes as f64 / 1e9;
    format!("{gigabytes:.2}")
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_owned()
}
