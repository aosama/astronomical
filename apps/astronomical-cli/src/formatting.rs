//! Display formatting shared by the CLI verbs.

/// The user-facing gigabyte is the one Apple prints on the box (AGENTS.md):
/// 1 GB = 1,073,741,824 bytes (2³⁰). Two decimals with trailing zeros and
/// the trailing dot trimmed, so `1610612736` renders as `1.5` and
/// `2147483648` as `2`.
pub(crate) fn format_gigabytes(size_bytes: u64) -> String {
    let gigabytes = size_bytes as f64 / 1_073_741_824.0;
    format!("{gigabytes:.2}")
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_owned()
}
