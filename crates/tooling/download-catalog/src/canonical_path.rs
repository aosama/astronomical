//! The repository-relative path form both the catalog and the engine's Hugging
//! Face client accept: ASCII, forward slashes, no empty, `.` or `..` parts.

/// Whether `relative_path` is a canonical repository path no longer than
/// `maximum_path_bytes`.
#[must_use]
pub fn is_canonical_ascii_path(relative_path: &str, maximum_path_bytes: usize) -> bool {
    !relative_path.is_empty()
        && relative_path.len() <= maximum_path_bytes
        && relative_path.is_ascii()
        && !relative_path.contains('\\')
        && !relative_path
            .bytes()
            .any(|character| character.is_ascii_control())
        && relative_path
            .split('/')
            .all(|component| !component.is_empty() && component != "." && component != "..")
}
