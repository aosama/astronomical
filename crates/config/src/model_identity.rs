/// Resolves a requested model ID to one known model ID: the exact ID when known, else its leaf.
///
/// Clients may send a provider prefix such as `mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit`
/// (the full Hugging Face ID that the Library and `models download` use), while the local
/// registry stores only the leaf directory name. An unknown ID comes back unchanged, so the
/// caller's own unknown-model path reports what the person typed.
pub fn resolve_model_id<'model_id>(
    requested_model_id: &'model_id str,
    known_model_ids: &[&str],
) -> &'model_id str {
    if known_model_ids.contains(&requested_model_id) {
        return requested_model_id;
    }
    let requested_leaf_model_id = leaf_model_id(requested_model_id);
    if known_model_ids.contains(&requested_leaf_model_id) {
        return requested_leaf_model_id;
    }
    requested_model_id
}

/// Picks the `models` entry that configures one discovered model, in this order: an entry
/// keyed by the exact discovered (leaf) ID; an entry keyed by the model's recorded full
/// Hugging Face ID (`provider_model_id`); and, only when no full ID is recorded, the first entry
/// in key order whose last path segment is the leaf. Once a full ID is recorded, an entry for a
/// different organisation's model of the same name never applies. Resolution and the
/// unmatched-entry report both use this one rule, so an entry that applies is never reported as
/// ignored, nor the reverse.
pub fn configured_model_id_for<'configured>(
    discovered_model_id: &str,
    provider_model_id: Option<&str>,
    configured_model_ids: impl IntoIterator<Item = &'configured str>,
) -> Option<&'configured str> {
    let mut provider_entry = None;
    let mut first_leaf_entry = None;
    for configured_model_id in configured_model_ids {
        if configured_model_id == discovered_model_id {
            return Some(configured_model_id);
        }
        if provider_model_id == Some(configured_model_id) {
            provider_entry = Some(configured_model_id);
        }
        if first_leaf_entry.is_none() && leaf_model_id(configured_model_id) == discovered_model_id {
            first_leaf_entry = Some(configured_model_id);
        }
    }
    if provider_model_id.is_some() {
        return provider_entry;
    }
    first_leaf_entry
}

/// Returns the leaf model ID used for serving and request routing.
#[must_use]
pub fn leaf_model_id(model_id: &str) -> &str {
    model_id.rsplit('/').next().unwrap_or(model_id)
}

/// Suggests up to three known model IDs that plausibly match the request:
/// case-insensitive substring matches in both directions, then candidates
/// sharing the request's leading token (typo tolerance for family prefixes).
/// Returned suggestions are lowercased.
pub fn near_model_matches(
    requested_model_id: &str,
    candidate_model_ids: &[impl AsRef<str>],
) -> Vec<String> {
    let requested = requested_model_id.to_ascii_lowercase();
    let leading_token = requested
        .split(['/', ' ', '-'])
        .next()
        .unwrap_or(&requested);
    let mut matches = candidate_model_ids
        .iter()
        .map(|model_id| model_id.as_ref().to_ascii_lowercase())
        .filter(|candidate| {
            candidate.contains(&requested)
                || requested.contains(candidate.as_str())
                || candidate.starts_with(leading_token)
        })
        .collect::<Vec<_>>();
    matches.sort();
    matches.dedup();
    matches.truncate(3);
    matches
}

/// Decodes a Hugging Face cache directory name into an `organization/model` ID.
pub fn decode_huggingface_cache_directory_name(directory_name: &str) -> Option<String> {
    let encoded_model_id = directory_name.strip_prefix("models--")?;
    if encoded_model_id.is_empty() {
        return None;
    }
    let mut encoded_model_id_parts = encoded_model_id.splitn(2, "--");
    let organization_name = encoded_model_id_parts.next()?;
    match encoded_model_id_parts.next() {
        Some(repository_name) => Some(format!("{organization_name}/{repository_name}")),
        None => Some(organization_name.to_owned()),
    }
}
