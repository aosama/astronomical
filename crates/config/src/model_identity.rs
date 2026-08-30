/// Resolves a requested model ID to one known leaf model ID.
///
/// Clients may send a provider prefix such as
/// `astronomical/fake-mixture-of-experts`, while the local registry stores
/// only the leaf directory name.
pub fn resolve_model_id<'model_id>(
    requested_model_id: &'model_id str,
    known_model_ids: &[&str],
) -> &'model_id str {
    if known_model_ids.contains(&requested_model_id) {
        return requested_model_id;
    }
    if let Some((_, provider_stripped_model_id)) = requested_model_id.split_once('/')
        && known_model_ids.contains(&provider_stripped_model_id)
    {
        return provider_stripped_model_id;
    }
    requested_model_id
}

/// Returns the leaf model ID used for serving and request routing.
#[must_use]
pub fn leaf_model_id(model_id: &str) -> &str {
    model_id.rsplit('/').next().unwrap_or(model_id)
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
