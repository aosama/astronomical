//! Canonicalizes the published root namespace of standalone Qwen MTP artifacts.

const CANONICAL_MTP_PREFIX: &str = "language_model.mtp.";
const MAXIMUM_TENSOR_NAME_BYTES: usize = 512;

/// Failure while mapping a standalone stored tensor into Astronomical's MTP namespace.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum StandaloneMtpNamespaceError {
    #[error("standalone MTP tensor name must be nonempty and at most 512 bytes")]
    InvalidLength,
    #[error("standalone MTP tensor '{stored_name}' has an unsupported namespace")]
    UnsupportedNamespace { stored_name: String },
}

/// Maps one supported standalone stored name to the canonical in-memory namespace.
pub fn normalize_qwen3_5_standalone_mtp_tensor_name(
    stored_name: &str,
) -> Result<String, StandaloneMtpNamespaceError> {
    if stored_name.is_empty() || stored_name.len() > MAXIMUM_TENSOR_NAME_BYTES {
        return Err(StandaloneMtpNamespaceError::InvalidLength);
    }
    if stored_name.split('.').any(str::is_empty) || !matches_supported_root(stored_name) {
        return Err(StandaloneMtpNamespaceError::UnsupportedNamespace {
            stored_name: stored_name.to_owned(),
        });
    }
    Ok(format!("{CANONICAL_MTP_PREFIX}{stored_name}"))
}

fn matches_supported_root(stored_name: &str) -> bool {
    stored_name.starts_with("fc.")
        || stored_name.starts_with("layers.0.")
        || stored_name.starts_with("norm.")
        || stored_name.starts_with("pre_fc_norm_embedding.")
        || stored_name.starts_with("pre_fc_norm_hidden.")
}
