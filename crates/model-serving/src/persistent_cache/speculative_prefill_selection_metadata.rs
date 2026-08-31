#[cfg(feature = "direct-mlx")]
use super::speculative_prefill_selection::{
    PERSISTENT_SPECULATIVE_PREFILL_SELECTION_FORMAT_VERSION,
    PersistentSpeculativePrefillSelectionContract,
};

#[cfg(feature = "direct-mlx")]
pub(crate) fn hex_encode(selection_hash_bytes: [u8; 32]) -> String {
    selection_hash_bytes
        .iter()
        .map(|selection_hash_byte| format!("{selection_hash_byte:02x}"))
        .collect()
}

#[cfg(feature = "direct-mlx")]
pub(crate) fn selection_file_metadata_entries(
    selection_contract: &PersistentSpeculativePrefillSelectionContract,
    selection_identity_hash: [u8; 32],
) -> [(&'static str, String); 12] {
    [
        (
            "format_version",
            PERSISTENT_SPECULATIVE_PREFILL_SELECTION_FORMAT_VERSION.to_owned(),
        ),
        (
            "target_model_id",
            selection_contract.target_model_id.clone(),
        ),
        (
            "target_model_revision",
            selection_contract.target_model_revision.clone(),
        ),
        ("model_id", selection_contract.draft_model_id.clone()),
        (
            "model_revision",
            selection_contract.draft_model_revision.clone(),
        ),
        (
            "token_identifier_mapping_digest",
            hex_encode(*selection_contract.token_identifier_mapping_digest()),
        ),
        (
            "keep_percentage",
            selection_contract.keep_percentage.to_string(),
        ),
        (
            "selection_chunk_token_count",
            selection_contract.selection_chunk_token_count.to_string(),
        ),
        (
            "mandatory_trailing_token_count",
            selection_contract
                .mandatory_trailing_token_count
                .to_string(),
        ),
        (
            "lookahead_token_count",
            selection_contract.lookahead_token_count.to_string(),
        ),
        (
            "importance_pooling_kernel_token_count",
            selection_contract
                .importance_pooling_kernel_token_count
                .to_string(),
        ),
        ("selection_identity", hex_encode(selection_identity_hash)),
    ]
}

#[cfg(feature = "direct-mlx")]
pub(crate) fn selection_prompt_metadata_entries(
    selection_contract: &PersistentSpeculativePrefillSelectionContract,
) -> [(&'static str, String); 2] {
    [
        (
            "position_tokens",
            selection_contract.position_tokens.to_string(),
        ),
        (
            "prompt_token_count",
            selection_contract.prompt_token_count.to_string(),
        ),
    ]
}
