use sha2::{Digest, Sha256};

use super::speculative_prefill_policy::PersistentSpeculativePrefillPolicyIdentity;

#[cfg(feature = "direct-mlx")]
use astronomical_runtime_integration::MlxArray;
#[cfg(feature = "direct-mlx")]
use std::collections::HashMap;
#[cfg(feature = "direct-mlx")]
use std::{fs::File, path::Path};

#[cfg(feature = "direct-mlx")]
use super::{
    model_contract::PersistentPromptCacheModelContract,
    persistent_safetensors_header::read_persistent_safetensors_header,
};

pub const PERSISTENT_SPECULATIVE_PREFILL_TARGET_STATE_FORMAT_VERSION: &str = "3";
#[cfg(feature = "direct-mlx")]
pub const SPECULATIVE_PREFILL_TARGET_SELECTED_POSITIONS_TENSOR_NAME: &str =
    "selected_target_token_positions";

#[cfg(feature = "direct-mlx")]
pub struct RestoredSpeculativePrefillTargetState {
    prompt_prefix_token_count: usize,
    selected_target_token_positions: MlxArray,
    decoder_state_tensors: HashMap<String, MlxArray>,
}

#[cfg(feature = "direct-mlx")]
impl RestoredSpeculativePrefillTargetState {
    pub(crate) fn new(
        prompt_prefix_token_count: usize,
        selected_target_token_positions: MlxArray,
        decoder_state_tensors: HashMap<String, MlxArray>,
    ) -> Self {
        Self {
            prompt_prefix_token_count,
            selected_target_token_positions,
            decoder_state_tensors,
        }
    }

    #[must_use]
    pub const fn prompt_prefix_token_count(&self) -> usize {
        self.prompt_prefix_token_count
    }

    #[must_use]
    pub const fn selected_target_token_positions(&self) -> &MlxArray {
        &self.selected_target_token_positions
    }

    #[must_use]
    pub const fn decoder_state_tensors(&self) -> &HashMap<String, MlxArray> {
        &self.decoder_state_tensors
    }

    pub(crate) fn into_parts(self) -> (usize, MlxArray, HashMap<String, MlxArray>) {
        (
            self.prompt_prefix_token_count,
            self.selected_target_token_positions,
            self.decoder_state_tensors,
        )
    }
}

/// Target and drafter selection identity required to reuse sparse target decoder state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistentSpeculativePrefillTargetStateContract {
    target_model_id: String,
    target_model_revision: String,
    drafter_model_id: String,
    drafter_model_revision: String,
    token_identifier_mapping_digest: [u8; 32],
    keep_percentage: u32,
    selection_chunk_token_count: u32,
    mandatory_trailing_token_count: u32,
    lookahead_token_count: u32,
    importance_pooling_kernel_token_count: u32,
}

impl PersistentSpeculativePrefillTargetStateContract {
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        target_model_id: String,
        target_model_revision: String,
        drafter_model_id: String,
        drafter_model_revision: String,
        token_identifier_mapping_digest: [u8; 32],
        keep_percentage: u32,
        selection_chunk_token_count: u32,
        mandatory_trailing_token_count: u32,
        lookahead_token_count: u32,
        importance_pooling_kernel_token_count: u32,
    ) -> Self {
        Self {
            target_model_id,
            target_model_revision,
            drafter_model_id,
            drafter_model_revision,
            token_identifier_mapping_digest,
            keep_percentage,
            selection_chunk_token_count,
            mandatory_trailing_token_count,
            lookahead_token_count,
            importance_pooling_kernel_token_count,
        }
    }

    #[must_use]
    pub fn target_state_identity_hash(
        &self,
        prompt_prefix_token_ids: &[u32],
        ordered_image_sha256_digests: &[[u8; 32]],
    ) -> [u8; 32] {
        let mut target_state_identity_hasher =
            self.target_state_identity_hasher(ordered_image_sha256_digests);
        for prompt_token_id in prompt_prefix_token_ids {
            target_state_identity_hasher.update(prompt_token_id.to_le_bytes());
        }
        target_state_identity_hasher.finalize().into()
    }

    /// Returns the target, drafter, and keep percentage bound to this state.
    #[must_use]
    pub fn policy_identity(&self) -> PersistentSpeculativePrefillPolicyIdentity {
        PersistentSpeculativePrefillPolicyIdentity::new(
            self.target_model_id.clone(),
            self.target_model_revision.clone(),
            self.drafter_model_id.clone(),
            self.drafter_model_revision.clone(),
            self.keep_percentage,
        )
    }

    #[cfg(feature = "direct-mlx")]
    pub(crate) fn target_model_id(&self) -> &str {
        &self.target_model_id
    }

    #[cfg(feature = "direct-mlx")]
    pub(crate) fn target_model_revision(&self) -> &str {
        &self.target_model_revision
    }

    fn target_state_identity_hasher(&self, ordered_image_sha256_digests: &[[u8; 32]]) -> Sha256 {
        let mut target_state_identity_hasher = Sha256::new();
        append_length_prefixed_bytes(
            &mut target_state_identity_hasher,
            b"astronomical-speculative-prefill-target-state-v1",
        );
        for model_identity_component in [
            self.target_model_id.as_bytes(),
            self.target_model_revision.as_bytes(),
            self.drafter_model_id.as_bytes(),
            self.drafter_model_revision.as_bytes(),
        ] {
            append_length_prefixed_bytes(
                &mut target_state_identity_hasher,
                model_identity_component,
            );
        }
        target_state_identity_hasher.update(self.token_identifier_mapping_digest);
        for selection_configuration_number in [
            self.keep_percentage,
            self.selection_chunk_token_count,
            self.mandatory_trailing_token_count,
            self.lookahead_token_count,
            self.importance_pooling_kernel_token_count,
        ] {
            target_state_identity_hasher.update(selection_configuration_number.to_le_bytes());
        }
        target_state_identity_hasher.update(
            u64::try_from(ordered_image_sha256_digests.len())
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        for image_sha256_digest in ordered_image_sha256_digests {
            target_state_identity_hasher.update(image_sha256_digest);
        }
        target_state_identity_hasher
    }
}

#[cfg(feature = "direct-mlx")]
pub(crate) struct PersistentSpeculativePrefillTargetStateFileHeader {
    prompt_prefix_token_count: usize,
    policy_identity: PersistentSpeculativePrefillPolicyIdentity,
    tensor_names: Vec<String>,
}

#[cfg(feature = "direct-mlx")]
impl PersistentSpeculativePrefillTargetStateFileHeader {
    pub(crate) fn read_model_bound_from_file(
        target_state_file: &File,
        target_state_file_path: &Path,
        target_model_contract: &PersistentPromptCacheModelContract,
    ) -> Result<Self, String> {
        let parsed_header =
            read_persistent_safetensors_header(target_state_file, target_state_file_path)
                .map_err(|header_error| header_error.to_string())?;
        require_metadata(
            &parsed_header.metadata,
            "format_version",
            PERSISTENT_SPECULATIVE_PREFILL_TARGET_STATE_FORMAT_VERSION,
        )?;
        require_metadata(
            &parsed_header.metadata,
            "target_model_id",
            target_model_contract.model_id(),
        )?;
        require_metadata(
            &parsed_header.metadata,
            "target_model_revision",
            target_model_contract.model_revision(),
        )?;
        let expected_identity_text = target_state_file_path
            .file_stem()
            .and_then(|file_stem| file_stem.to_str())
            .ok_or_else(|| "sparse target state filename has no UTF-8 identity".to_owned())?;
        require_metadata(
            &parsed_header.metadata,
            "target_state_identity_sha256",
            expected_identity_text,
        )?;
        let prompt_prefix_token_count = parsed_header
            .metadata
            .get("prompt_prefix_token_count")
            .ok_or_else(|| "sparse target state is missing prompt_prefix_token_count".to_owned())?
            .parse::<usize>()
            .map_err(|_| "sparse target prompt_prefix_token_count is invalid".to_owned())?;
        if prompt_prefix_token_count == 0
            || parsed_header.tensor_views.len() <= 1
            || !parsed_header
                .tensor_views
                .contains_key(SPECULATIVE_PREFILL_TARGET_SELECTED_POSITIONS_TENSOR_NAME)
        {
            return Err("sparse target state tensor inventory is incomplete".to_owned());
        }
        Ok(Self {
            prompt_prefix_token_count,
            policy_identity: PersistentSpeculativePrefillPolicyIdentity::new(
                required_metadata(&parsed_header.metadata, "target_model_id")?.to_owned(),
                required_metadata(&parsed_header.metadata, "target_model_revision")?.to_owned(),
                required_metadata(&parsed_header.metadata, "drafter_model_id")?.to_owned(),
                required_metadata(&parsed_header.metadata, "drafter_model_revision")?.to_owned(),
                required_metadata(&parsed_header.metadata, "keep_percentage")?
                    .parse::<u32>()
                    .map_err(|_| "sparse target state keep_percentage is invalid".to_owned())?,
            ),
            tensor_names: parsed_header.tensor_views.into_keys().collect(),
        })
    }

    pub(crate) const fn prompt_prefix_token_count(&self) -> usize {
        self.prompt_prefix_token_count
    }

    pub(crate) const fn policy_identity(&self) -> &PersistentSpeculativePrefillPolicyIdentity {
        &self.policy_identity
    }

    pub(crate) fn tensor_names(&self) -> &[String] {
        &self.tensor_names
    }
}

#[cfg(feature = "direct-mlx")]
pub(crate) fn target_state_metadata_entries(
    target_state_contract: &PersistentSpeculativePrefillTargetStateContract,
    target_state_identity_hash: [u8; 32],
    prompt_prefix_token_count: usize,
) -> Vec<(&'static str, String)> {
    vec![
        (
            "format_version",
            PERSISTENT_SPECULATIVE_PREFILL_TARGET_STATE_FORMAT_VERSION.to_owned(),
        ),
        (
            "target_model_id",
            target_state_contract.target_model_id.clone(),
        ),
        (
            "target_model_revision",
            target_state_contract.target_model_revision.clone(),
        ),
        (
            "drafter_model_id",
            target_state_contract.drafter_model_id.clone(),
        ),
        (
            "drafter_model_revision",
            target_state_contract.drafter_model_revision.clone(),
        ),
        (
            "keep_percentage",
            target_state_contract.keep_percentage.to_string(),
        ),
        (
            "target_state_identity_sha256",
            hexadecimal_sha256(target_state_identity_hash),
        ),
        (
            "prompt_prefix_token_count",
            prompt_prefix_token_count.to_string(),
        ),
    ]
}

#[cfg(feature = "direct-mlx")]
fn require_metadata(
    metadata_entries: &std::collections::HashMap<String, String>,
    metadata_name: &'static str,
    expected_metadata_text: &str,
) -> Result<(), String> {
    let actual_metadata_text = metadata_entries
        .get(metadata_name)
        .ok_or_else(|| format!("sparse target state is missing {metadata_name}"))?;
    if actual_metadata_text != expected_metadata_text {
        return Err(format!(
            "sparse target state {metadata_name} does not match its contract"
        ));
    }
    Ok(())
}

#[cfg(feature = "direct-mlx")]
fn required_metadata<'a>(
    metadata_entries: &'a std::collections::HashMap<String, String>,
    metadata_name: &'static str,
) -> Result<&'a str, String> {
    metadata_entries
        .get(metadata_name)
        .map(String::as_str)
        .ok_or_else(|| format!("sparse target state is missing {metadata_name}"))
}

#[cfg(feature = "direct-mlx")]
fn hexadecimal_sha256(sha256_digest: [u8; 32]) -> String {
    let mut encoded_digest = String::with_capacity(64);
    for digest_byte in sha256_digest {
        use std::fmt::Write;
        let _ = write!(encoded_digest, "{digest_byte:02x}");
    }
    encoded_digest
}

/// Finds the longest strict prompt prefix whose selection-bound sparse target state exists.
#[must_use]
pub fn longest_reusable_speculative_prefill_target_prefix(
    target_state_contract: &PersistentSpeculativePrefillTargetStateContract,
    prompt_token_ids: &[u32],
    ordered_image_sha256_digests: &[[u8; 32]],
    mut has_target_state_identity: impl FnMut([u8; 32]) -> bool,
) -> Option<usize> {
    let mut target_state_identity_hasher =
        target_state_contract.target_state_identity_hasher(ordered_image_sha256_digests);
    let mut longest_reusable_prefix_token_count = None;
    for (prompt_token_position, prompt_token_id) in prompt_token_ids.iter().enumerate() {
        target_state_identity_hasher.update(prompt_token_id.to_le_bytes());
        let prompt_prefix_token_count = prompt_token_position.saturating_add(1);
        if prompt_prefix_token_count < prompt_token_ids.len()
            && has_target_state_identity(target_state_identity_hasher.clone().finalize().into())
        {
            longest_reusable_prefix_token_count = Some(prompt_prefix_token_count);
        }
    }
    longest_reusable_prefix_token_count
}

fn append_length_prefixed_bytes(identity_hasher: &mut Sha256, identity_component: &[u8]) {
    identity_hasher.update(
        u64::try_from(identity_component.len())
            .unwrap_or(u64::MAX)
            .to_le_bytes(),
    );
    identity_hasher.update(identity_component);
}
