use std::collections::BTreeSet;

use super::{Qwen3_5Config, Qwen3_5ShardIndex, qwen3_5_mtp_tensor_names};

/// Data-driven MTP capability discovered from the validated artifact inventory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Qwen3_5MtpArtifactCapability {
    /// The artifact contains no compatible MTP tensor inventory and serves target-only.
    TargetOnly,
    /// The artifact contains the supported MTP tensor inventory.
    MtpCapable {
        discovered_mtp_layer_count: usize,
        supported_mtp_draft_depth: usize,
        mtp_tensor_count: usize,
    },
}

impl Qwen3_5MtpArtifactCapability {
    const DISCOVERED_SUPPORTED_MTP_LAYER_COUNT: usize = 1;
    const SUPPORTED_MTP_DRAFT_DEPTH: usize = 1;

    /// Classifies MTP capability from shard-index tensor inventory only.
    #[must_use]
    pub fn from_shard_index(
        qwen3_5_config: &Qwen3_5Config,
        shard_index: &Qwen3_5ShardIndex,
    ) -> Self {
        let actual_mtp_tensor_names = shard_index
            .mtp_tensor_name_to_shard_file_name()
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        if actual_mtp_tensor_names.is_empty() {
            return Self::TargetOnly;
        }
        if qwen3_5_config.mtp_layer_count() != Self::DISCOVERED_SUPPORTED_MTP_LAYER_COUNT as u32 {
            // The current executor is depth-one. Keep a valid target model
            // usable when a future artifact contains a different predictor
            // depth instead of treating that optional package as corruption.
            return Self::TargetOnly;
        }
        let expected_mtp_tensor_names = qwen3_5_mtp_tensor_names(qwen3_5_config);
        if let Some(unexpected_tensor_name) = actual_mtp_tensor_names
            .difference(&expected_mtp_tensor_names)
            .next()
        {
            tracing::debug!(
                tensor_name = %unexpected_tensor_name,
                "ignoring an optional MTP inventory that is outside the supported depth-one graph"
            );
            return Self::TargetOnly;
        }
        if let Some(missing_tensor_name) = expected_mtp_tensor_names
            .difference(&actual_mtp_tensor_names)
            .next()
        {
            tracing::debug!(
                tensor_name = %missing_tensor_name,
                "ignoring an incomplete optional MTP inventory"
            );
            return Self::TargetOnly;
        }
        Self::MtpCapable {
            discovered_mtp_layer_count: Self::DISCOVERED_SUPPORTED_MTP_LAYER_COUNT,
            supported_mtp_draft_depth: Self::SUPPORTED_MTP_DRAFT_DEPTH,
            mtp_tensor_count: actual_mtp_tensor_names.len(),
        }
    }

    #[must_use]
    pub const fn is_mtp_capable(&self) -> bool {
        matches!(self, Self::MtpCapable { .. })
    }
}
