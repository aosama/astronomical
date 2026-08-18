//! Resolves strict configured MTP pairings against independent discovery snapshots.

use astronomical_config::{DiscoveredModel, DiscoveredQwen3_5MtpDrafter, MtpPairingConfig};
use astronomical_ipc_protocol::WorkerMtpPairingConfiguration;

use crate::config_reload::ResolvedRuntimeConfigError;

pub(crate) fn resolve_mtp_pairings(
    configured_pairings: &[MtpPairingConfig],
    discovered_targets: &[DiscoveredModel],
    discovered_drafters: &[DiscoveredQwen3_5MtpDrafter],
) -> Result<Vec<WorkerMtpPairingConfiguration>, ResolvedRuntimeConfigError> {
    configured_pairings
        .iter()
        .map(|configured_pairing| {
            if !discovered_targets.iter().any(|discovered_target| {
                discovered_target.model_id == configured_pairing.target_model_id()
            }) {
                return Err(
                    ResolvedRuntimeConfigError::MtpPairingTargetModelNotDiscovered {
                        target_model_id: configured_pairing.target_model_id().to_owned(),
                    },
                );
            }
            let discovered_drafter = discovered_drafters.iter().find(|discovered_drafter| {
                discovered_drafter.model_id == configured_pairing.drafter_model_id()
            });
            Ok(WorkerMtpPairingConfiguration {
                target_model_id: configured_pairing.target_model_id().to_owned(),
                drafter_model_id: configured_pairing.drafter_model_id().to_owned(),
                drafter_model_directory: discovered_drafter
                    .map(|drafter| drafter.model_directory.clone()),
                discovered_drafter_revision: discovered_drafter
                    .map(|drafter| drafter.revision.clone()),
            })
        })
        .collect()
}
