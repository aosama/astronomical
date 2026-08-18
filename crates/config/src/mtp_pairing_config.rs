use std::collections::{BTreeMap, BTreeSet};

use super::AstronomicalConfigError;
use super::config_file::MtpPairingConfigFile;

/// A validated target-to-standalone-MTP-drafter pairing.
///
/// This owner carries the trimmed, non-empty target and drafter model IDs
/// plus a directed-graph invariant that each target appears in at most one
/// pairing. The same drafter may be referenced by multiple targets; each
/// loaded target still performs an independent deep compatibility decision.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MtpPairingConfig {
    target_model_id: String,
    drafter_model_id: String,
}

impl MtpPairingConfig {
    /// Creates a validated pairing from pre-trimmed, non-empty identifiers.
    #[must_use]
    pub const fn new(target_model_id: String, drafter_model_id: String) -> Self {
        Self {
            target_model_id,
            drafter_model_id,
        }
    }

    /// Returns the exact target model identity bound to this pairing.
    #[must_use]
    pub fn target_model_id(&self) -> &str {
        &self.target_model_id
    }

    /// Returns the exact drafter model identity bound to this pairing.
    #[must_use]
    pub fn drafter_model_id(&self) -> &str {
        &self.drafter_model_id
    }
}

/// Validates a collection of parsed pairing objects into zero-or-more
/// `MtpPairingConfig` values and rejects duplicate, self-referential, or
/// conflicting, or cyclic declarations with bounded, actionable diagnostics.
///
/// Invariants:
///   - A target may appear in at most one pairing.
///   - Exact duplicate declarations (same target and same drafter pair)
///     are rejected.
///   - One target mapped to different drafters is rejected.
///   - The same drafter may be reused by multiple targets.
pub(crate) fn resolve_mtp_pairings(
    parsed_pairings: &[MtpPairingConfigFile],
) -> Result<Vec<MtpPairingConfig>, AstronomicalConfigError> {
    let mut resolved = Vec::with_capacity(parsed_pairings.len());
    let mut target_to_drafter = BTreeMap::new();

    for parsed_pairing in parsed_pairings {
        let target_model_id = parsed_pairing.target_model_id.trim().to_owned();
        let drafter_model_id = parsed_pairing.drafter_model_id.trim().to_owned();

        if target_model_id.is_empty() {
            return Err(AstronomicalConfigError::MtpPairingTargetModelIdMustNotBeEmpty);
        }
        if drafter_model_id.is_empty() {
            return Err(AstronomicalConfigError::MtpPairingDrafterModelIdMustNotBeEmpty);
        }

        if target_model_id == drafter_model_id {
            return Err(AstronomicalConfigError::MtpPairingSelfReference { target_model_id });
        }

        match target_to_drafter.get(&target_model_id) {
            None => {
                target_to_drafter.insert(target_model_id.clone(), drafter_model_id.clone());
                resolved.push(MtpPairingConfig::new(target_model_id, drafter_model_id));
            }
            Some(existing_drafter) => {
                if existing_drafter == &drafter_model_id {
                    // Exact duplicate declaration is an error.
                    return Err(AstronomicalConfigError::MtpPairingDuplicateTarget {
                        target_model_id,
                    });
                } else {
                    return Err(
                        AstronomicalConfigError::MtpPairingConflictingTargetMapping {
                            target_model_id,
                            drafter_model_id_a: existing_drafter.clone(),
                            drafter_model_id_b: drafter_model_id,
                        },
                    );
                }
            }
        }
    }

    reject_pairing_cycles(&target_to_drafter)?;
    Ok(resolved)
}

fn reject_pairing_cycles(
    target_to_drafter: &BTreeMap<String, String>,
) -> Result<(), AstronomicalConfigError> {
    for starting_target_model_id in target_to_drafter.keys() {
        let mut current_model_id = starting_target_model_id.as_str();
        let mut visited_model_ids = BTreeSet::new();
        while let Some(drafter_model_id) = target_to_drafter.get(current_model_id) {
            if !visited_model_ids.insert(current_model_id.to_owned()) {
                return Err(AstronomicalConfigError::MtpPairingCycle {
                    model_id: current_model_id.to_owned(),
                });
            }
            current_model_id = drafter_model_id;
        }
    }
    Ok(())
}
