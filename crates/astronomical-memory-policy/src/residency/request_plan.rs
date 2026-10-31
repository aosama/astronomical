//! One request-stable answer to pin, stream, and shrink.
//!
//! Leftover RAM arithmetic may be recomputed on every chunk. This owner freezes
//! the Prefill contract at request open: pinned complete layers stay pinned,
//! streamed layers stay streamed, and a capacity failure may only shrink the
//! pin set. Generation handoff discards the contract and replans from leftover.

use super::{
    CurrentExpertLayerResidency, ExpertLayerGeometry, ExpertLayerResidencyTarget,
    ExpertResidencyPlan, RetainedExpertPageClass,
};
use crate::phase::MemoryPhase;

/// Role of one sparse layer for the rest of this request's Prefill.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestExpertLayerRole {
    /// Keep or promote the complete layer until the request ends or shrinks.
    PinnedComplete,
    /// Read when a forward needs it and keep what was read when leftover RAM
    /// admits the payload. Never planned into the pin set, but a mandatory
    /// read already paid the whole-layer cost, so discarding it would make
    /// every later chunk re-stream the same bytes from SSD.
    Streamed,
    /// A capacity recovery evicted this layer's retention. It must stream
    /// operation-local for the rest of the request: re-admitting what a
    /// recovery just released lets every later attempt re-seat, re-fail, and
    /// re-evict the same table — the thrash loop behind the 2026-10-15
    /// paged-prefill stall (issue #1278).
    DemotedByCapacityFailure,
}

/// Request-scoped expert contract. Family crates enact it; they do not replan it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestExpertResidency {
    layer_roles: Vec<RequestExpertLayerRole>,
}

impl RequestExpertResidency {
    /// Pins every complete-layer target from the opening leftover plan.
    #[must_use]
    pub fn open_prefill(candidate_plan: &ExpertResidencyPlan) -> Self {
        let mut layer_roles =
            vec![RequestExpertLayerRole::Streamed; candidate_plan.layer_targets.len()];
        for layer_index in &candidate_plan.complete_layer_targets {
            if let Some(layer_role) = layer_roles.get_mut(*layer_index) {
                *layer_role = RequestExpertLayerRole::PinnedComplete;
            }
        }
        Self { layer_roles }
    }

    #[must_use]
    pub fn layer_role(&self, layer_index: usize) -> Option<RequestExpertLayerRole> {
        self.layer_roles.get(layer_index).copied()
    }

    #[must_use]
    pub fn pinned_complete_layer_indexes(&self) -> Vec<usize> {
        self.layer_roles
            .iter()
            .enumerate()
            .filter_map(|(layer_index, layer_role)| {
                matches!(layer_role, RequestExpertLayerRole::PinnedComplete).then_some(layer_index)
            })
            .collect()
    }

    /// Drops pinned complete layers from the tail until `required_reclamation_bytes` is covered.
    ///
    /// Those layers are demoted beyond re-admission for the rest of Prefill: a
    /// later leftover plan must neither pin them again nor admit them on read,
    /// because the bytes a recovery reclaimed return to the exact pressure
    /// that evicted them the moment a retry re-seats the layer.
    #[must_use]
    pub fn shrink_after_capacity_failure(
        &self,
        required_reclamation_bytes: u64,
        layer_geometries: &[ExpertLayerGeometry],
    ) -> Self {
        if required_reclamation_bytes == 0 {
            return self.clone();
        }
        let mut layer_roles = self.layer_roles.clone();
        let mut remaining_reclamation_bytes = required_reclamation_bytes;
        for layer_index in (0..layer_roles.len()).rev() {
            if remaining_reclamation_bytes == 0 {
                break;
            }
            if layer_roles[layer_index] != RequestExpertLayerRole::PinnedComplete {
                continue;
            }
            let complete_layer_payload_bytes = layer_geometries
                .get(layer_index)
                .map(|geometry| geometry.complete_layer_payload_bytes)
                .unwrap_or(0);
            layer_roles[layer_index] = RequestExpertLayerRole::DemotedByCapacityFailure;
            remaining_reclamation_bytes =
                remaining_reclamation_bytes.saturating_sub(complete_layer_payload_bytes);
        }
        Self { layer_roles }
    }

    /// Rewrites a leftover Prefill plan so it cannot promote or release against this contract.
    ///
    /// Streamed-role layers keep what a mandatory read materialized only while
    /// the composed leftover budget still has room after every pinned layer's
    /// claim; a streamed layer the budget cannot hold resolves to
    /// operation-local streaming so the plan never authorizes an adoption
    /// offer it budgets zero bytes for (issue #1278).
    #[must_use]
    pub fn stabilize_prefill_plan(
        &self,
        mut candidate_plan: ExpertResidencyPlan,
        current_residencies: &[CurrentExpertLayerResidency],
        layer_geometries: &[ExpertLayerGeometry],
    ) -> ExpertResidencyPlan {
        let layer_is_currently_complete =
            currently_complete_layer_flags(self.layer_roles.len(), current_residencies);
        if candidate_plan.layer_targets.len() < self.layer_roles.len() {
            // Extend-only: a candidate plan longer than the roles carries
            // planner targets for layers beyond the contract's span, and
            // truncating it would silently drop those layers' dispositions.
            candidate_plan.layer_targets.resize(
                self.layer_roles.len(),
                ExpertLayerResidencyTarget::StreamOperationLocal,
            );
        }
        let PrefillFundingState {
            pin_claim_is_funded,
            stream_admission_headroom_bytes,
            expected_preserved_bytes,
        } = self.prefill_funding_state(
            &layer_is_currently_complete,
            current_residencies,
            layer_geometries,
            candidate_plan.retained_expert_ceiling_bytes,
        );
        let mut stream_admission_headroom_bytes = stream_admission_headroom_bytes;
        let mut maximum_new_retained_bytes = 0_u64;
        for (layer_index, layer_role) in self.layer_roles.iter().enumerate() {
            candidate_plan.layer_targets[layer_index] = stabilized_layer_target(
                *layer_role,
                layer_index,
                layer_is_currently_complete[layer_index],
                pin_claim_is_funded[layer_index],
                layer_geometries,
                &mut stream_admission_headroom_bytes,
                &mut maximum_new_retained_bytes,
            );
        }
        candidate_plan.complete_layer_targets = self
            .layer_roles
            .iter()
            .enumerate()
            .filter_map(|(layer_index, layer_role)| {
                matches!(layer_role, RequestExpertLayerRole::PinnedComplete)
                    .then_some(layer_index)
                    .filter(|funded_layer_index| pin_claim_is_funded[*funded_layer_index])
            })
            .collect();
        candidate_plan.reserved_routed_overlay_bytes = 0;
        candidate_plan.expected_preserved_bytes = expected_preserved_bytes;
        candidate_plan.maximum_new_retained_bytes = maximum_new_retained_bytes;
        candidate_plan.is_low_budget_partial_mode = false;
        candidate_plan
    }

    /// Charges every preserved seated layer against the composed ceiling and
    /// funds the remaining unseated pin claims, in that order. The
    /// stream-admission headroom is what remains of the composed ceiling
    /// after every claimable pinned layer: each pin, seated or not, reserves
    /// its full complete payload, because a seated pin's table holds exactly
    /// that payload and an unseated pin claims it on promotion. A seated
    /// pin's claim always holds — its bytes are already resident — but an
    /// unseated pin only claims while the ceiling can fund it: pins frozen at
    /// request open must not keep claiming past a ceiling that context growth
    /// shrank, because the over-claim saturates headroom to zero while naming
    /// promotion targets the budget cannot honor (the 2026-10-23
    /// under-utilization flight: 34 claims of 855,638,016 bytes against a
    /// 28,938,405,632-byte ceiling left 26.95 GB unused and streamed 389 GB
    /// from SSD). Unfunded pins fall through to streamed handling; funding is
    /// spent in ascending layer order, so the loss of funding lands on the
    /// tail pins first, mirroring the contract's shrink order.
    ///
    /// Every seated layer the plan preserves — pinned or streamed role — is
    /// charged here, before anything new is funded. A seated streamed layer
    /// is preserved without a fresh admission, so leaving its bytes out of
    /// the headroom let every chunk admit one more complete layer past the
    /// ceiling (the 2026-10-25 cached-suffix flight: a 12-layer contract grew
    /// to 14 seated layers and 14.5 GB of retained experts against an 11.16
    /// GB ceiling, then the 2,048-token chunk ran out of active memory twice
    /// and the request failed).
    fn prefill_funding_state(
        &self,
        layer_is_currently_complete: &[bool],
        current_residencies: &[CurrentExpertLayerResidency],
        layer_geometries: &[ExpertLayerGeometry],
        retained_expert_ceiling_bytes: u64,
    ) -> PrefillFundingState {
        let mut pin_claim_is_funded = vec![false; self.layer_roles.len()];
        let mut stream_admission_headroom_bytes = retained_expert_ceiling_bytes;
        let mut expected_preserved_bytes = 0_u64;
        for (layer_index, layer_role) in self.layer_roles.iter().enumerate() {
            let is_preserved_seated_layer = matches!(
                layer_role,
                RequestExpertLayerRole::PinnedComplete | RequestExpertLayerRole::Streamed
            ) && layer_is_currently_complete[layer_index];
            if !is_preserved_seated_layer {
                continue;
            }
            if *layer_role == RequestExpertLayerRole::PinnedComplete {
                pin_claim_is_funded[layer_index] = true;
            }
            let seated_complete_claim_bytes = layer_geometries
                .get(layer_index)
                .map(|geometry| geometry.complete_layer_payload_bytes)
                .unwrap_or(0);
            stream_admission_headroom_bytes =
                stream_admission_headroom_bytes.saturating_sub(seated_complete_claim_bytes);
            if let Some(current_residency) = current_residencies
                .iter()
                .find(|residency| residency.layer_index == layer_index)
            {
                expected_preserved_bytes =
                    expected_preserved_bytes.saturating_add(current_residency.payload_bytes);
            }
        }
        for (layer_index, layer_role) in self.layer_roles.iter().enumerate() {
            if *layer_role != RequestExpertLayerRole::PinnedComplete
                || layer_is_currently_complete[layer_index]
            {
                continue;
            }
            let pinned_complete_claim_bytes = layer_geometries
                .get(layer_index)
                .map(|geometry| geometry.complete_layer_payload_bytes)
                .unwrap_or(0);
            if pinned_complete_claim_bytes <= stream_admission_headroom_bytes {
                pin_claim_is_funded[layer_index] = true;
                stream_admission_headroom_bytes =
                    stream_admission_headroom_bytes.saturating_sub(pinned_complete_claim_bytes);
            }
        }
        PrefillFundingState {
            pin_claim_is_funded,
            stream_admission_headroom_bytes,
            expected_preserved_bytes,
        }
    }
}

/// Marks each layer currently resident as a complete layer. A residency
/// entry naming a layer outside the contract's span is ignored.
fn currently_complete_layer_flags(
    layer_count: usize,
    current_residencies: &[CurrentExpertLayerResidency],
) -> Vec<bool> {
    let mut layer_is_currently_complete = vec![false; layer_count];
    for current_residency in current_residencies {
        if current_residency.layer_index < layer_is_currently_complete.len()
            && current_residency.class == RetainedExpertPageClass::StableCompleteLayer
        {
            layer_is_currently_complete[current_residency.layer_index] = true;
        }
    }
    layer_is_currently_complete
}

/// The funding facts the target-assignment pass reads: which pin claims the
/// ceiling funds, the headroom left for streamed-layer admissions, and the
/// preserved bytes the plan expects to keep.
struct PrefillFundingState {
    pin_claim_is_funded: Vec<bool>,
    stream_admission_headroom_bytes: u64,
    expected_preserved_bytes: u64,
}

/// Decides one layer's stabilized target. Seated layers preserve; funded
/// unseated pins promote on mandatory read; a streamed role whose whole
/// expert set is already seated preserves too — the mandatory read paid the
/// complete-layer cost in an earlier chunk, and evicting it would make every
/// later chunk re-stream the same bytes (the 2026-10-24 flight: 31 streamed
/// layers froze the allowance while 19.6 GB of the paid ceiling sat unused
/// and 90 GB were re-streamed). Remaining streamed and unfunded-pin layers
/// admit only while the leftover budget fits their complete payload, else
/// they stay operation-local. A capacity-recovery demotion may never re-seat:
/// re-admitting what a recovery just released lets every later attempt
/// re-seat, re-fail, and re-evict the same table (issue #1278).
fn stabilized_layer_target(
    layer_role: RequestExpertLayerRole,
    layer_index: usize,
    layer_is_currently_complete: bool,
    pin_claim_is_funded: bool,
    layer_geometries: &[ExpertLayerGeometry],
    stream_admission_headroom_bytes: &mut u64,
    maximum_new_retained_bytes: &mut u64,
) -> ExpertLayerResidencyTarget {
    match layer_role {
        RequestExpertLayerRole::PinnedComplete if layer_is_currently_complete => {
            ExpertLayerResidencyTarget::PreserveComplete
        }
        RequestExpertLayerRole::PinnedComplete if pin_claim_is_funded => {
            ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead
        }
        RequestExpertLayerRole::Streamed if layer_is_currently_complete => {
            ExpertLayerResidencyTarget::PreserveComplete
        }
        RequestExpertLayerRole::Streamed | RequestExpertLayerRole::PinnedComplete => {
            let complete_layer_payload_bytes = layer_geometries
                .get(layer_index)
                .map(|geometry| geometry.complete_layer_payload_bytes)
                .unwrap_or(u64::MAX);
            if complete_layer_payload_bytes <= *stream_admission_headroom_bytes {
                *stream_admission_headroom_bytes =
                    stream_admission_headroom_bytes.saturating_sub(complete_layer_payload_bytes);
                *maximum_new_retained_bytes =
                    maximum_new_retained_bytes.saturating_add(complete_layer_payload_bytes);
                // A streamed layer's mandatory read pays the complete-layer
                // cost the moment the chunk routes nearly every expert, so
                // keeping what was read is pure reuse while the leftover
                // budget admits the payload.
                ExpertLayerResidencyTarget::AdmitCompleteOnMandatoryRead
            } else {
                ExpertLayerResidencyTarget::StreamOperationLocal
            }
        }
        RequestExpertLayerRole::DemotedByCapacityFailure => {
            ExpertLayerResidencyTarget::StreamOperationLocal
        }
    }
}

/// Counts the layers that are actually resident as complete layers right now.
/// The request contract's pin count is intent; this is occupancy. Publishing
/// the two as one number made the 2026-10-24 flight log claim
/// `pinned_complete_layer_count=34` while nine layers were resident.
#[must_use]
pub fn seated_complete_layer_count(current_residencies: &[CurrentExpertLayerResidency]) -> usize {
    current_residencies
        .iter()
        .filter(|current_residency| {
            current_residency.class == RetainedExpertPageClass::StableCompleteLayer
        })
        .count()
}

/// Floors the retained-page ceiling at complete layers already in RAM.
///
/// Leftover arithmetic can tighten after a chunk or a decode token because a
/// learned context reserve grew — including when lazy seated pages materialize
/// and get mis-attributed as activation. Evicting a seated complete layer to
/// match that smaller number throws away a page this request already paid to
/// read, then every later token streams it again. Real capacity failure still
/// shrinks through `shrink_after_capacity_failure` or request-pressure deficit.
#[must_use]
pub const fn retained_complete_layer_ceiling_after_prefill_budget_refresh(
    leftover_expert_budget_bytes: u64,
    current_complete_layer_payload_bytes: u64,
) -> u64 {
    if leftover_expert_budget_bytes > current_complete_layer_payload_bytes {
        leftover_expert_budget_bytes
    } else {
        current_complete_layer_payload_bytes
    }
}

/// Floors the retained-page ceiling at every resident expert page, warm
/// tables included.
///
/// A learned budget that tightens between requests (context-reserve growth)
/// must not proactively discard routing knowledge any more than it discards
/// seated complete layers: the hot-expert tables cost disk re-reads to rebuild
/// and the request-pressure paths already reclaim them exactly when a real
/// forward needs the bytes. Use the complete-layer variant instead where the
/// caller reclaims under pressure — warm tables are the release valve there.
#[must_use]
pub const fn retained_resident_ceiling_after_budget_refresh(
    leftover_expert_budget_bytes: u64,
    current_resident_payload_bytes: u64,
) -> u64 {
    if leftover_expert_budget_bytes > current_resident_payload_bytes {
        leftover_expert_budget_bytes
    } else {
        current_resident_payload_bytes
    }
}

/// Binds leftover packing to the request contract for the current phase.
///
/// Prefill opens once, then only shrinks. Generation, decode, and idle discard
/// the Prefill contract and keep the leftover candidate unchanged.
#[must_use]
pub fn publish_request_stable_residency_plan(
    phase: MemoryPhase,
    existing_request_residency: Option<&RequestExpertResidency>,
    candidate_plan: ExpertResidencyPlan,
    current_residencies: &[CurrentExpertLayerResidency],
    released_complete_payload_bytes: u64,
    layer_geometries: &[ExpertLayerGeometry],
) -> (Option<RequestExpertResidency>, ExpertResidencyPlan) {
    match phase {
        MemoryPhase::Prefill => {
            let mut request_residency = existing_request_residency
                .cloned()
                .unwrap_or_else(|| RequestExpertResidency::open_prefill(&candidate_plan));
            if existing_request_residency.is_some() && released_complete_payload_bytes > 0 {
                request_residency = request_residency.shrink_after_capacity_failure(
                    released_complete_payload_bytes,
                    layer_geometries,
                );
            }
            let stabilized_plan = request_residency.stabilize_prefill_plan(
                candidate_plan,
                current_residencies,
                layer_geometries,
            );
            (Some(request_residency), stabilized_plan)
        }
        MemoryPhase::GenerationPreparation | MemoryPhase::Decode | MemoryPhase::Idle => {
            (None, candidate_plan)
        }
    }
}
