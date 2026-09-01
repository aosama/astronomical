//! Request-context admission traces for live diagnosis.
//!
//! A fitting model can still demote on RequestAdmission while every expert byte
//! remains in RAM. These fields exist so the next agent turn prints the exact
//! addends that crossed the ceiling, instead of reconstructing the formula from
//! code after the fact.

use crate::memory::{ContextAdmissionRequirements, MemoryAdmissionDecision};

pub(crate) fn log_generation_context_workspace_reservation(
    total_context_token_count: usize,
    prompt_token_count: usize,
    can_use_persistent_prompt_cache: bool,
    context_memory_reservation_bytes_per_token: usize,
    direct_publication_workspace_bytes: usize,
    restore_overlap_workspace_bytes: usize,
    prefill_activation_workspace_bytes: usize,
    complete_layer_scratch_bytes: usize,
    temporary_workspace_reservation_bytes: usize,
    additional_maximum_expert_page_reservation_bytes: usize,
) {
    tracing::info!(
        total_context_token_count,
        prompt_token_count,
        can_use_persistent_prompt_cache,
        context_memory_reservation_bytes_per_token,
        direct_publication_workspace_bytes,
        restore_overlap_workspace_bytes,
        prefill_activation_workspace_bytes,
        complete_layer_scratch_bytes,
        temporary_workspace_reservation_bytes,
        additional_maximum_expert_page_reservation_bytes,
        "composed generation-context workspace reservation"
    );
}

pub(crate) fn log_context_admission_projection(
    decision_stage: &'static str,
    context_token_count_requiring_reservation: usize,
    context_memory_reservation_bytes_per_token: usize,
    requirements: ContextAdmissionRequirements,
) {
    let projected_active_memory_bytes_overflowed =
        requirements.projected_active_memory_bytes().is_none();
    let projected_active_memory_bytes = requirements
        .projected_active_memory_bytes()
        .unwrap_or(usize::MAX);
    let projected_bytes_above_ceiling =
        projected_active_memory_bytes.saturating_sub(requirements.active_memory_ceiling_bytes);
    let admission_decision = requirements.decide();
    let admission_decision_name = admission_decision_name(admission_decision);
    tracing::info!(
        decision_stage,
        context_token_count_requiring_reservation,
        context_memory_reservation_bytes_per_token,
        current_active_memory_bytes = requirements.current_active_memory_bytes,
        context_growth_bytes = requirements.context_growth_bytes,
        expert_page_reservation_bytes = requirements.expert_page_reservation_bytes,
        temporary_workspace_bytes = requirements.temporary_workspace_bytes,
        retained_expert_payload_bytes = requirements.retained_expert_payload_bytes,
        active_memory_ceiling_bytes = requirements.active_memory_ceiling_bytes,
        complete_experts_are_resident = requirements.complete_experts_are_resident,
        projected_active_memory_bytes,
        projected_active_memory_bytes_overflowed,
        projected_bytes_above_ceiling,
        admission_decision_name,
        "context admission projection"
    );
}

fn admission_decision_name(admission_decision: MemoryAdmissionDecision) -> &'static str {
    match admission_decision {
        MemoryAdmissionDecision::Admit => "admit",
        MemoryAdmissionDecision::Reclaim { .. } => "reclaim",
        MemoryAdmissionDecision::DemoteCompleteResidency { .. } => "demote_complete_residency",
        MemoryAdmissionDecision::Reject { .. } => "reject",
    }
}
