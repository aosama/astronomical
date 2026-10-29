//! Unit contracts for the pure paged-forward rejection message: the three
//! shapes keyed on plan satisfiability and the previous reclamation pass.

use astronomical_memory_policy::reclamation::ExpertReclamationPlan;

use super::{PagedForwardProjectionNumbers, paged_rejection_reason};

/// A satisfiable plan: retained payload covers every boundary deficit, so no
/// unresolved shortfall remains.
fn satisfiable_plan() -> ExpertReclamationPlan {
    ExpertReclamationPlan::for_projected_memory(
        /* stable_projected_bytes */ 1_000, /* peak_projected_bytes */ 1_200,
        /* recovery_projected_bytes */ 1_400, /* stable_memory_ceiling_bytes */ 2_000,
        /* transient_memory_ceiling_bytes */ 2_000,
        /* retained_expert_payload_bytes */ 4_096,
    )
}

/// An unsatisfiable plan: the recovery projection exceeds its ceiling by more
/// bytes than retained experts can release.
fn unsatisfiable_plan() -> ExpertReclamationPlan {
    ExpertReclamationPlan::for_projected_memory(
        /* stable_projected_bytes */ 1_000, /* peak_projected_bytes */ 1_200,
        /* recovery_projected_bytes */ 3_400, /* stable_memory_ceiling_bytes */ 2_000,
        /* transient_memory_ceiling_bytes */ 2_000,
        /* retained_expert_payload_bytes */ 1_000,
    )
}

fn projection_numbers() -> PagedForwardProjectionNumbers {
    PagedForwardProjectionNumbers {
        stable_projected_bytes: 1_000,
        peak_projected_bytes: 1_200,
        recovery_projected_bytes: 1_400,
        active_memory_ceiling_bytes: 2_000,
        allowed_active_memory_bytes: 2_400,
    }
}

#[test]
fn should_quote_the_unresolved_shortfall_when_the_plan_cannot_satisfy_every_boundary() {
    let reason = paged_rejection_reason(
        projection_numbers(),
        &unsatisfiable_plan(),
        /* previous_pass_released_pages */ None,
    );
    // recovery deficit 1_400 over retained 1_000 leaves 400 unresolved.
    assert_eq!(
        reason,
        "adaptive RAM growth exceeds reclaimable expert capacity by 400 bytes"
    );
}

#[test]
fn should_quote_the_paging_unavailable_shape_after_a_pass_released_nothing() {
    let reason = paged_rejection_reason(
        projection_numbers(),
        &satisfiable_plan(),
        /* previous_pass_released_pages */ Some(false),
    );
    assert_eq!(
        reason,
        "adaptive RAM growth rejected: stable projection of 1000 bytes, peak projection of 1200 bytes, and recovery projection of 1400 bytes do not fit C=2000 bytes and P=2400 bytes while retained expert paging is unavailable"
    );
}

#[test]
fn should_quote_the_post_reclamation_shape_after_a_releasing_pass() {
    // Both a fresh loop (None) and a releasing pass (Some(true)) fall through
    // to the same post-reclamation shape: the plan is satisfiable, so the
    // remaining message describes projections still above a boundary.
    for previous_pass_released_pages in [None, Some(true)] {
        let reason = paged_rejection_reason(
            projection_numbers(),
            &satisfiable_plan(),
            previous_pass_released_pages,
        );
        assert_eq!(
            reason,
            "adaptive RAM growth rejected: stable projection of 1000 bytes, peak projection of 1200 bytes, or recovery projection of 1400 bytes remains above C=2000 bytes or P=2400 bytes after retained-expert reclamation"
        );
    }
}
