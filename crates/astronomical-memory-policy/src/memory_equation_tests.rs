use crate::admission::context::ContextAdmissionRequirements;
use crate::budget::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthExecutionProfile, AdaptiveRamGrowthGuard,
    TransientWorkspaceReserve, TransientWorkspaceReserveSource,
};
use crate::forward_admission::{ResidentForwardAdmissionDecision, StreamingEngineRequirement};
use crate::memory_equation::MemoryEquation;
use crate::vocabulary::MemoryAdmissionDecision;

const ACTIVE_MEMORY_CEILING_BYTES: usize = 10_000;

#[test]
fn should_compose_stable_peak_and_recovery_memory_once() {
    let memory_equation = MemoryEquation::new(
        1_000,
        200,
        50,
        100,
        TransientWorkspaceReserve::analytic_floor(300, 1_024),
        ACTIVE_MEMORY_CEILING_BYTES,
    )
    .expect("the small memory equation should fit the platform range");

    assert_eq!(memory_equation.stable_projected_bytes(), 1_250);
    assert_eq!(memory_equation.peak_projected_bytes(), 1_650);
    assert_eq!(memory_equation.recovery_projected_bytes(), 2_050);
    assert_eq!(memory_equation.allowed_active_memory_bytes(), 10_100);
    assert_eq!(memory_equation.stable_shortfall_bytes(), 0);
    assert_eq!(memory_equation.peak_shortfall_bytes(), 0);
    assert_eq!(memory_equation.recovery_reserve_shortfall_bytes(), 0);
    assert!(memory_equation.fits_stable_and_peak_limits());
}

#[test]
fn should_fail_closed_when_projection_arithmetic_overflows() {
    assert_eq!(
        MemoryEquation::new(
            usize::MAX,
            1,
            0,
            0,
            TransientWorkspaceReserve::global_maximum(0, 1),
            ACTIVE_MEMORY_CEILING_BYTES,
        ),
        None
    );
}

#[test]
fn should_apply_the_peak_allowance_at_its_exact_boundary() {
    let memory_equation = MemoryEquation::new(
        ACTIVE_MEMORY_CEILING_BYTES,
        0,
        0,
        ACTIVE_MEMORY_CEILING_BYTES / 100,
        TransientWorkspaceReserve::global_maximum(0, 1),
        ACTIVE_MEMORY_CEILING_BYTES,
    )
    .expect("the exact boundary should be representable");

    assert_eq!(
        memory_equation.peak_projected_bytes(),
        memory_equation.allowed_active_memory_bytes()
    );
    assert!(memory_equation.fits_stable_and_peak_limits());
    assert!(!memory_equation.fits_request_memory_limit());
}

#[test]
fn should_match_request_and_resident_guard_verdicts_when_the_analytic_floor_is_the_same_workspace()
{
    // Fixed ownership so only the workspace byte count moves the projections.
    let context_growth_bytes = 500;
    let routed_expert_page_reservation_bytes = 250;

    let request_admission_verdict =
        |current_active_memory_bytes: usize, temporary_workspace_bytes: usize| -> (bool, usize) {
            let request_admission_requirements = ContextAdmissionRequirements {
                current_active_memory_bytes,
                context_growth_bytes,
                expert_page_reservation_bytes: routed_expert_page_reservation_bytes,
                temporary_workspace_bytes,
                retained_expert_payload_bytes: 0,
                active_memory_ceiling_bytes: ACTIVE_MEMORY_CEILING_BYTES,
                complete_experts_are_resident: false,
            };
            let request_admission_decision = request_admission_requirements.decide();
            (
                matches!(request_admission_decision, MemoryAdmissionDecision::Admit),
                request_admission_requirements
                    .projected_active_memory_bytes()
                    .expect("the same inputs should project"),
            )
        };
    let resident_guard_verdict = |current_active_memory_bytes: usize,
                                  unobserved_scope_transient_floor_bytes: usize|
     -> (bool, usize) {
        let adaptive_growth_guard =
            AdaptiveRamGrowthGuard::new(ACTIVE_MEMORY_CEILING_BYTES).expect("valid ceiling");
        let forward_projection = adaptive_growth_guard
            .project_growth_for_context(
                AdaptiveRamGrowthContext::prefill(
                    2_048,
                    0,
                    false,
                    AdaptiveRamGrowthExecutionProfile::Resident,
                ),
                current_active_memory_bytes,
                context_growth_bytes,
                routed_expert_page_reservation_bytes,
                0,
                unobserved_scope_transient_floor_bytes,
            )
            .expect("the forward equation should fit the platform range");
        assert_eq!(
            forward_projection.observed_transient_high_water_bytes(),
            unobserved_scope_transient_floor_bytes
        );
        (
            ResidentForwardAdmissionDecision::for_resident_forward(
                &forward_projection,
                Some(2_048),
            )
            .is_admitted(),
            forward_projection.peak_projected_bytes(),
        )
    };

    // Both projections compose the same workspace when it sits below the strict
    // ceiling, and both engines admit.
    assert_eq!(
        request_admission_verdict(4_000, 1_000),
        resident_guard_verdict(4_000, 1_000),
    );
    assert_eq!(request_admission_verdict(4_000, 1_000), (true, 5_750));

    // The exact strict-ceiling byte admits under both verdicts: request
    // admission has zero shortfall and the forward guard still fits the peak.
    let strict_ceiling_boundary_peak_bytes = ACTIVE_MEMORY_CEILING_BYTES;
    let strict_ceiling_boundary_workspace_bytes = strict_ceiling_boundary_peak_bytes
        - 4_000
        - context_growth_bytes
        - routed_expert_page_reservation_bytes;
    assert_eq!(
        request_admission_verdict(4_000, strict_ceiling_boundary_workspace_bytes),
        (true, strict_ceiling_boundary_peak_bytes),
    );
    assert_eq!(
        resident_guard_verdict(4_000, strict_ceiling_boundary_workspace_bytes),
        (true, strict_ceiling_boundary_peak_bytes),
    );

    // One byte over the strict ceiling is the deliberate divergence point.
    // Request admission promises the workspace before any forward exists, so it
    // holds the strict ceiling and rejects. The resident forward guard knows the
    // concrete forward shape and may spend its one-percent transient allowance
    // (10_000 -> 10_100), so the same byte count stays admissible for the
    // forward. Asserting the divergence keeps the two ceilings from silently
    // converging later.
    assert_eq!(
        request_admission_verdict(4_000, strict_ceiling_boundary_workspace_bytes + 1),
        (false, 10_001),
    );
    assert_eq!(
        resident_guard_verdict(4_000, strict_ceiling_boundary_workspace_bytes + 1),
        (true, 10_001),
    );

    // Above the one-percent allowance both verdicts reject the same peak bytes.
    assert_eq!(
        request_admission_verdict(4_000, strict_ceiling_boundary_workspace_bytes + 101),
        (false, 10_101),
    );
    assert_eq!(
        resident_guard_verdict(4_000, strict_ceiling_boundary_workspace_bytes + 101),
        (false, 10_101),
    );

    // A stable shortfall rejects under both verdicts, and the resident guard
    // names the typed stable cause rather than a transient one.
    let (request_admitted_at_stable_shortfall, stable_shortfall_peak_bytes) =
        request_admission_verdict(10_500, 0);
    assert!(!request_admitted_at_stable_shortfall);
    let adaptive_growth_guard =
        AdaptiveRamGrowthGuard::new(ACTIVE_MEMORY_CEILING_BYTES).expect("valid ceiling");
    let stable_shortfall_projection = adaptive_growth_guard
        .project_growth_for_context(
            AdaptiveRamGrowthContext::prefill(
                2_048,
                0,
                false,
                AdaptiveRamGrowthExecutionProfile::Resident,
            ),
            10_500,
            context_growth_bytes,
            routed_expert_page_reservation_bytes,
            0,
            0,
        )
        .expect("the stable-shortfall equation should fit the platform range");
    assert_eq!(
        ResidentForwardAdmissionDecision::for_resident_forward(
            &stable_shortfall_projection,
            Some(2_048),
        ),
        ResidentForwardAdmissionDecision::RequireStreamingEngine {
            cause: StreamingEngineRequirement::StableShortfall {
                shortfall_bytes: 1_250
            },
        }
    );
    assert_eq!(
        stable_shortfall_projection.peak_projected_bytes(),
        stable_shortfall_peak_bytes
    );
}

#[test]
fn should_compose_request_admission_without_learned_transient_evidence() {
    let memory_equation =
        MemoryEquation::for_request_admission(1_000, 200, 50, 100, ACTIVE_MEMORY_CEILING_BYTES)
            .expect("the request equation should fit the platform range");

    // The reserve contributes zero bytes, exactly as before, but its source now
    // states that no learned evidence exists instead of borrowing the
    // global-maximum label.
    assert_eq!(memory_equation.peak_projected_bytes(), 1_350);
    assert_eq!(
        memory_equation
            .transient_workspace_reserve()
            .reserve_bytes(),
        0
    );
    assert_eq!(
        memory_equation.transient_workspace_reserve().source(),
        TransientWorkspaceReserveSource::NoLearnedReserve
    );
}

#[test]
fn should_report_a_typed_stable_shortfall_for_a_resident_forward() {
    let adaptive_growth_guard =
        AdaptiveRamGrowthGuard::new(ACTIVE_MEMORY_CEILING_BYTES).expect("valid ceiling");
    let forward_projection = adaptive_growth_guard
        .project_growth_for_context(
            AdaptiveRamGrowthContext::decode(1, AdaptiveRamGrowthExecutionProfile::Resident),
            ACTIVE_MEMORY_CEILING_BYTES + 1,
            0,
            0,
            0,
            0,
        )
        .expect("the stable shortfall should be representable");

    assert_eq!(
        ResidentForwardAdmissionDecision::for_resident_forward(&forward_projection, None),
        ResidentForwardAdmissionDecision::RequireStreamingEngine {
            cause: StreamingEngineRequirement::StableShortfall { shortfall_bytes: 1 },
        }
    );
}
