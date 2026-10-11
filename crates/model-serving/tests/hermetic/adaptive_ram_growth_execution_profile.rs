use astronomical_model_serving::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthExecutionProfile, AdaptiveRamGrowthGuard,
};

#[test]
fn should_reserve_transient_headroom_for_unseen_prefill_contexts() {
    let mut adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(1_000)
        .expect("a positive active-memory limit should create a guard");
    let observed_prefill_context =
        AdaptiveRamGrowthContext::prefill(128, 7, true, AdaptiveRamGrowthExecutionProfile::Paged);
    adaptive_ram_growth_guard.record_completed_growth_for_context(
        observed_prefill_context,
        true,
        0,
        0,
        200,
        0,
    );

    for independent_prefill_context in [
        AdaptiveRamGrowthContext::prefill(128, 8, true, AdaptiveRamGrowthExecutionProfile::Paged),
        AdaptiveRamGrowthContext::prefill(128, 7, false, AdaptiveRamGrowthExecutionProfile::Paged),
        AdaptiveRamGrowthContext::prefill(
            128,
            7,
            true,
            AdaptiveRamGrowthExecutionProfile::Resident,
        ),
    ] {
        assert_eq!(
            adaptive_ram_growth_guard
                .project_growth_for_context(independent_prefill_context, 500, 100, 0, 0)
                .expect("an independent context should project")
                .observed_transient_high_water_bytes(),
            200
        );
    }
}
