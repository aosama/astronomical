//! The policy crate's integration-lane entry: the same network-denied,
//! graphics-processor-free contract the other workspace packages honor, over
//! a representative slice of the pure arithmetic (the layering contract
//! guarantees nothing in this crate can reach the network or the MLX
//! bindings, so these tests run anywhere the integration lane runs).

use astronomical_memory_policy::budget::adaptive_growth::AdaptiveRamGrowthGuard;
use astronomical_memory_policy::phase::MemoryPhase;
use astronomical_memory_policy::serving_ceiling::machine_ceiling_from_physical_ram_bytes;

#[test]
fn should_run_the_memory_policy_integration_smoke_contract() {
    // One representative fact per family of pure arithmetic, enough to prove
    // the crate's test surface is alive inside the integration lane; the
    // exhaustive coverage lives in the lib unit tests (187 of them).
    assert_eq!(
        machine_ceiling_from_physical_ram_bytes(51_539_607_552),
        42_949_672_960
    );
    assert_eq!(MemoryPhase::Prefill, MemoryPhase::Prefill);
    let _ = AdaptiveRamGrowthGuard::new(1_000).expect("a positive limit should create a guard");
}
