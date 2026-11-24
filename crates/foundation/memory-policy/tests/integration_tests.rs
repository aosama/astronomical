//! The policy crate's integration-lane entry: the same network-denied,
//! graphics-processor-free contract the other workspace packages honor, over
//! a representative slice of the pure arithmetic (the layering contract
//! guarantees nothing in this crate can reach the network or the MLX
//! bindings, so these tests run anywhere the integration lane runs).

use memory_policy::ledger::{MemoryGovernor, ModelMemoryShape};
use memory_policy::phase::MemoryPhase;
use memory_policy::serving_ceiling::machine_ceiling_from_physical_ram_bytes;

#[test]
fn should_run_the_memory_policy_integration_smoke_contract() {
    // One representative fact per family of pure arithmetic, enough to prove
    // the crate's test surface is alive inside the integration lane; the
    // exhaustive coverage lives in the lib unit tests (187 of them).
    // The whole of physical memory is the machine ceiling (#1902).
    assert_eq!(
        machine_ceiling_from_physical_ram_bytes(51_539_607_552),
        51_539_607_552
    );
    assert_eq!(MemoryPhase::Prefill, MemoryPhase::Prefill);
    let everything_streamable = ModelMemoryShape {
        core_weight_bytes: 0,
        streamable_weight_bytes: 4_000,
        reading_prompt_stream_slot_bytes: 0,
        writing_reply_stream_slot_bytes: 0,
    };
    let memory_governor =
        MemoryGovernor::new(1_000, everything_streamable).expect("a positive limit opens");
    assert!(memory_governor.kept_weight_capacity().bytes() <= 1_000);
}
