//! The production engine choice as the assigned memory ceiling shrinks.
//!
//! 24 GB holds every expert, so startup selects the resident engine; 22 GB is
//! the first ceiling where startup selects the streaming engine; 21, 20, and
//! 19 GB page progressively more expert layers from SSD. Read top to bottom,
//! the cells show the cost of the engine swap and then the cost of each
//! gigabyte of paging, under the identical summarization workload.

use astronomical_ipc_protocol::WorkerDiagnosticQwen3_5EngineOverride;
use serial_test::serial;

use super::summarization_journey::{BYTES_PER_GIGABYTE, measure_summarization_throughput};
use crate::performance_throughput::support as throughput_support;

/// Every sweep ceiling, largest first, in GB.
const AUTOMATIC_ENGINE_SWEEP_CEILING_GIGABYTES: [u64; 5] = [24, 22, 21, 20, 19];

#[test]
fn should_keep_every_sweep_ceiling_below_the_hard_journey_cap_and_strictly_descending() {
    for ceiling_gigabytes in AUTOMATIC_ENGINE_SWEEP_CEILING_GIGABYTES {
        assert!(
            ceiling_gigabytes * BYTES_PER_GIGABYTE
                <= throughput_support::MAXIMUM_MODEL_MEMORY_BYTES,
            "the {ceiling_gigabytes} GB sweep ceiling must stay at or below the hard journey cap"
        );
    }
    assert!(
        AUTOMATIC_ENGINE_SWEEP_CEILING_GIGABYTES
            .windows(2)
            .all(|adjacent_ceilings| adjacent_ceilings[0] > adjacent_ceilings[1]),
        "sweep ceilings must strictly descend so no two cells measure the same ceiling"
    );
}

// Each cell reads its ceiling from the shared sweep table by position, so the
// ordering test above guards the exact ceilings the cells run.
macro_rules! automatic_engine_ceiling_cell {
    ($test_function_name:ident, $sweep_ceiling_index:literal, $ignore_reason:literal) => {
        #[test]
        #[ignore = $ignore_reason]
        #[serial]
        fn $test_function_name() {
            measure_summarization_throughput(
                AUTOMATIC_ENGINE_SWEEP_CEILING_GIGABYTES[$sweep_ceiling_index],
                WorkerDiagnosticQwen3_5EngineOverride::Automatic,
                None,
            );
        }
    };
}

// The roomy reference: every expert resident, no paging.
automatic_engine_ceiling_cell!(
    should_measure_automatic_engine_summarization_throughput_under_a_24gb_memory_ceiling,
    0,
    "loads the resident 35B sparse-MoE model and measures automatic-engine summarization throughput under a 24 GB memory ceiling"
);
// The first ceiling at which startup swaps to the streaming engine.
automatic_engine_ceiling_cell!(
    should_measure_automatic_engine_summarization_throughput_under_a_22gb_memory_ceiling,
    1,
    "loads the resident 35B sparse-MoE model and measures automatic-engine summarization throughput under a 22 GB memory ceiling"
);
// One expert layer no longer fits: the first ceiling with real paging.
automatic_engine_ceiling_cell!(
    should_measure_automatic_engine_summarization_throughput_under_a_21gb_memory_ceiling,
    2,
    "loads the resident 35B sparse-MoE model and measures automatic-engine summarization throughput under a 21 GB memory ceiling"
);
// Deeper paging: whether the decode handoff that seats every leftover-funded
// layer still leaves the next prefill room to run without shedding residency.
automatic_engine_ceiling_cell!(
    should_measure_automatic_engine_summarization_throughput_under_a_20gb_memory_ceiling,
    3,
    "loads the resident 35B sparse-MoE model and measures automatic-engine summarization throughput under a 20 GB memory ceiling"
);
automatic_engine_ceiling_cell!(
    should_measure_automatic_engine_summarization_throughput_under_a_19gb_memory_ceiling,
    4,
    "loads the resident 35B sparse-MoE model and measures automatic-engine summarization throughput under a 19 GB memory ceiling"
);
