//! Cells that force one engine so the engine's cost and the ceiling's cost
//! can be read separately.
//!
//! The forced streaming engine at 24 and 32 GB has every expert seated and
//! reads nothing from disk, so any gap to the 24 GB resident cell is the
//! streaming engine's own overhead; closing that gap is what lets tight
//! ceilings run near resident speed. The forced resident engine at 22 GB shows
//! what the tighter ceiling alone costs when the engine does not change.
//!
//! These cells print their throughput record and keep their worker evidence,
//! but the runner never appends them to the durable history: the record
//! carries no engine identity, so a forced line would be indistinguishable
//! from an automatic one at the same ceiling. Compare forced cells from the
//! printed record or the evidence directory of the same run.

use astronomical_ipc_protocol::WorkerDiagnosticQwen3_5EngineOverride;
use serial_test::serial;

use super::summarization_journey::measure_summarization_throughput;

macro_rules! forced_engine_cell {
    ($test_function_name:ident, $ceiling_gigabytes:literal, $engine_override:ident) => {
        #[test]
        #[ignore = concat!(
                                    "loads the resident 35B sparse-MoE model with the ",
                                    stringify!($engine_override),
                                    " engine forced under a ",
                                    stringify!($ceiling_gigabytes),
                                    " GB memory ceiling to separate engine cost from ceiling cost"
                                )]
        #[serial]
        fn $test_function_name() {
            measure_summarization_throughput(
                $ceiling_gigabytes,
                WorkerDiagnosticQwen3_5EngineOverride::$engine_override,
                None,
            );
        }
    };
}

forced_engine_cell!(
    should_measure_forced_streaming_engine_summarization_throughput_under_a_24gb_memory_ceiling,
    24,
    Streaming
);
forced_engine_cell!(
    should_measure_forced_streaming_engine_summarization_throughput_under_a_32gb_memory_ceiling,
    32,
    Streaming
);
forced_engine_cell!(
    should_measure_forced_resident_engine_summarization_throughput_under_a_22gb_memory_ceiling,
    22,
    Resident
);

/// The forced resident engine at 22 GB with the static three-layer prefill
/// activation floor removed, so admission relies only on learned activation
/// evidence. It answers whether the resident engine serves this shape inside
/// the ceiling once the floor stops rejecting it.
#[test]
#[ignore = "loads the resident 35B sparse-MoE model with the Resident engine forced and the static prefill activation floor removed under a 22 GB memory ceiling"]
#[serial]
fn should_measure_forced_resident_engine_without_static_prefill_floor_under_a_22gb_memory_ceiling()
{
    measure_summarization_throughput(22, WorkerDiagnosticQwen3_5EngineOverride::Resident, Some(0));
}
