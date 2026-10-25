//! The one summarization workload every engine-versus-ceiling cell serves, so
//! any throughput difference between cells comes from the ceiling or the
//! engine and never from the prompt.

use astronomical_ipc_protocol::{
    WorkerDecodeExpertCachePolicy, WorkerDiagnosticQwen3_5EngineOverride,
};

use crate::performance_throughput::historical_record::ThroughputJourneyKind;
use crate::performance_throughput::support::{self as throughput_support, ThroughputJourney};
use crate::support;

/// The short warmup instruction shared with the sibling throughput lanes.
const WARMUP_INPUT_INSTRUCTION: &str = "Continue the supplied Romeo and Juliet story above.";

/// The Romeo and Juliet source used for the bounded warmup prompt.
const WARMUP_ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../../fixtures/model_metrics_warmup_romeo_and_juliet.txt");

/// The warmup output cap: ~50 tokens spin up first-use JIT kernels so they
/// never inflate the measured prompt-processing rate.
const WARMUP_MAXIMUM_OUTPUT_TOKENS: u16 = 50;

/// The summarization instruction prepended to the measured Romeo and Juliet
/// source. Naming the target length keeps the summary filling the 300-token
/// output cap instead of stopping after a few sentences, the same trick the
/// canonical lanes use for their 500-token continuation cap.
const MEASURED_INPUT_INSTRUCTION: &str =
    "Summarize the supplied Romeo and Juliet story above in approximately 300 tokens.";

/// The ~5,000-token Romeo and Juliet source; with the instruction the full
/// input tokenizes to roughly 5,300 tokens.
const MEASURED_ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../../fixtures/model_metrics_5000_tokens_romeo_and_juliet.txt");

/// The measured output cap: 300 tokens, after which generation stops.
const MEASURED_MAXIMUM_OUTPUT_TOKENS: u16 = 300;

/// The sampling temperature in thousandths (1_000 = 1.0, unclamped), matching
/// the sibling lanes so the cells stay comparable with them.
const TEMPERATURE_THOUSANDTHS: u16 = 1_000;

/// Apple marketing gigabytes: 1 GB = 2^30 bytes, the convention every
/// user-facing memory value in this repo uses.
pub(super) const BYTES_PER_GIGABYTE: u64 = 1 << 30;

/// Serves the summarization workload once and verifies its measured shape.
///
/// Every cell carries the production probe kind, but only automatic-engine
/// runs reach the durable history: the throughput runner skips the append for
/// any forced engine or overridden prefill floor, so a diagnostic run is never
/// read back as what the production engine choice delivers at that ceiling.
pub(super) fn measure_summarization_throughput(
    assigned_ceiling_gigabytes: u64,
    engine_override: WorkerDiagnosticQwen3_5EngineOverride,
    resident_prefill_activation_floor_bytes: Option<u64>,
) {
    let mut journey = summarization_journey(
        ThroughputJourneyKind::ResidentMemoryCeilingProbe,
        assigned_ceiling_gigabytes,
    );
    journey.diagnostic_qwen3_5_engine_override = engine_override;
    journey.diagnostic_qwen3_5_resident_prefill_activation_floor_bytes =
        resident_prefill_activation_floor_bytes;
    let measured_throughput = throughput_support::run_journey_with_timeout(
        support::resident_sparse_moe_model_id(),
        journey,
    );
    assert_summarization_measurement_shape(&measured_throughput, assigned_ceiling_gigabytes);
}

/// Attribution stays off so every cell records clean, production-faithful
/// rates; diagnostics are opted into per run through the runner configuration.
fn summarization_journey(
    journey_kind: ThroughputJourneyKind,
    assigned_ceiling_gigabytes: u64,
) -> ThroughputJourney {
    let mut journey = ThroughputJourney::production_default(
        journey_kind,
        throughput_support::romeo_and_juliet_warmup_prompt(
            WARMUP_INPUT_INSTRUCTION,
            WARMUP_ROMEO_AND_JULIET_SOURCE,
        ),
        Vec::new(),
        WARMUP_MAXIMUM_OUTPUT_TOKENS,
        format!("{MEASURED_INPUT_INSTRUCTION}\n\n{MEASURED_ROMEO_AND_JULIET_SOURCE}"),
        Vec::new(),
        MEASURED_MAXIMUM_OUTPUT_TOKENS,
        TEMPERATURE_THOUSANDTHS,
    );
    journey.maximum_mlx_memory_bytes = Some(assigned_ceiling_gigabytes * BYTES_PER_GIGABYTE);
    journey.decode_expert_cache_policy = WorkerDecodeExpertCachePolicy::Baseline;
    journey.attribution_enabled = false;
    journey
}

/// Verifies the measured shape: a ~5,000-token Romeo and Juliet summarization
/// that filled (within the standard 10 percent tolerance) its 300-token output
/// cap, with zero persistent-prompt-cache reuse. The bounds are deliberately
/// wider than the fixture's nominal token count because the tokenizer's exact
/// count drifts with the instruction prefix.
fn assert_summarization_measurement_shape(
    measured_throughput: &throughput_support::ThroughputMeasurement,
    assigned_ceiling_gigabytes: u64,
) {
    assert!(
        (4_500..=5_800).contains(&measured_throughput.total_input_tokens),
        "the {assigned_ceiling_gigabytes} GB cell's measured Romeo and Juliet prompt should \
         contain roughly 5,000 input tokens, got {}",
        measured_throughput.total_input_tokens,
    );
    assert!(
        (270..=300).contains(&measured_throughput.total_output_tokens),
        "the {assigned_ceiling_gigabytes} GB cell's measured completion should generate 270 to \
         300 output tokens, got {}",
        measured_throughput.total_output_tokens,
    );
    assert_eq!(
        measured_throughput.cached_tokens, 0,
        "the {assigned_ceiling_gigabytes} GB cell's measured completion must not reuse the \
         persistent prompt cache",
    );
}
