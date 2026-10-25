//! Resident-model memory-ceiling comparison probe: the same 5,000-token Romeo
//! and Juliet summarization journey served twice, once under a 24 GB assigned
//! MLX memory ceiling and once under a 22 GB ceiling.
//!
//! This lane is a directed comparison probe, not the canonical text series:
//! the repo's standard throughput journeys send >=10,000 input tokens and
//! acquire 500 output tokens, while this probe deliberately measures the
//! user-directed 5,000-input/300-output shape so both ceilings can be compared
//! on an identical, smaller workload inside the journey budget. It therefore
//! records under its own `ResidentMemoryCeilingProbe` journey kind, keeping
//! the canonical text history series unpolluted by the smaller shape.
//!
//! Both cells measure the resident 35B sparse-MoE model (the
//! `resident_sparse_moe` registry role) over the daemon IPC boundary with the
//! SSD (persistent prompt) cache disabled and 2,048-token prefill chunks, and
//! report the usual server-attributed prompt-processing (prefill) and decode
//! tokens-per-second. The interesting signal is the delta between the two
//! cells: the tighter 22 GB ceiling forces the worker to fit the same request
//! into less wired memory, and the cost of that squeeze shows up directly in
//! the two rates.
//!
//! Gigabytes here follow the Apple marketing convention used across the repo:
//! 1 GB = 1,073,741,824 bytes (2^30).
//!
//! This is a laptop-only journey: it loads real weights into wired GPU memory,
//! so it is `#[ignore]`d and not wired into CI. Invoke the release worker
//! build first, then each cell exactly:
//! - `cargo build --release --bin astronomical-inference-worker`
//! - `cargo run -p xtask -- test-run --package astronomical-acceptance --features performance_throughput --test performance_throughput --release --filter resident_memory_ceiling_probe --ignored`
//! or use the `cargo test-resident-memory-ceiling-probe` alias which runs both
//! cells back to back.

use astronomical_ipc_protocol::WorkerDecodeExpertCachePolicy;
use serial_test::serial;

use crate::performance_throughput::historical_record::ThroughputJourneyKind;
use crate::performance_throughput::support::{self as throughput_support, ThroughputJourney};
use crate::support;

/// The short warmup instruction shared with the sibling throughput lanes.
const WARMUP_INPUT_INSTRUCTION: &str = "Continue the supplied Romeo and Juliet story above.";

/// The Romeo and Juliet source used for the bounded warmup prompt.
const WARMUP_ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_warmup_romeo_and_juliet.txt");

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
    include_str!("../../fixtures/model_metrics_5000_tokens_romeo_and_juliet.txt");

/// The measured output cap: 300 tokens, after which generation stops.
const MEASURED_MAXIMUM_OUTPUT_TOKENS: u16 = 300;

/// The sampling temperature in thousandths (1_000 = 1.0, unclamped), matching
/// the sibling lanes so the cells stay comparable with them.
const TEMPERATURE_THOUSANDTHS: u16 = 1_000;

/// Apple marketing gigabytes: 1 GB = 2^30 bytes, the convention every
/// user-facing memory value in this repo uses.
const BYTES_PER_GIGABYTE: u64 = 1 << 30;

/// The roomier cell's assigned ceiling, in GB.
const PROBE_CEILING_ROOMY_GIGABYTES: u64 = 24;

/// The tighter cell's assigned ceiling, in GB.
const PROBE_CEILING_TIGHT_GIGABYTES: u64 = 22;

/// Both probe ceilings must sit strictly below the harness's hard 36 GB
/// journey cap (enforced again at worker launch) and stay exactly 2 GB apart,
/// so a cell cannot silently drift into the cap or into its sibling.
#[test]
fn should_keep_both_probe_ceilings_below_the_hard_journey_cap_and_two_gigabytes_apart() {
    let roomy_ceiling_bytes = PROBE_CEILING_ROOMY_GIGABYTES * BYTES_PER_GIGABYTE;
    let tight_ceiling_bytes = PROBE_CEILING_TIGHT_GIGABYTES * BYTES_PER_GIGABYTE;
    assert!(
        roomy_ceiling_bytes <= throughput_support::MAXIMUM_MODEL_MEMORY_BYTES,
        "the roomy {PROBE_CEILING_ROOMY_GIGABYTES} GB probe ceiling must stay at or below the hard journey cap"
    );
    assert!(
        roomy_ceiling_bytes.saturating_sub(tight_ceiling_bytes) == 2 * BYTES_PER_GIGABYTE,
        "the probe ceilings must stay exactly 2 GB apart"
    );
    assert!(
        tight_ceiling_bytes < roomy_ceiling_bytes,
        "the tight probe ceiling must be the smaller of the two cells"
    );
}

/// The 24 GB cell: the resident sparse-MoE model summarizes the ~5,000-token
/// Romeo and Juliet prompt into 300 tokens under a 24 GB assigned MLX memory
/// ceiling, reporting prompt-processing and decode tokens-per-second.
#[test]
#[ignore = "loads the resident 35B sparse-MoE model and measures summarization throughput under a 24 GB memory ceiling"]
#[serial]
fn should_measure_resident_sparse_moe_summarization_throughput_under_a_24gb_memory_ceiling() {
    run_resident_memory_ceiling_probe(PROBE_CEILING_ROOMY_GIGABYTES);
}

/// The 22 GB cell: the identical journey under a 22 GB assigned MLX memory
/// ceiling, so the pair isolates the throughput cost of the tighter squeeze.
#[test]
#[ignore = "loads the resident 35B sparse-MoE model and measures summarization throughput under a 22 GB memory ceiling"]
#[serial]
fn should_measure_resident_sparse_moe_summarization_throughput_under_a_22gb_memory_ceiling() {
    run_resident_memory_ceiling_probe(PROBE_CEILING_TIGHT_GIGABYTES);
}

/// Drives one probe cell and verifies its measured shape. The journey keeps
/// attribution off so both cells record clean, production-faithful rates into
/// the durable history under the `ResidentMemoryCeilingProbe` kind.
fn run_resident_memory_ceiling_probe(assigned_ceiling_gigabytes: u64) {
    let journey = resident_summarization_journey(assigned_ceiling_gigabytes);
    let measured_throughput = throughput_support::run_journey_with_timeout(
        support::resident_sparse_moe_model_id(),
        journey,
    );
    assert_probe_measurement_shape(&measured_throughput, assigned_ceiling_gigabytes);
}

fn resident_summarization_journey(assigned_ceiling_gigabytes: u64) -> ThroughputJourney {
    let mut journey = ThroughputJourney::production_default(
        ThroughputJourneyKind::ResidentMemoryCeilingProbe,
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

/// Verifies the probe's measured shape: a ~5,000-token Romeo and Juliet
/// summarization that filled (within the standard 10 percent tolerance) its
/// 300-token output cap, with zero persistent-prompt-cache reuse. The bounds
/// are deliberately wider than the fixture's nominal token count because the
/// tokenizer's exact count drifts with the instruction prefix.
fn assert_probe_measurement_shape(
    measured_throughput: &throughput_support::ThroughputMeasurement,
    assigned_ceiling_gigabytes: u64,
) {
    assert!(
        (4_500..=5_800).contains(&measured_throughput.total_input_tokens),
        "the {assigned_ceiling_gigabytes} GB probe's measured Romeo and Juliet prompt should \
         contain roughly 5,000 input tokens, got {}",
        measured_throughput.total_input_tokens,
    );
    assert!(
        (270..=300).contains(&measured_throughput.total_output_tokens),
        "the {assigned_ceiling_gigabytes} GB probe's measured completion should generate 270 to \
         300 output tokens, got {}",
        measured_throughput.total_output_tokens,
    );
    assert_eq!(
        measured_throughput.cached_tokens, 0,
        "the {assigned_ceiling_gigabytes} GB probe's measured completion must not reuse the \
         persistent prompt cache",
    );
}
