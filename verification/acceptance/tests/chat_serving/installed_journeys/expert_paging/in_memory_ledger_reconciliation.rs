//! The in-memory engine's memory ledger against what MLX reports.
//!
//! The engine tells its memory governor what every forward held and
//! needed. If its books and MLX disagree by much, some allocation has no
//! owner, and the question the engine asks before each forward would be
//! answered from wrong figures.

use model_serving::Qwen3_5ResidentEngine;
use model_serving::Qwen3_5ResidentEngineTestControls;

/// The most the ledger may fail to account for, as a share of the limit.
const MAXIMUM_UNACCOUNTED_PERMILLE_OF_LIMIT: u64 = 20;

/// Asserts that, across every forward the loaded engine has run, what MLX
/// held beyond the ledger's accounts stayed within 2% of the limit, and
/// that the MLX working buffers of a forward was measured at load.
pub(super) async fn assert_the_in_memory_ledger_matches_mlx(
    qwen3_5_engine: &Qwen3_5ResidentEngine,
) {
    let ledger_reading = qwen3_5_engine
        .memory_ledger_reading_for_tests()
        .await
        .expect("the engine should answer for its ledger")
        .expect("an engine that has served requests has measured forwards");
    let limit_bytes = ledger_reading.accounts.limit_bytes;
    let allowed_unaccounted_bytes = limit_bytes / 1_000 * MAXIMUM_UNACCOUNTED_PERMILLE_OF_LIMIT;
    eprintln!(
        "[in-memory-ledger] measured_steps={} largest_unaccounted_gb={:.3} largest_overcounted_gb={:.3} allowed_gb={:.3} probed_mlx_working_buffers_gb={:.3} largest_prompt_step_rise_gb={:.3} largest_reply_step_rise_gb={:.3}",
        ledger_reading.observed_step_count,
        ledger_reading.largest_unaccounted_bytes as f64 / 1_073_741_824.0,
        ledger_reading.largest_overcounted_bytes as f64 / 1_073_741_824.0,
        allowed_unaccounted_bytes as f64 / 1_073_741_824.0,
        ledger_reading.probed_mlx_working_buffer_bytes.unwrap_or(0) as f64 / 1_073_741_824.0,
        ledger_reading.largest_reading_step_peak_beyond_held_bytes as f64 / 1_073_741_824.0,
        ledger_reading.largest_writing_step_peak_beyond_held_bytes as f64 / 1_073_741_824.0,
    );
    assert!(
        ledger_reading.observed_step_count > 0,
        "no forward was reported to the memory governor"
    );
    assert!(
        ledger_reading.probed_mlx_working_buffer_bytes.is_some(),
        "the MLX working buffers of a forward was not measured at load"
    );
    assert!(
        ledger_reading.largest_unaccounted_bytes <= allowed_unaccounted_bytes,
        "MLX held {} bytes the ledger does not account for, more than the {} allowed under a {} byte limit",
        ledger_reading.largest_unaccounted_bytes,
        allowed_unaccounted_bytes,
        limit_bytes
    );
}
