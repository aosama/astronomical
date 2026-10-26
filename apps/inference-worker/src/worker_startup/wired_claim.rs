//! The worker's Metal wired-residency claim: how much memory MLX may keep
//! pinned for the worker lifetime, and the actual claim it issues at startup.

use astronomical_runtime_integration::maximum_recommended_gpu_working_set_size_bytes;
use mlx_c_rust::set_wired_limit;

/// The largest wired-residency budget MLX will grant for the machine.
///
/// MLX rejects claims above Metal's recommended working set (its
/// `set_wired_limit` throws `invalid_argument` above the recommendation),
/// so the recommendation is the absolute wiring ceiling even on machines
/// where the memory ceiling sits higher. The claim is sized to the machine
/// ceiling, not the configured serving ceiling: the wired limit only caps how
/// many allocated bytes may stay wired, and the serving ceiling already bounds
/// allocation, so a larger claim wires nothing extra. Sizing it to the
/// configured setting instead froze every later live raise at the startup
/// setting, because the claim lasts the worker lifetime and the menu stepper
/// offers nothing above the reported budget. The engine still clamps its
/// ceiling to the granted budget: a ceiling above it is the 2026-10-16 fatal
/// gap (issue #1297), where admission planned up to the 42,949,672,960-byte
/// machine ceiling while the driver could keep only 40,000,028,672 bytes
/// GPU-resident.
pub(crate) fn wired_residency_claim_budget_bytes(
    machine_mlx_memory_ceiling_bytes: usize,
    recommended_working_set_bytes: usize,
) -> usize {
    machine_mlx_memory_ceiling_bytes.min(recommended_working_set_bytes)
}

/// Claims Metal wired residency for the worker lifetime up to the machine
/// ceiling, capped at the system recommendation — the mlx-lm large-model
/// pattern — so that under memory pressure the operating system pages other
/// processes instead of the loaded model's weights. Returns the granted
/// budget the engine must clamp its ceiling to, or `None` when serving
/// continues without a wired pool (recommendation unreadable or the claim
/// rejected). A failed claim never blocks
/// serving: with no wired budget the allocations stay pageable, so the
/// ceiling keeps its original value; the kernel releases the claim when
/// the worker exits.
pub(crate) fn claim_wired_residency_up_to(
    machine_mlx_memory_ceiling_bytes: usize,
) -> Option<usize> {
    let recommended_working_set_bytes = match maximum_recommended_gpu_working_set_size_bytes() {
        Ok(recommended_working_set_bytes) => recommended_working_set_bytes,
        Err(recommendation_error) => {
            tracing::warn!(
                error = %recommendation_error,
                "could not read the recommended working set; wired claim skipped"
            );
            return None;
        }
    };
    let claim_budget_bytes = wired_residency_claim_budget_bytes(
        machine_mlx_memory_ceiling_bytes,
        recommended_working_set_bytes,
    );
    match set_wired_limit(claim_budget_bytes) {
        Ok(previous_wired_limit_bytes) => {
            tracing::info!(
                claimed_wired_limit_bytes = claim_budget_bytes,
                previous_wired_limit_bytes,
                machine_mlx_memory_ceiling_bytes,
                "claimed Metal wired residency for the worker lifetime"
            );
            Some(claim_budget_bytes)
        }
        Err(claim_error) => {
            tracing::warn!(
                error = %claim_error,
                "wired-residency claim failed; serving continues unprotected"
            );
            None
        }
    }
}
