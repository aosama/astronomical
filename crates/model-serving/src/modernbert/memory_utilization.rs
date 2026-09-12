//! Embeddings-lane ceiling-utilization owners (issue #510).
//!
//! Embedding inference is one synchronous encoder forward pass with no
//! experts, no context state, and no sequential release plan, so the chat
//! owner vocabulary would invent promises nobody made. The honest owners are
//! the same shape the image lane uses under the shared identity contract:
//!
//! - **Transient reserve** — the forward workspace the engine has actually
//!   observed, measured as the MLX peak minus the current active bytes.
//!   Evidence-driven, never a fixed constant: an idle engine carries the
//!   high-water of its last forward as the promise for the next one, and a
//!   mid-forward engine carries the partial peak above current usage.
//! - **Free headroom** — everything else, free by design; the embeddings
//!   engine does not partition its ceiling, so there is no unexplained term
//!   and the free bytes travel through the unused remainder.
//! - **Overrun** — inherited unchanged: a reserve larger than the headroom
//!   surfaces instead of hiding.
//!
//! Resident weights attribute active memory instead: the whole safetensors
//! payload is the model core, and active bytes beyond it are transient work
//! charged against the reserve so they are never counted twice.

use crate::memory::MemoryCeilingUtilization;

/// Composes the embeddings-lane split of one ceiling measurement.
///
/// `weights_payload_bytes` is the resident safetensors payload; active bytes
/// beyond it are transient forward work. `observed_transient_high_water_bytes`
/// is the measured peak above the current active bytes at this instant.
#[must_use]
pub fn compose_modernbert_memory_ceiling_utilization(
    mlx_active_memory_ceiling_bytes: u64,
    active_memory_bytes: u64,
    weights_payload_bytes: u64,
    observed_transient_high_water_bytes: u64,
) -> MemoryCeilingUtilization {
    let unused_headroom_bytes = mlx_active_memory_ceiling_bytes.saturating_sub(active_memory_bytes);
    let unattributed_active_bytes = active_memory_bytes.saturating_sub(weights_payload_bytes);
    let reserved_transient_bytes =
        observed_transient_high_water_bytes.saturating_sub(unattributed_active_bytes);
    // A reserve larger than the headroom means the forward workspace overran
    // its promise; surfacing it keeps the overrun visible instead of silent.
    let owner_overrun_bytes = reserved_transient_bytes.saturating_sub(unused_headroom_bytes);
    MemoryCeilingUtilization {
        mlx_active_memory_ceiling_bytes,
        active_memory_bytes,
        unused_headroom_bytes,
        reserved_model_core_slack_bytes: 0,
        reserved_context_growth_bytes: 0,
        reserved_activation_and_workspace_bytes: reserved_transient_bytes,
        unseated_expert_entitlement_bytes: 0,
        speculative_draft_payload_bytes: 0,
        unexplained_headroom_bytes: 0,
        owner_overrun_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIGABYTE: u64 = 1_000_000_000;

    #[test]
    fn should_charge_the_observed_transient_high_water_minus_occupied_transients() {
        // 400 MB of weights resident, 100 MB of forward transients active,
        // and the allocator peak proves 250 MB of transient high-water.
        let active_memory_bytes = 500_000_000;
        let utilization = compose_modernbert_memory_ceiling_utilization(
            2 * GIGABYTE,
            active_memory_bytes,
            400_000_000,
            250_000_000,
        );
        assert_eq!(
            utilization.unused_headroom_bytes,
            2 * GIGABYTE - active_memory_bytes
        );
        assert_eq!(
            utilization.reserved_activation_and_workspace_bytes,
            250_000_000 - 100_000_000
        );
        assert!(utilization.is_fully_explained());
        assert_eq!(utilization.unexplained_headroom_bytes, 0);
    }

    #[test]
    fn should_read_idle_headroom_beyond_the_high_water_as_free() {
        // After cleanup the transients are released, so the whole high-water
        // is promised and the rest of the headroom is genuinely idle.
        let utilization = compose_modernbert_memory_ceiling_utilization(
            2 * GIGABYTE,
            400_000_000,
            400_000_000,
            250_000_000,
        );
        assert_eq!(
            utilization.reserved_activation_and_workspace_bytes,
            250_000_000
        );
        assert_eq!(
            utilization.unused_headroom_bytes,
            250_000_000 + (2 * GIGABYTE - 400_000_000 - 250_000_000)
        );
        assert_eq!(utilization.unexplained_headroom_bytes, 0);
        assert!(utilization.is_fully_explained());
    }

    #[test]
    fn should_surface_a_reserve_that_overruns_the_headroom() {
        let utilization = compose_modernbert_memory_ceiling_utilization(
            500_000_000,
            400_000_000,
            400_000_000,
            150_000_000,
        );
        assert_eq!(utilization.unused_headroom_bytes, 100_000_000);
        // The charged reserve is not clamped to the headroom; the excess
        // surfaces as the overrun instead.
        assert_eq!(
            utilization.reserved_activation_and_workspace_bytes,
            150_000_000
        );
        assert_eq!(utilization.owner_overrun_bytes, 50_000_000);
        assert!(!utilization.is_fully_explained());
    }
}
