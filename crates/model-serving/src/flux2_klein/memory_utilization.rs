//! Image-lane ceiling-utilization owners (issue #510).
//!
//! The shared chat compose names chat owners: model-core slack, context
//! growth, activations, and unseated expert entitlement. Mapping image onto
//! those fields would lie: the sequential plan releases the text encoder and
//! transformer deliberately, so unloaded component weights are not a held
//! promise, and there are no expert or context owners at all. This module
//! composes the same identity contract - named owners plus residual plus
//! overrun reconstruct unused headroom - with image-truthful owners:
//!
//! - **Phase transient reserve** — the measured transient bytes the residency
//!   plan promises for the active phase (streaming load page, conditioning
//!   taps, latent state, denoising workspace, VAE workspace, host RGB, PNG,
//!   and base64 overlap). Charged only for the part transient work does not
//!   already occupy, exactly like the chat activation reserve, so bytes
//!   counted inside active memory are not promised twice.
//! - **Free headroom** — everything else. The plan releases weights between
//!   phases, so bytes beyond the phase promise are genuinely idle until the
//!   next phase streams its component; they read as Unused, not as a reserve.
//!
//! Unlike chat, the sequential plan does not partition the ceiling into a
//! complete owner set, so there is no unexplained term: the residue after the
//! phase promise is free by design and flows through the unused remainder.
//! What image does inherit from chat is the overrun term - a phase reserve
//! larger than the headroom surfaces instead of hiding.
//!
//! Resident component weights attribute active memory instead: the
//! transformer reports its resident payload exactly, the VAE decoder is
//! all-or-nothing, and the streamed text encoder stays unattributed because
//! its own phase reserve is what the plan promises for it.

use crate::memory::MemoryCeilingUtilization;

use super::Flux2KleinMemoryGeometry;

/// The sequential image phase a measurement instant belongs to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Flux2KleinMemoryPhase {
    /// Text encoder streaming and conditioning-tap production.
    TextConditioning,
    /// Transformer denoising over retained and streamed blocks.
    Denoising,
    /// Complete VAE decoder execution.
    VaeDecoding,
    /// RGB to PNG to base64 encoding overlap.
    Encoding,
    /// No request is active; transient work is released.
    Idle,
}

/// Transient bytes the residency plan promises for one phase.
///
/// Every term comes from artifact measurements in `Flux2KleinMemoryGeometry`,
/// never from a fixed constant, so the reserve adapts to any artifact and
/// any requested image dimensions.
#[must_use]
pub fn flux2_klein_phase_transient_reserve_bytes(
    phase: Flux2KleinMemoryPhase,
    geometry: &Flux2KleinMemoryGeometry,
) -> u64 {
    match phase {
        // Streaming keeps one materialized source page beside durable taps.
        Flux2KleinMemoryPhase::TextConditioning => geometry
            .largest_component_load_page_bytes
            .saturating_add(geometry.conditioning_bytes),
        // The plan's denoising peak keeps one streaming load page beside the
        // latent state and workspace, for blocks the ceiling did not retain.
        Flux2KleinMemoryPhase::Denoising => geometry
            .latent_state_bytes
            .saturating_add(geometry.denoising_workspace_bytes)
            .saturating_add(geometry.largest_component_load_page_bytes),
        // RGB is the handoff owner shared with encoding.
        Flux2KleinMemoryPhase::VaeDecoding => geometry
            .vae_workspace_bytes
            .saturating_add(geometry.host_rgb_bytes),
        Flux2KleinMemoryPhase::Encoding => geometry
            .host_rgb_bytes
            .saturating_add(geometry.maximum_png_bytes)
            .saturating_add(geometry.maximum_base64_bytes),
        Flux2KleinMemoryPhase::Idle => 0,
    }
}

/// Composes the image-lane split of one ceiling measurement.
///
/// `resident_attributed_bytes` counts component weights the engine knows are
/// resident at this instant (transformer resident payload, complete VAE
/// decoder payload, durable conditioning taps). Active bytes beyond that are
/// transient work living inside the phase reserve, so charging them against
/// the reserve keeps the identity exact instead of counting them twice.
#[must_use]
pub fn compose_flux2_klein_memory_ceiling_utilization(
    mlx_active_memory_ceiling_bytes: u64,
    active_memory_bytes: u64,
    resident_attributed_bytes: u64,
    phase: Flux2KleinMemoryPhase,
    geometry: &Flux2KleinMemoryGeometry,
) -> MemoryCeilingUtilization {
    let unused_headroom_bytes = mlx_active_memory_ceiling_bytes.saturating_sub(active_memory_bytes);
    let unattributed_active_bytes = active_memory_bytes.saturating_sub(resident_attributed_bytes);
    let phase_transient_reserve_bytes = flux2_klein_phase_transient_reserve_bytes(phase, geometry);
    let reserved_phase_transient_bytes =
        phase_transient_reserve_bytes.saturating_sub(unattributed_active_bytes);
    // A phase reserve larger than the headroom means an owner overran its
    // promise; surfacing it keeps the overrun visible instead of silent.
    // Everything below the promise is free by design, so the unexplained
    // term stays zero and the free bytes travel through the unused remainder.
    let owner_overrun_bytes = reserved_phase_transient_bytes.saturating_sub(unused_headroom_bytes);
    MemoryCeilingUtilization {
        mlx_active_memory_ceiling_bytes,
        active_memory_bytes,
        unused_headroom_bytes,
        reserved_model_core_slack_bytes: 0,
        reserved_context_growth_bytes: 0,
        reserved_activation_and_workspace_bytes: reserved_phase_transient_bytes,
        unseated_expert_entitlement_bytes: 0,
        speculative_draft_payload_bytes: 0,
        unexplained_headroom_bytes: 0,
        owner_overrun_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> Flux2KleinMemoryGeometry {
        Flux2KleinMemoryGeometry {
            text_encoder_payload_bytes: 1_000_000_000,
            transformer_payload_bytes: 4_000_000_000,
            transformer_block_payload_bytes: vec![100_000_000; 25],
            vae_payload_bytes: 500_000_000,
            largest_component_load_page_bytes: 300_000_000,
            conditioning_bytes: 200_000_000,
            latent_state_bytes: 50_000_000,
            denoising_workspace_bytes: 700_000_000,
            vae_workspace_bytes: 600_000_000,
            host_rgb_bytes: 100_000_000,
            maximum_png_bytes: 20_000_000,
            maximum_base64_bytes: 30_000_000,
        }
    }

    #[test]
    fn should_charge_the_denoising_reserve_minus_occupied_transients() {
        let geometry = geometry();
        // Transformer fully resident, 600 MB of transient work already active.
        let active_memory_bytes = 4_000_000_000 + 600_000_000;
        let utilization = compose_flux2_klein_memory_ceiling_utilization(
            9_000_000_000,
            active_memory_bytes,
            4_000_000_000,
            Flux2KleinMemoryPhase::Denoising,
            &geometry,
        );
        assert_eq!(
            utilization.unused_headroom_bytes,
            9_000_000_000 - active_memory_bytes
        );
        // The phase promise is latent state, denoising workspace, and one
        // streaming load page; the 600 MB of already-occupied transients is
        // charged against it.
        let denoising_reserve_bytes = 50_000_000 + 700_000_000 + 300_000_000;
        assert_eq!(
            utilization.reserved_activation_and_workspace_bytes,
            denoising_reserve_bytes - 600_000_000
        );
        assert!(utilization.is_fully_explained());
        assert_eq!(utilization.unexplained_headroom_bytes, 0);
        // Free headroom beyond the phase promise is genuine idle capacity.
        assert_eq!(
            utilization.unused_headroom_bytes,
            utilization.reserved_activation_and_workspace_bytes
                + (9_000_000_000 - 4_000_000_000 - denoising_reserve_bytes)
        );
    }

    #[test]
    fn should_read_idle_headroom_as_free_after_request_cleanup() {
        let geometry = geometry();
        let utilization = compose_flux2_klein_memory_ceiling_utilization(
            9_000_000_000,
            4_000_000_000,
            4_000_000_000,
            Flux2KleinMemoryPhase::Idle,
            &geometry,
        );
        assert_eq!(utilization.reserved_activation_and_workspace_bytes, 0);
        assert_eq!(utilization.unexplained_headroom_bytes, 0);
        assert_eq!(utilization.owner_overrun_bytes, 0);
        assert!(utilization.is_fully_explained());
    }

    #[test]
    fn should_charge_the_vae_workspace_reserve_during_decoding() {
        let geometry = geometry();
        // Complete VAE decoder resident with its workspace partially occupied.
        let active_memory_bytes = 500_000_000 + 300_000_000;
        let utilization = compose_flux2_klein_memory_ceiling_utilization(
            2_000_000_000,
            active_memory_bytes,
            500_000_000,
            Flux2KleinMemoryPhase::VaeDecoding,
            &geometry,
        );
        // The phase promise is VAE workspace plus the shared RGB handoff.
        assert_eq!(
            utilization.reserved_activation_and_workspace_bytes,
            600_000_000 + 100_000_000 - 300_000_000
        );
        assert_eq!(
            utilization.unused_headroom_bytes,
            2_000_000_000 - active_memory_bytes
        );
        assert!(utilization.is_fully_explained());
    }

    #[test]
    fn should_surface_a_phase_reserve_that_overruns_the_headroom() {
        let geometry = geometry();
        // Encoding promises RGB plus PNG plus base64, but the ceiling is
        // nearly consumed; the overrun must surface, not hide.
        let utilization = compose_flux2_klein_memory_ceiling_utilization(
            150_000_000,
            120_000_000,
            0,
            Flux2KleinMemoryPhase::Encoding,
            &geometry,
        );
        assert_eq!(utilization.unused_headroom_bytes, 30_000_000);
        assert_eq!(
            utilization.reserved_activation_and_workspace_bytes,
            30_000_000
        );
        assert_eq!(utilization.owner_overrun_bytes, 0);
        assert!(utilization.is_fully_explained());

        let tighter = compose_flux2_klein_memory_ceiling_utilization(
            140_000_000,
            120_000_000,
            0,
            Flux2KleinMemoryPhase::Encoding,
            &geometry,
        );
        assert_eq!(tighter.unused_headroom_bytes, 20_000_000);
        assert_eq!(
            tighter.owner_overrun_bytes,
            150_000_000 - 20_000_000 - 120_000_000
        );
        assert!(!tighter.is_fully_explained());
    }

    #[test]
    fn should_charge_streamed_text_encoder_work_against_its_own_phase_reserve() {
        let geometry = geometry();
        // During streamed conditioning nothing is durably attributed: the
        // streamed layer weights are transient work inside the streaming
        // reserve the plan promises for this phase.
        let active_memory_bytes = 900_000_000;
        let utilization = compose_flux2_klein_memory_ceiling_utilization(
            2_000_000_000,
            active_memory_bytes,
            0,
            Flux2KleinMemoryPhase::TextConditioning,
            &geometry,
        );
        // Occupied transients beyond the streaming reserve saturate the
        // charge at zero, mirroring the chat reserve algebra; the rest of the
        // headroom is free by design.
        assert_eq!(utilization.reserved_activation_and_workspace_bytes, 0);
        assert_eq!(
            utilization.unused_headroom_bytes,
            2_000_000_000 - active_memory_bytes
        );
        assert_eq!(utilization.unexplained_headroom_bytes, 0);
        assert!(utilization.is_fully_explained());
    }
}
