//! Initial empty state for the sequential native component owner.

use super::*;

impl Flux2KleinMlxComponents {
    pub(in crate::flux2_klein) fn new(
        serving_model_id: String,
        model_directory: PathBuf,
        provenance: Flux2KleinArtifactProvenance,
        effective_mlx_memory_ceiling_bytes: usize,
        allocator_cache_memory_limit_bytes: usize,
        performance_attribution_enabled: bool,
        performance_attribution_log_path: PathBuf,
    ) -> Self {
        Self {
            serving_model_id,
            model_directory,
            provenance,
            effective_mlx_memory_ceiling_bytes,
            original_allocator_cache_memory_limit_bytes: allocator_cache_memory_limit_bytes,
            allocator_cache_memory_limit_bytes,
            performance_attribution_enabled,
            performance_attribution_log_path,
            performance_attribution_log: None,
            runtime: None,
            validated_artifact: None,
            residency_plan: None,
            transformer_geometry: None,
            request_attribution: None,
            request_id: None,
            request_seed: None,
            request_start_memory: None,
            dimensions: None,
            latent_layout: None,
            transformer_file: None,
            vae_file: None,
            text_conditioning_state: None,
            conditioning: None,
            transformer: None,
            forward_state: None,
            forward_step_index: None,
            denoising_step_started_at: None,
            latents: None,
            image_position_ids: None,
            text_position_ids: None,
            vae_decoder: None,
            vae_decode_state: None,
            decoded_rgb: None,
            post_cleanup_memory_telemetry: None,
        }
    }
}
