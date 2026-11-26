//! Memory geometry of the installed sparse Qwen3.5 artifact, as the engine-selection
//! and live-ceiling journeys use it to pick a ceiling that holds the whole model.

use std::path::Path;

use model_serving::{
    CompleteResidencyHeadroomBoundary, ModelMemoryGeometry, Qwen3_5ArtifactValidator,
    model_memory_geometry_from_validated_artifact,
};

/// Journeys never ask a real model for more than this ceiling, so they stay
/// runnable on the 36 GB reference Mac.
pub(super) const MAXIMUM_REAL_MODEL_MEMORY_BYTES: u64 = 36_000_000_000;
pub(super) const MAXIMUM_OUTPUT_TOKEN_COUNT: u32 = 8;

pub(super) struct SparseArtifactResidencyGeometry {
    pub(super) complete_residency_with_headroom_bytes: u64,
    /// The measured geometry the memory governor (and therefore the request-time
    /// activation reserve) is composed from.
    pub(super) model_geometry: ModelMemoryGeometry,
}

pub(super) fn measure_sparse_artifact_residency_geometry(
    model_directory: &Path,
) -> SparseArtifactResidencyGeometry {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, MAXIMUM_OUTPUT_TOKEN_COUNT)
        .expect("the configured sparse artifact should validate");
    let (model_geometry, required_headroom_bytes) =
        model_memory_geometry_from_validated_artifact(&validated_artifact, model_directory)
            .expect("the configured sparse artifact should expose disk RAM geometry");
    let complete_residency_boundary = CompleteResidencyHeadroomBoundary::from_model_geometry(
        model_geometry,
        required_headroom_bytes,
    );
    SparseArtifactResidencyGeometry {
        complete_residency_with_headroom_bytes: complete_residency_boundary
            .static_complete_residency_bytes
            .saturating_add(complete_residency_boundary.required_headroom_bytes),
        model_geometry,
    }
}

pub(super) fn context_memory_reservation_bytes(
    model_directory: &Path,
    context_token_count: usize,
) -> u64 {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, MAXIMUM_OUTPUT_TOKEN_COUNT)
        .expect("the configured sparse artifact should validate");
    let reservation_bytes = validated_artifact
        .config()
        .context_memory_reservation_bytes(context_token_count)
        .expect("the acceptance prompt context should fit the model geometry");
    u64::try_from(reservation_bytes)
        .expect("the acceptance prompt context reservation should fit u64")
}

pub(super) fn artifact_payload_bytes(model_directory: &Path) -> u64 {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, MAXIMUM_OUTPUT_TOKEN_COUNT)
        .expect("the configured sparse artifact should validate");
    validated_artifact.total_payload_bytes()
}

/// Room these journeys leave for one prompt chunk's MLX working buffers when
/// they pick a limit that should hold the whole model. The engine
/// measures the real figure at load; three times the largest expert layer
/// has been ample for the models these journeys load.
pub(super) const fn prompt_chunk_mlx_working_buffer_allowance_bytes(
    model_geometry: &ModelMemoryGeometry,
) -> u64 {
    model_geometry
        .largest_complete_expert_layer_bytes
        .saturating_mul(3)
}
