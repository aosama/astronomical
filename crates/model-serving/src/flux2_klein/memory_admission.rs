//! Sequential unified-memory admission for conditioning, denoising, and decoding.

use thiserror::Error;

use crate::memory::{MlxRamBudget, MlxRamBudgetModelGeometry};

/// Validated payload and transient byte owners supplied by artifact/runtime measurements.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Flux2KleinMemoryGeometry {
    pub text_encoder_payload_bytes: u64,
    pub transformer_payload_bytes: u64,
    pub transformer_block_payload_bytes: Vec<u64>,
    pub vae_payload_bytes: u64,
    pub largest_component_load_page_bytes: u64,
    pub conditioning_bytes: u64,
    pub latent_state_bytes: u64,
    pub denoising_workspace_bytes: u64,
    pub vae_workspace_bytes: u64,
    pub host_rgb_bytes: u64,
    pub maximum_png_bytes: u64,
    pub maximum_base64_bytes: u64,
}

/// Component strategy selected without changing BF16 checkpoint precision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Flux2KleinResidencyMode {
    Complete,
    Streamed,
}

/// A sequential plan consumed directly by the future concrete engine.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Flux2KleinResidencyPlan {
    text_encoder_mode: Flux2KleinResidencyMode,
    retained_transformer_block_indices: Vec<usize>,
    vae_mode: Flux2KleinResidencyMode,
    conditioning_peak_bytes: u64,
    denoising_peak_bytes: u64,
    decoding_peak_bytes: u64,
    encoding_peak_bytes: u64,
    peak_required_bytes: u64,
    minimum_mlx_memory_ceiling_bytes: u64,
}

impl Flux2KleinResidencyPlan {
    pub const fn text_encoder_mode(&self) -> Flux2KleinResidencyMode {
        self.text_encoder_mode
    }
    pub fn retained_transformer_block_count(&self) -> usize {
        self.retained_transformer_block_indices.len()
    }
    pub fn retained_transformer_block_indices(&self) -> &[usize] {
        &self.retained_transformer_block_indices
    }
    pub const fn vae_mode(&self) -> Flux2KleinResidencyMode {
        self.vae_mode
    }
    pub const fn conditioning_peak_bytes(&self) -> u64 {
        self.conditioning_peak_bytes
    }
    pub const fn denoising_peak_bytes(&self) -> u64 {
        self.denoising_peak_bytes
    }
    pub const fn decoding_peak_bytes(&self) -> u64 {
        self.decoding_peak_bytes
    }
    pub const fn encoding_peak_bytes(&self) -> u64 {
        self.encoding_peak_bytes
    }
    pub const fn peak_required_bytes(&self) -> u64 {
        self.peak_required_bytes
    }
    pub const fn minimum_mlx_memory_ceiling_bytes(&self) -> u64 {
        self.minimum_mlx_memory_ceiling_bytes
    }
    pub const fn releases_text_encoder_before_transformer(&self) -> bool {
        true
    }
    pub const fn releases_transformer_before_vae(&self) -> bool {
        true
    }
}

/// Failure to fit bounded streaming or the exact complete VAE decoder.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum Flux2KleinMemoryAdmissionError {
    #[error("image memory admission requires a positive MLX ceiling")]
    InvalidCeiling,
    #[error(
        "image memory geometry requires exactly 25 transformer block payloads, received {actual_count}"
    )]
    InvalidTransformerBlockCount { actual_count: usize },
    #[error("image memory geometry byte accounting overflowed")]
    GeometryOverflow,
    #[error(
        "exact BF16 image execution requires {required_bytes} bytes, exceeding the {ceiling_bytes}-byte ceiling"
    )]
    InsufficientMemory {
        required_bytes: u64,
        ceiling_bytes: u64,
    },
    #[error(
        "exact BF16 VAE decoding requires {required_bytes} bytes, exceeding the {ceiling_bytes}-byte ceiling; increase maximum MLX memory or request smaller image dimensions because independent tiles change global GroupNorm and middle-attention arithmetic"
    )]
    CompleteVaeRequiresMoreMemory {
        required_bytes: u64,
        ceiling_bytes: u64,
    },
}

/// Composes image owners against the same process MLX ceiling primitive as text serving.
pub struct Flux2KleinMemoryAdmission;

impl Flux2KleinMemoryAdmission {
    /// Returns the maximum of streamed text, fixed transformer execution with no
    /// retained blocks, complete VAE decoding, and RGB/PNG/base64 encoding overlap.
    pub fn minimum_mlx_memory_ceiling_bytes(
        geometry: &Flux2KleinMemoryGeometry,
    ) -> Result<u64, Flux2KleinMemoryAdmissionError> {
        let execution_peaks = minimum_execution_peaks(geometry)?;
        Ok(execution_peaks.minimum_mlx_memory_ceiling_bytes())
    }

    pub fn plan(
        mlx_active_memory_ceiling_bytes: u64,
        geometry: &Flux2KleinMemoryGeometry,
    ) -> Result<Flux2KleinResidencyPlan, Flux2KleinMemoryAdmissionError> {
        let budget = MlxRamBudget::with_bootstrap_context_window_reserve_bytes(
            mlx_active_memory_ceiling_bytes,
            MlxRamBudgetModelGeometry {
                model_core_payload_bytes: 0,
                complete_expert_payload_bytes: 0,
                largest_complete_expert_layer_bytes: 0,
                largest_routed_expert_page_bytes: 0,
                sequence_state_bytes_per_token: 0,
            },
            0,
        )
        .map_err(|_| Flux2KleinMemoryAdmissionError::InvalidCeiling)?;
        let ceiling_bytes = budget.mlx_active_memory_ceiling_bytes();
        let execution_peaks = minimum_execution_peaks(geometry)?;

        ensure_fits(execution_peaks.streamed_text_peak_bytes, ceiling_bytes)?;
        let complete_text_bytes = checked_sum(&[
            geometry.text_encoder_payload_bytes,
            geometry.conditioning_bytes,
            geometry.largest_component_load_page_bytes,
        ])?;
        let text_encoder_mode = if complete_text_bytes <= ceiling_bytes {
            Flux2KleinResidencyMode::Complete
        } else {
            Flux2KleinResidencyMode::Streamed
        };

        ensure_fits(
            execution_peaks.transformer_fixed_execution_peak_bytes,
            ceiling_bytes,
        )?;
        let retained_block_budget_bytes =
            ceiling_bytes - execution_peaks.transformer_fixed_execution_peak_bytes;
        let mut block_candidates = geometry
            .transformer_block_payload_bytes
            .iter()
            .copied()
            .enumerate()
            .collect::<Vec<_>>();
        block_candidates.sort_by_key(|(block_index, payload_bytes)| (*payload_bytes, *block_index));
        let mut retained_transformer_block_indices = Vec::new();
        let mut retained_block_bytes = 0_u64;
        for (block_index, block_payload_bytes) in block_candidates {
            let candidate_bytes = retained_block_bytes
                .checked_add(block_payload_bytes)
                .ok_or(Flux2KleinMemoryAdmissionError::GeometryOverflow)?;
            if candidate_bytes > retained_block_budget_bytes {
                break;
            }
            retained_block_bytes = candidate_bytes;
            retained_transformer_block_indices.push(block_index);
        }
        retained_transformer_block_indices.sort_unstable();

        // VAE arrays are released before PNG/base64 encoding. RGB is the handoff
        // owner shared by both stages, so adding encoded bytes to VAE residency
        // would defeat the sequential plan and reject otherwise viable laptops.
        ensure_fits(execution_peaks.encoding_overlap_peak_bytes, ceiling_bytes)?;
        if execution_peaks.complete_vae_peak_bytes > ceiling_bytes {
            return Err(
                Flux2KleinMemoryAdmissionError::CompleteVaeRequiresMoreMemory {
                    required_bytes: execution_peaks.complete_vae_peak_bytes,
                    ceiling_bytes,
                },
            );
        }
        let selected_text_bytes = if text_encoder_mode == Flux2KleinResidencyMode::Complete {
            complete_text_bytes
        } else {
            execution_peaks.streamed_text_peak_bytes
        };
        let selected_transformer_bytes = execution_peaks
            .transformer_fixed_execution_peak_bytes
            .checked_add(retained_block_bytes)
            .ok_or(Flux2KleinMemoryAdmissionError::GeometryOverflow)?;
        let selected_vae_bytes = execution_peaks
            .complete_vae_peak_bytes
            .max(execution_peaks.encoding_overlap_peak_bytes);
        let minimum_mlx_memory_ceiling_bytes = execution_peaks.minimum_mlx_memory_ceiling_bytes();
        Ok(Flux2KleinResidencyPlan {
            text_encoder_mode,
            retained_transformer_block_indices,
            vae_mode: Flux2KleinResidencyMode::Complete,
            conditioning_peak_bytes: selected_text_bytes,
            denoising_peak_bytes: selected_transformer_bytes,
            decoding_peak_bytes: execution_peaks.complete_vae_peak_bytes,
            encoding_peak_bytes: execution_peaks.encoding_overlap_peak_bytes,
            peak_required_bytes: selected_text_bytes
                .max(selected_transformer_bytes)
                .max(selected_vae_bytes),
            minimum_mlx_memory_ceiling_bytes,
        })
    }
}

struct MinimumExecutionPeaks {
    streamed_text_peak_bytes: u64,
    transformer_fixed_execution_peak_bytes: u64,
    complete_vae_peak_bytes: u64,
    encoding_overlap_peak_bytes: u64,
}

impl MinimumExecutionPeaks {
    fn minimum_mlx_memory_ceiling_bytes(&self) -> u64 {
        self.streamed_text_peak_bytes
            .max(self.transformer_fixed_execution_peak_bytes)
            .max(self.complete_vae_peak_bytes)
            .max(self.encoding_overlap_peak_bytes)
    }
}

fn minimum_execution_peaks(
    geometry: &Flux2KleinMemoryGeometry,
) -> Result<MinimumExecutionPeaks, Flux2KleinMemoryAdmissionError> {
    if geometry.transformer_block_payload_bytes.len() != 25 {
        return Err(
            Flux2KleinMemoryAdmissionError::InvalidTransformerBlockCount {
                actual_count: geometry.transformer_block_payload_bytes.len(),
            },
        );
    }

    // Streaming keeps one materialized source page beside durable conditioning taps.
    let streamed_text_peak_bytes = checked_sum(&[
        geometry.largest_component_load_page_bytes,
        geometry.conditioning_bytes,
    ])?;
    let transformer_block_total_bytes = checked_sum(&geometry.transformer_block_payload_bytes)?;
    let transformer_fixed_payload_bytes = geometry
        .transformer_payload_bytes
        .checked_sub(transformer_block_total_bytes)
        .ok_or(Flux2KleinMemoryAdmissionError::GeometryOverflow)?;
    let transformer_fixed_execution_peak_bytes = checked_sum(&[
        transformer_fixed_payload_bytes,
        geometry.conditioning_bytes,
        geometry.latent_state_bytes,
        geometry.denoising_workspace_bytes,
        geometry.largest_component_load_page_bytes,
    ])?;
    let complete_vae_peak_bytes = checked_sum(&[
        geometry.vae_payload_bytes,
        geometry.vae_workspace_bytes,
        geometry.host_rgb_bytes,
    ])?;
    let encoding_overlap_peak_bytes = checked_sum(&[
        geometry.host_rgb_bytes,
        geometry.maximum_png_bytes,
        geometry.maximum_base64_bytes,
    ])?;

    Ok(MinimumExecutionPeaks {
        streamed_text_peak_bytes,
        transformer_fixed_execution_peak_bytes,
        complete_vae_peak_bytes,
        encoding_overlap_peak_bytes,
    })
}

fn checked_sum(byte_counts: &[u64]) -> Result<u64, Flux2KleinMemoryAdmissionError> {
    byte_counts
        .iter()
        .try_fold(0_u64, |total_bytes, byte_count| {
            total_bytes
                .checked_add(*byte_count)
                .ok_or(Flux2KleinMemoryAdmissionError::GeometryOverflow)
        })
}

fn ensure_fits(
    required_bytes: u64,
    ceiling_bytes: u64,
) -> Result<(), Flux2KleinMemoryAdmissionError> {
    if required_bytes <= ceiling_bytes {
        Ok(())
    } else {
        Err(Flux2KleinMemoryAdmissionError::InsufficientMemory {
            required_bytes,
            ceiling_bytes,
        })
    }
}
