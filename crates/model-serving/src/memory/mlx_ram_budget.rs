//! Single-source MLX RAM budget owner for context, activations, streaming, and experts.
//!
//! This owner is the only place that composes the production split:
//!
//! ```text
//! retained_expert_budget_bytes =
//!     mlx_active_memory_ceiling_bytes
//!     - model_core_payload_bytes
//!     - context_window_reserve_bytes
//!     - activation_headroom_bytes
//!     - complete_layer_stream_slot_bytes
//!     - other_fixed_bytes
//! ```
//!
//! Callers supply measured bytes and token counts. The owner starts from a
//! bootstrap context-window reserve and learns a conservative envelope from reports.
//!
//! # Categories are intentionally non-overlapping
//!
//! - **Model core** is the loaded non-expert model payload.
//! - **Context reserve** protects persistent request state as context grows.
//! - **Activation headroom** protects temporary work learned from completed
//!   forwards. Prefill and decode retain separate high-water marks.
//! - **Complete-layer stream slot** is scratch ownership for one nonresident
//!   sparse layer. It is required even when most complete layers are retained.
//! - **Other fixed bytes** lets request-specific owners, such as a draft model or
//!   prompt-cache publication workspace, participate without duplicating policy.
//! - **Retained experts** receive only the bytes left after every required owner.
//!
//! An idle status snapshot can make context and streaming space appear unused.
//! That does not make the bytes available to expert retention: the next request
//! needs those categories concurrently with retained experts.
//!
//! # Safety policy
//!
//! Arithmetic saturates while composing a plan. Saturation intentionally fails
//! closed by reducing the leftover expert budget to zero instead of wrapping to a
//! large value. Learning is high-water-only; one cheaper request cannot erase
//! evidence required by a larger future request.

use std::collections::BTreeMap;

use thiserror::Error;

use super::{
    expert_reclamation_bytes_to_fit_fixed_forward,
    required_complete_residency_activation_headroom_bytes,
};

/// Bootstrap context-window reserve before any live measurement exists (1 GB SI).
pub const BOOTSTRAP_CONTEXT_WINDOW_RESERVE_BYTES: u64 = 1_000_000_000;

/// Coarse token buckets used for high-water context-window learning.
const CONTEXT_TOKEN_BUCKET_WIDTH: u64 = 1_024;

/// Extra safety margin applied on top of measured context-window need.
const CONTEXT_WINDOW_MEASUREMENT_SAFETY_BUFFER_BYTES: u64 = 64_000_000;

/// Forward class that consumed or will consume temporary activation memory.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlxRamBudgetPhase {
    Prefill,
    Decode,
    Idle,
}

/// Immutable inputs known once a model is loaded against one ceiling.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MlxRamBudgetModelGeometry {
    /// Non-expert resident model payload (language core, optional vision/MTP).
    pub model_core_payload_bytes: u64,
    /// Bytes required if every sparse expert is fully resident.
    pub complete_expert_payload_bytes: u64,
    /// One complete sparse layer; reserved as the streaming workspace.
    pub largest_complete_expert_layer_bytes: u64,
    /// Largest exact top-K page used by one-token decode.
    pub largest_routed_expert_page_bytes: u64,
}

/// One composed RAM split for a planned operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MlxRamBudgetSnapshot {
    /// Total MLX active-memory ceiling for this plan.
    pub mlx_active_memory_ceiling_bytes: u64,
    /// Non-expert model core already charged against the ceiling.
    pub model_core_payload_bytes: u64,
    /// Reserved bytes for context-window growth at the planned token count.
    pub context_window_reserve_bytes: u64,
    /// Reserved bytes for temporary activations / transient workspace.
    pub activation_headroom_bytes: u64,
    /// Reserved bytes for one complete-layer stream workspace.
    pub complete_layer_stream_slot_bytes: u64,
    /// Any additional fixed non-expert owners (draft model, publication workspace, …).
    pub other_fixed_bytes: u64,
    /// Leftover budget that may pin retained expert layers in MLX.
    pub retained_expert_budget_bytes: u64,
}

/// Live measurement that refines context-window reserve and activation headroom.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MlxRamBudgetMeasurement {
    /// Execution class whose activation high-water this sample may raise.
    pub phase: MlxRamBudgetPhase,
    /// Context size used to choose a monotonic coarse learning bucket.
    pub context_token_count: u64,
    /// Measured request-owned persistent and transient bytes above model core.
    pub measured_context_and_activation_bytes: u64,
    /// Transient-only high-water independently learned by forward admission.
    pub observed_activation_headroom_bytes: u64,
    /// Explicit operation workspace already reserved by forward admission.
    pub exact_temporary_workspace_bytes: u64,
}

/// Invalid configuration for the RAM budget owner.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum MlxRamBudgetError {
    #[error("MLX RAM budget requires a positive active-memory ceiling")]
    InvalidCeiling,
}

/// Single-source owner for MLX RAM policy across streaming and expert retention.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MlxRamBudget {
    /// User/machine-resolved hard production ceiling; never a model-specific constant.
    mlx_active_memory_ceiling_bytes: u64,
    /// Startup-validated payload geometry for the currently loaded model.
    model_geometry: MlxRamBudgetModelGeometry,
    /// Conservative floor used until at least one real context measurement exists.
    bootstrap_context_window_reserve_bytes: u64,
    /// High-water request workspace keyed by 1,024-token context buckets.
    measured_context_window_high_water_by_token_bucket: BTreeMap<u64, u64>,
    /// Phase-specific activation evidence; these values can grow but never shrink.
    prefill_activation_headroom_bytes: u64,
    decode_activation_headroom_bytes: u64,
    /// Distinguishes an unobserved prefill phase from a valid zero-byte sample.
    has_prefill_activation_measurement: bool,
    /// Distinguishes an unobserved decode phase from a valid zero-byte sample.
    has_decode_activation_measurement: bool,
    /// Distinguishes "no evidence" from a valid measured value of zero.
    has_context_window_measurement: bool,
}

impl MlxRamBudget {
    /// Creates the owner with bootstrap context-window reserve of 1 GB SI.
    pub fn new(
        mlx_active_memory_ceiling_bytes: u64,
        model_geometry: MlxRamBudgetModelGeometry,
    ) -> Result<Self, MlxRamBudgetError> {
        Self::with_bootstrap_context_window_reserve_bytes(
            mlx_active_memory_ceiling_bytes,
            model_geometry,
            BOOTSTRAP_CONTEXT_WINDOW_RESERVE_BYTES,
        )
    }

    pub fn with_bootstrap_context_window_reserve_bytes(
        mlx_active_memory_ceiling_bytes: u64,
        model_geometry: MlxRamBudgetModelGeometry,
        bootstrap_context_window_reserve_bytes: u64,
    ) -> Result<Self, MlxRamBudgetError> {
        if mlx_active_memory_ceiling_bytes == 0 {
            return Err(MlxRamBudgetError::InvalidCeiling);
        }
        Ok(Self {
            mlx_active_memory_ceiling_bytes,
            model_geometry,
            bootstrap_context_window_reserve_bytes,
            measured_context_window_high_water_by_token_bucket: BTreeMap::new(),
            prefill_activation_headroom_bytes: 0,
            decode_activation_headroom_bytes: 0,
            has_prefill_activation_measurement: false,
            has_decode_activation_measurement: false,
            has_context_window_measurement: false,
        })
    }

    pub fn update_mlx_active_memory_ceiling_bytes(
        &mut self,
        mlx_active_memory_ceiling_bytes: u64,
    ) -> Result<(), MlxRamBudgetError> {
        // Keep learned model/workload evidence across live ceiling changes. The
        // same bytes are re-composed against the new ceiling on the next plan.
        if mlx_active_memory_ceiling_bytes == 0 {
            return Err(MlxRamBudgetError::InvalidCeiling);
        }
        self.mlx_active_memory_ceiling_bytes = mlx_active_memory_ceiling_bytes;
        Ok(())
    }

    pub fn update_model_geometry(&mut self, model_geometry: MlxRamBudgetModelGeometry) {
        // The owner follows the loaded model. Callers must provide geometry from
        // validated artifacts, not infer payload sizes from model names.
        self.model_geometry = model_geometry;
    }

    #[must_use]
    pub const fn mlx_active_memory_ceiling_bytes(&self) -> u64 {
        self.mlx_active_memory_ceiling_bytes
    }

    #[must_use]
    pub const fn model_geometry(&self) -> MlxRamBudgetModelGeometry {
        self.model_geometry
    }

    #[must_use]
    pub const fn has_context_window_measurement(&self) -> bool {
        self.has_context_window_measurement
    }

    /// Context-window reserve for `context_token_count` tokens.
    ///
    /// Before measurements: bootstrap constant (1 GB SI).
    /// After measurements: conservative high-water by token bucket.
    #[must_use]
    pub fn context_window_reserve_bytes(&self, context_token_count: u64) -> u64 {
        if !self.has_context_window_measurement {
            return self.bootstrap_context_window_reserve_bytes;
        }
        let token_bucket = context_token_bucket(context_token_count);
        // Prefer the highest evidence at or below the requested bucket. Context
        // memory normally grows with position, so a smaller earlier observation
        // must not override a larger known lower-bucket high-water.
        let at_or_below_bucket_bytes = self
            .measured_context_window_high_water_by_token_bucket
            .range(..=token_bucket)
            .map(|(_, measured_bytes)| *measured_bytes)
            .max()
            .unwrap_or(0);
        let larger_bucket_floor_bytes = self
            .measured_context_window_high_water_by_token_bucket
            .range(token_bucket..)
            .map(|(_, measured_bytes)| *measured_bytes)
            .min()
            .unwrap_or(0);
        // If this process has measured only larger requests, reuse the smallest
        // larger measurement rather than pretending the new smaller bucket has no
        // cost. This is conservative until direct evidence for the bucket exists.
        let learned_context_window_reserve_bytes = if at_or_below_bucket_bytes > 0 {
            at_or_below_bucket_bytes
        } else {
            larger_bucket_floor_bytes
        };
        learned_context_window_reserve_bytes.max(self.bootstrap_context_window_reserve_bytes)
    }

    #[must_use]
    pub fn activation_headroom_bytes(&self, phase: MlxRamBudgetPhase) -> u64 {
        match phase {
            MlxRamBudgetPhase::Prefill => {
                // Keep the geometry-derived startup floor after learning. A
                // cheaper completed chunk must not let retained pages expand
                // into the operating interval needed by a later large chunk.
                required_complete_residency_activation_headroom_bytes(
                    self.model_geometry.complete_expert_payload_bytes,
                    self.prefill_activation_headroom_bytes,
                )
            }
            MlxRamBudgetPhase::Decode => {
                if self.has_decode_activation_measurement {
                    self.decode_activation_headroom_bytes
                } else if self.has_prefill_activation_measurement {
                    required_complete_residency_activation_headroom_bytes(
                        self.model_geometry.complete_expert_payload_bytes,
                        self.prefill_activation_headroom_bytes,
                    )
                } else {
                    // Decode follows prefill in the user journey. Until one
                    // decode completes, prefill high-water is the only live
                    // evidence preventing warm fill from occupying transient
                    // space that the first token immediately needs back.
                    required_complete_residency_activation_headroom_bytes(
                        self.model_geometry.complete_expert_payload_bytes,
                        0,
                    )
                }
            }
            MlxRamBudgetPhase::Idle => 0,
        }
    }

    /// Records one live observation and never lowers prior high-water evidence.
    pub fn record_measurement(&mut self, measurement: MlxRamBudgetMeasurement) {
        // The fixed buffer absorbs measurement jitter and small untracked MLX
        // bookkeeping. It is added before high-water comparison so every stored
        // bucket is directly usable as a future reserve.
        self.has_context_window_measurement = true;
        let token_bucket = context_token_bucket(measurement.context_token_count);
        // The forward sample includes both persistent context and transient
        // activation bytes. Activation has its own separately composed owner, so
        // subtract that evidence before learning context or the same workspace is
        // reserved twice. Saturation fails safe when independently sampled
        // transient evidence is larger because of allocator timing differences.
        let measured_persistent_context_bytes = measurement
            .measured_context_and_activation_bytes
            .saturating_sub(measurement.observed_activation_headroom_bytes)
            .saturating_sub(measurement.exact_temporary_workspace_bytes);
        let measured_with_buffer = measured_persistent_context_bytes
            .saturating_add(CONTEXT_WINDOW_MEASUREMENT_SAFETY_BUFFER_BYTES);
        self.measured_context_window_high_water_by_token_bucket
            .entry(token_bucket)
            .and_modify(|existing_high_water_bytes| {
                *existing_high_water_bytes = (*existing_high_water_bytes).max(measured_with_buffer);
            })
            .or_insert(measured_with_buffer);

        match measurement.phase {
            MlxRamBudgetPhase::Prefill => {
                self.has_prefill_activation_measurement = true;
                self.prefill_activation_headroom_bytes = self
                    .prefill_activation_headroom_bytes
                    .max(measurement.observed_activation_headroom_bytes);
            }
            MlxRamBudgetPhase::Decode => {
                self.has_decode_activation_measurement = true;
                self.decode_activation_headroom_bytes = self
                    .decode_activation_headroom_bytes
                    .max(measurement.observed_activation_headroom_bytes);
            }
            MlxRamBudgetPhase::Idle => {}
        }
        // Idle has no activation operation to learn from. It can still mark that
        // context evidence exists if a caller deliberately records an idle sample.
    }

    /// Composes the full budget for one planned operation.
    #[must_use]
    pub fn plan(
        &self,
        phase: MlxRamBudgetPhase,
        context_token_count: u64,
        other_fixed_bytes: u64,
    ) -> MlxRamBudgetSnapshot {
        // Idle refill happens after request arrays are released, so it needs no
        // live context reserve. Prefill/decode plans protect the next operation's
        // context even if the current allocator snapshot is temporarily lower.
        let context_window_reserve_bytes = match phase {
            MlxRamBudgetPhase::Idle => 0,
            MlxRamBudgetPhase::Prefill | MlxRamBudgetPhase::Decode => {
                self.context_window_reserve_bytes(context_token_count)
            }
        };
        let activation_headroom_bytes = self.activation_headroom_bytes(phase);
        let complete_layer_stream_slot_bytes = match phase {
            MlxRamBudgetPhase::Prefill => self.model_geometry.largest_complete_expert_layer_bytes,
            MlxRamBudgetPhase::Decode => self.model_geometry.largest_routed_expert_page_bytes,
            MlxRamBudgetPhase::Idle => 0,
        };
        let fixed_non_expert_bytes = self
            .model_geometry
            .model_core_payload_bytes
            .saturating_add(context_window_reserve_bytes)
            .saturating_add(activation_headroom_bytes)
            .saturating_add(complete_layer_stream_slot_bytes)
            .saturating_add(other_fixed_bytes);
        let mlx_active_memory_ceiling_bytes = self.mlx_active_memory_ceiling_bytes();
        let retained_expert_budget_bytes =
            mlx_active_memory_ceiling_bytes.saturating_sub(fixed_non_expert_bytes);

        MlxRamBudgetSnapshot {
            mlx_active_memory_ceiling_bytes,
            model_core_payload_bytes: self.model_geometry.model_core_payload_bytes,
            context_window_reserve_bytes,
            activation_headroom_bytes,
            complete_layer_stream_slot_bytes,
            other_fixed_bytes,
            retained_expert_budget_bytes,
        }
    }

    /// Refines retained ownership against one concrete admitted forward.
    ///
    /// `current_active_memory_bytes` already includes current retained experts,
    /// while `admitted_forward_reserve_bytes` includes exact persistent growth,
    /// one phase-correct expert page, and expected transient work. Adding the
    /// remaining strict-ceiling capacity to current retention therefore yields
    /// the largest expert payload that may coexist with that forward.
    #[must_use]
    pub const fn retained_expert_budget_for_admitted_forward(
        &self,
        current_active_memory_bytes: u64,
        current_retained_expert_payload_bytes: u64,
        admitted_forward_reserve_bytes: u64,
    ) -> u64 {
        let projected_active_memory_bytes =
            current_active_memory_bytes.saturating_add(admitted_forward_reserve_bytes);
        let mlx_active_memory_ceiling_bytes = self.mlx_active_memory_ceiling_bytes();
        if projected_active_memory_bytes <= mlx_active_memory_ceiling_bytes {
            return current_retained_expert_payload_bytes
                .saturating_add(mlx_active_memory_ceiling_bytes - projected_active_memory_bytes);
        }
        current_retained_expert_payload_bytes
            .saturating_sub(projected_active_memory_bytes - mlx_active_memory_ceiling_bytes)
    }

    /// Reclamation needed so a fixed forward workspace fits the ceiling.
    #[must_use]
    pub fn expert_reclamation_bytes_for_fixed_forward(
        &self,
        current_active_memory_bytes: u64,
        retained_expert_payload_bytes: u64,
        fixed_forward_workspace_bytes: u64,
    ) -> u64 {
        u64::try_from(expert_reclamation_bytes_to_fit_fixed_forward(
            usize::try_from(current_active_memory_bytes).unwrap_or(usize::MAX),
            usize::try_from(retained_expert_payload_bytes).unwrap_or(usize::MAX),
            usize::try_from(self.mlx_active_memory_ceiling_bytes()).unwrap_or(usize::MAX),
            usize::try_from(fixed_forward_workspace_bytes).unwrap_or(usize::MAX),
        ))
        .unwrap_or(u64::MAX)
    }
}

/// Shared coarse bucket keeps context-growth evidence comparable across memory policies.
pub(crate) const fn context_token_bucket(context_token_count: u64) -> u64 {
    context_token_count / CONTEXT_TOKEN_BUCKET_WIDTH
}

/// Separates request workspace from expert ownership acquired during a forward.
///
/// The MLX peak includes both categories. Charging newly retained experts to
/// context learning would reserve those bytes again and evict the topology that
/// produced the measurement.
#[must_use]
pub const fn measured_non_expert_forward_growth_bytes(
    active_memory_bytes_before_growth: u64,
    peak_memory_bytes_during_growth: u64,
    retained_expert_payload_bytes_before_growth: u64,
    retained_expert_payload_bytes_after_growth: u64,
) -> u64 {
    let newly_retained_expert_payload_bytes = retained_expert_payload_bytes_after_growth
        .saturating_sub(retained_expert_payload_bytes_before_growth);
    peak_memory_bytes_during_growth
        .saturating_sub(active_memory_bytes_before_growth)
        .saturating_sub(newly_retained_expert_payload_bytes)
}
