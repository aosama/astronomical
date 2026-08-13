//! Request-scoped orchestration for Qwen3.5 SpecPrefill.
//!
//! SpecPrefill reduces target-model prompt work without changing the prompt's
//! logical positions. A smaller Qwen3.5 "drafter" reads the complete prompt,
//! assigns an importance score to each selectable conversation token, and picks
//! the positions that the larger target model must process. The target then
//! executes those rows as one compact sparse forward while rotary position
//! encoding still sees every row's original absolute prompt position.
//!
//! # End-to-end request lifecycle
//!
//! 1. Model loading validates that target and drafter use the same canonical
//!    token-to-identifier mapping. It records compatibility metadata, but drops
//!    the startup drafter so target residency is not permanently displaced.
//! 2. Request startup restores reusable target state when possible and evaluates
//!    request-specific eligibility. System/tool control tokens remain dense.
//! 3. At the first selectable conversation position, the engine first looks for
//!    an exact reusable selection in worker memory and then on disk.
//! 4. On a miss, the engine temporarily reclaims pageable target experts, loads
//!    a request-scoped drafter, restores the longest safe drafter prefix, scores
//!    the remaining prompt, and selects absolute target positions.
//! 5. Every image-pad position and the configured trailing window are mandatory;
//!    importance ranking may remove neither.
//! 6. The drafter and all drafter-owned MLX state are released before sparse
//!    target execution resumes. Complete target residency is restored when the
//!    live request state and exact expert payload fit together.
//! 7. A successful sparse target prefix can be persisted under the complete
//!    target/drafter/policy/prompt identity for exact later reuse.
//!
//! # Failure policy
//!
//! Once configured SpecPrefill work begins, errors fail the request (or model
//! activation during loading). The engine deliberately does not retry the same
//! request through target-only execution: a retry after partial decoder-state or
//! cache mutation could duplicate work or silently change the promised policy.
//!
//! # Ownership map
//!
//! Pure planning and selection formulas are available without MLX so hermetic
//! tests can exercise them. MLX execution items are feature-gated where required.
//! All children remain private; the re-exports below are the intentionally small
//! boundary used by sibling inference stages.

// Pure formulas used by prompt-prefill orchestration and hermetic tests.
mod chunk_mode;
mod speculative_prefill_control_span;
// Drafter-state restoration and synchronous block publication.
mod speculative_prefill_draft_artifact_loading;
mod speculative_prefill_draft_cache;
mod speculative_prefill_draft_release;
// Request-specific policy gate evaluated after any target prefix restore.
mod speculative_prefill_eligibility;
// Bounded public errors and rich local failure evidence.
mod speculative_prefill_failure;
mod speculative_prefill_failure_diagnostics;
// MLX input assembly and exact memory admission for the drafter phase.
mod speculative_prefill_gpu_input;
mod speculative_prefill_memory_admission;
// Startup/request-scoped drafter validation, loading, and policy activation.
mod speculative_prefill_model_loading;
mod speculative_prefill_policy_activation;
// Drafter scoring orchestration and pure/GPU position selection.
mod speculative_prefill_scoring;
mod speculative_prefill_selection;
#[cfg(feature = "direct-mlx")]
mod speculative_prefill_selection_gpu;
// Selection reuse/publication and bounded worker-memory stores.
mod speculative_prefill_selection_persistence;
mod speculative_prefill_selection_reuse;
mod speculative_prefill_store;
// Selection-bound target-state persistence and visual drafter input.
mod speculative_prefill_target_cache;
mod speculative_prefill_visual_embedding;

// Public pure helpers are retained for crate consumers and direct source tests.
pub use self::chunk_mode::{
    Qwen3_5SpeculativePrefillChunckMode, qwen3_5_speculative_prefill_chunck_mode,
};
pub use self::speculative_prefill_control_span::{
    qwen3_5_prefill_chunck_end_at_ordinary_target_control_span_boundary,
    qwen3_5_speculative_prefill_sparse_target_is_active,
};
pub use self::speculative_prefill_selection::{
    Qwen3_5SpeculativePrefillSelectionError, qwen3_5_select_speculative_prefill_token_positions,
    qwen3_5_selected_speculative_prefill_positions_for_range,
};

// Runtime orchestration details remain crate-visible rather than public API.
pub(crate) use self::speculative_prefill_draft_artifact_loading::{
    load_speculative_prefill_draft_model, token_identifier_mapping_digest,
};
pub(crate) use self::speculative_prefill_eligibility::{
    Qwen3_5SpeculativePrefillRequestEligibility, qwen3_5_speculative_prefill_request_eligibility,
};
pub(crate) use self::speculative_prefill_failure::{
    configured_speculative_prefill_activation_failure, configured_speculative_prefill_failure,
};
pub(crate) use self::speculative_prefill_scoring::SpeculativePrefillSelectionPreparation;
pub(crate) use self::speculative_prefill_store::{
    Qwen3_5SpeculativePrefillDraftPrefixStoreEntry, Qwen3_5SpeculativePrefillStoreKey,
};
