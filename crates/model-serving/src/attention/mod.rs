//! Architecture-neutral attention math used by more than one model family.
//!
//! CPU helpers live here so hermetic tests can prove YaRN denominators,
//! sliding-window visibility, and rotating admission geometry without linking
//! MLX or importing a concrete family. Family owners supply geometry; they do
//! not fork these formulas.

#[cfg(feature = "direct-mlx")]
mod mlx_sliding_window_mask;
mod rotating_admission;
mod sliding_window_visibility;
mod yarn_frequencies;

#[cfg(feature = "direct-mlx")]
pub use mlx_sliding_window_mask::build_causal_sliding_window_mask;
pub use rotating_admission::{
    RotatingAdmissionError, rotating_committed_token_count, rotating_prefill_transient_token_count,
};
pub use sliding_window_visibility::{
    SlidingWindowVisibilityError, sliding_window_position_is_visible,
    sliding_window_visibility_table,
};
pub use yarn_frequencies::{
    RopeFrequencyError, YarnRopeFrequencyDenominators, compute_default_rope_frequency_denominators,
    compute_yarn_rope_frequency_denominators,
};
