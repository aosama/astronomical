//! Values the journeys must agree on, because comparing them is the point: a chunk
//! size or an accuracy bar that drifted in one file would make its numbers
//! incomparable with the rest.

#[cfg(feature = "mlx-baseline")]
use std::time::Duration;

/// The prefill chunk every synthetic journey measures.
pub(crate) const PREFILL_CHUNK_TOKEN_COUNT: usize = 2048;
/// An fp16 product of unit-variance activations and 0.02-spread weights over
/// 4,096 terms lands within this cosine of the fp32 result when the engine
/// accumulates in at least fp16; anything lower means wrong arithmetic, not
/// rounding.
pub(crate) const MINIMUM_COSINE_SIMILARITY: f64 = 0.9999;
/// How long a synthetic-weight prediction may take before it counts as hung.
/// Real weights in a whole MLP keep their own longer limit.
#[cfg(feature = "mlx-baseline")]
pub(crate) const NEURAL_ENGINE_FINISH_TIMEOUT: Duration = Duration::from_secs(5);
