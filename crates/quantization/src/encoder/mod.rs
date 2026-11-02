//! Low-bit affine quantization that picks each group's range by search.
//!
//! Stock MLX affine quantization derives a group's scale and bias from its
//! true minimum and maximum. That is a fine default at 4 bits and above and a
//! poor one at 2: with only four levels, a single outlier stretches the range
//! until the bulk of the weights fall between levels and collapse onto the
//! same code. OptiQ measured stock 2-bit on DeepSeek-V4's routed experts at
//! 50% exact zeros against a source with 12.8%, which destroyed the model
//! while mean error looked ordinary. Preserving the weight distribution is
//! the criterion, not mean error.
//!
//! The search (adapted by OptiQ from the oQ quantizer in oMLX, Apache-2.0;
//! both lineages are attributed in `third-party/THIRD_PARTY_NOTICES`) tries,
//! per group, seven scale factors anchored at the group's maximum and again
//! at its minimum, and keeps whichever candidate lowers the squared
//! reconstruction error. It initializes from stock min/max and only takes a
//! candidate that improves, so it never does worse. Measured improvement over
//! stock by width: 2-bit 9.0%, 3-bit 8.2%, 4-bit 5.8%, 5-bit and up none.
//!
//! The output layout is identical to `mx.quantize(..., mode="affine")`:
//! codes packed low-to-high into `uint32`, scales and biases of shape
//! `[..., columns / group_size]` in the input dtype. An artifact built this
//! way loads under stock mlx-lm, OptiQ, and Astronomical with no custom
//! kernel; only the choice of range differs.
//!
//! [`mod@reference`] is a pure-Rust float32 implementation of the same search
//! and packing. It exists so the GPU encoder has an oracle in the direct-MLX
//! contract tests, and so the packing rules are unit-testable on any host.

pub mod range_search_policy;
pub mod reference;

#[cfg(feature = "direct-mlx")]
pub mod mlx;

pub use range_search_policy::{
    DEFAULT_RANGE_SEARCH_MAXIMUM_BITS, NEVER_SEARCH_ABOVE_BITS, RANGE_SEARCH_CHUNK_ELEMENTS,
    RANGE_SEARCH_SCALE_FACTORS, RangeSearchPolicy, SCALE_EPSILON,
};
