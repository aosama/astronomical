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
//! [`reference`] is a pure-Rust float32 implementation of the same search
//! and packing. It exists so the GPU encoder has an oracle in the direct-MLX
//! contract tests, and so the packing rules are unit-testable on any host.

pub mod reference;

#[cfg(feature = "direct-mlx")]
pub mod mlx;

use crate::bit_width::BitWidth;

/// The scale factors tried at each anchor, OptiQ's `_OQ_FACTORS`. Factors
/// above 1.0 are searched as well as below; the range is chosen per group,
/// anchored at an edge rather than shrunk about the midpoint.
pub const RANGE_SEARCH_SCALE_FACTORS: [f32; 7] = [0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25];

/// Widest width the search applies to by default. 4-bit is deliberately
/// excluded: nearly every published OptiQ quant is 4-bit, min/max behaves
/// acceptably there (5.8% is not the difference between working and broken),
/// and leaving it alone keeps those artifacts byte-reproducible. 2- and
/// 3-bit is where min/max collapses weights onto a single code.
pub const DEFAULT_RANGE_SEARCH_MAXIMUM_BITS: u32 = 3;

/// Above this the search found no improvement on any tensor measured.
pub const NEVER_SEARCH_ABOVE_BITS: u32 = 4;

/// Numerical floor for a group's scale.
pub const SCALE_EPSILON: f32 = 1e-7;

/// Above this many elements a 3-D tensor is encoded in slices along its
/// leading axis. Groups run along the last axis, so slicing the first
/// changes nothing about the result; it only bounds how much is live.
pub const RANGE_SEARCH_CHUNK_ELEMENTS: usize = 1 << 28;

/// Whether the range search runs for a width under the configured maximum.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RangeSearchPolicy {
    pub enabled: bool,
    /// Clamped to [`NEVER_SEARCH_ABOVE_BITS`].
    pub maximum_bits: u32,
}

impl RangeSearchPolicy {
    /// OptiQ's default: on, through 3 bits.
    #[must_use]
    pub const fn default_policy() -> Self {
        Self {
            enabled: true,
            maximum_bits: DEFAULT_RANGE_SEARCH_MAXIMUM_BITS,
        }
    }

    /// Reproduces a pre-search artifact exactly (stock min/max everywhere).
    #[must_use]
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            maximum_bits: 0,
        }
    }

    #[must_use]
    pub fn applies_to(&self, bit_width: BitWidth) -> bool {
        self.enabled
            && bit_width.bits() <= self.maximum_bits.min(NEVER_SEARCH_ABOVE_BITS)
            && !bit_width.keeps_bfloat16()
    }
}
