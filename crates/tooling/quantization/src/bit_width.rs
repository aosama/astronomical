//! Validated bit widths and group sizes for MLX affine quantization.
//!
//! MLX's affine mode accepts bit widths 1 through 8 and group sizes 32, 64,
//! and 128. Astronomical's artifact readers accept the subset 2, 3, 4, 5, 6,
//! 8, and OptiQ adds one pseudo width: 16 means "leave this module at its
//! source bfloat16 precision". The keep-bfloat16 tier is not a finer grid, it
//! is the lossless tier: the allocator can spend budget to buy back exactness
//! on the modules that need it, and the requantizer passes such tensors
//! through untouched.

use serde::{Deserialize, Serialize};

use crate::error::QuantizationError;

#[cfg(test)]
#[path = "bit_width_tests.rs"]
mod bit_width_tests;

/// The pseudo bit width meaning "do not quantize this module".
pub const KEEP_BFLOAT16_BITS: u32 = 16;

/// A bit width the allocator may assign to a module.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct BitWidth(u32);

impl BitWidth {
    /// Accepts the MLX affine widths Astronomical reads (2, 3, 4, 5, 6, 8) and
    /// the keep-bfloat16 tier (16). Rejects 1 and 7: MLX packs them, but no
    /// published artifact uses them and the model-serving readers refuse them.
    pub fn new(bits: u32) -> Result<Self, QuantizationError> {
        if matches!(bits, 2 | 3 | 4 | 5 | 6 | 8 | KEEP_BFLOAT16_BITS) {
            Ok(Self(bits))
        } else {
            Err(QuantizationError::UnsupportedBitWidth { bits })
        }
    }

    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// True for the lossless tier that leaves the weight at bfloat16.
    #[must_use]
    pub const fn keeps_bfloat16(self) -> bool {
        self.0 == KEEP_BFLOAT16_BITS
    }
}

/// A group size the MLX affine encoder accepts.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GroupSize(u32);

impl GroupSize {
    pub fn new(group_size: u32) -> Result<Self, QuantizationError> {
        if matches!(group_size, 32 | 64 | 128) {
            Ok(Self(group_size))
        } else {
            Err(QuantizationError::UnsupportedGroupSize { group_size })
        }
    }

    #[must_use]
    pub const fn elements(self) -> u32 {
        self.0
    }
}

/// An ascending, de-duplicated ladder of candidate bit widths, for example
/// `[2, 4, 8]` or `[4, 8, 16]`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CandidateBitWidths(Vec<BitWidth>);

impl CandidateBitWidths {
    /// Validates each width, sorts ascending, and rejects duplicates and empty
    /// ladders. The allocator relies on ascending order to find "the next
    /// width up" for a module.
    pub fn new(candidate_bits: &[u32]) -> Result<Self, QuantizationError> {
        // An empty ladder is rejected outright: `lowest()` indexes the first
        // entry and would panic on an empty ladder, and allocation needs at
        // least one width to choose from.
        if candidate_bits.is_empty() {
            return Err(QuantizationError::InvalidCandidateBitWidths {
                candidate_bit_widths: candidate_bits.to_vec(),
            });
        }

        let mut widths: Vec<BitWidth> = Vec::with_capacity(candidate_bits.len());
        for &bits in candidate_bits {
            // A width outside the validated grid (1, 7, 9, ...) fails here with
            // its own specific error before the ladder is ever sorted.
            widths.push(BitWidth::new(bits)?);
        }

        // Sort ascending and reject duplicates: the allocator walks this ladder
        // to find "the next width up", so a repeated width would double-count a
        // budget step. `BitWidth` is `Ord`, so adjacent equality after sorting
        // finds every duplicate.
        widths.sort();
        if widths.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(QuantizationError::InvalidCandidateBitWidths {
                candidate_bit_widths: candidate_bits.to_vec(),
            });
        }

        Ok(Self(widths))
    }

    #[must_use]
    pub fn lowest(&self) -> BitWidth {
        self.0[0]
    }

    #[must_use]
    pub fn highest(&self) -> BitWidth {
        self.0[self.0.len() - 1]
    }

    /// The second-lowest width, which the block-aware floor and the run guard
    /// promote modules to. `None` when the ladder has a single width.
    #[must_use]
    pub fn second_lowest(&self) -> Option<BitWidth> {
        self.0.get(1).copied()
    }

    /// The width directly above `current` on the ladder, or `None` at the top
    /// or when `current` is not on the ladder.
    #[must_use]
    pub fn next_above(&self, current: BitWidth) -> Option<BitWidth> {
        // The ladder is stored ascending (see `new`), so the width directly
        // above `current` is simply the next entry after its position. A width
        // not on the ladder, or the highest one, has no successor.
        let index = self.0.iter().position(|&width| width == current)?;
        self.0.get(index + 1).copied()
    }

    #[must_use]
    pub fn as_slice(&self) -> &[BitWidth] {
        &self.0
    }
}
