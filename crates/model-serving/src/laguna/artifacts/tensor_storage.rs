use crate::laguna::{LagunaAffineProfile, LagunaNvfp4Profile, LagunaSymmetricPackedAffineProfile};

/// Physical encoding applied to one canonical execution-shape component.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LagunaTensorStorageEncoding {
    /// The source weight is stored directly in the model float dtype.
    Unquantized,
    /// The component belongs to one MLX affine matrix with an exact profile.
    DirectAffine { profile: LagunaAffineProfile },
    /// Runtime-ready packed affine storage with scale-derived symmetric bias.
    SymmetricPackedAffine {
        profile: LagunaSymmetricPackedAffineProfile,
    },
    /// Runtime-ready native MLX NVFP4 codes and E4M3 scales.
    NativeNvfp4 { profile: LagunaNvfp4Profile },
    /// Exact two-level E2M1/E4M3/global-scale storage awaiting its kernel.
    TwoLevelCompressedNvfp4 { profile: LagunaNvfp4Profile },
    /// Exact E4M3 weights and complete F32 scale-block coverage awaiting its kernel.
    BlockFp8 {
        block_row_extent: usize,
        block_column_extent: usize,
    },
}
