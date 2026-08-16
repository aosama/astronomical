use std::collections::BTreeMap;

use super::compressed_storage_descriptor::LagunaCompressedStorageDescriptor;

/// Direct MLX affine bit width and group size for one module.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LagunaAffineProfile {
    bits: u32,
    group_size: u32,
}

impl LagunaAffineProfile {
    pub(super) const fn new(bits: u32, group_size: u32) -> Self {
        Self { bits, group_size }
    }

    /// Returns the affine packed bit width.
    #[must_use]
    pub const fn bits(&self) -> u32 {
        self.bits
    }

    /// Returns the affine group size.
    #[must_use]
    pub const fn group_size(&self) -> u32 {
        self.group_size
    }
}

/// Canonical direct-affine defaults and module overrides without raw namespace state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LagunaDirectAffineStorageDescriptor {
    default_profile: LagunaAffineProfile,
    module_overrides: BTreeMap<String, LagunaAffineProfile>,
}

/// Declares whether exact retained storage already has a runtime consumer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LagunaExactStorageSupport {
    RuntimeReady,
    FutureExactKernel,
}

/// Exact evidenced profile for symmetric packed affine storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LagunaSymmetricPackedAffineProfile {
    bits: u32,
    group_size: u32,
}

impl LagunaSymmetricPackedAffineProfile {
    pub(super) const fn evidenced() -> Self {
        Self {
            bits: 4,
            group_size: 32,
        }
    }

    #[must_use]
    pub const fn bits(&self) -> u32 {
        self.bits
    }

    #[must_use]
    pub const fn group_size(&self) -> u32 {
        self.group_size
    }

    #[must_use]
    pub const fn support(&self) -> LagunaExactStorageSupport {
        LagunaExactStorageSupport::RuntimeReady
    }
}

/// Exact native or two-level NVFP4 profile without implying scale conversion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LagunaNvfp4Profile {
    support: LagunaExactStorageSupport,
}

impl LagunaNvfp4Profile {
    pub(super) const fn native() -> Self {
        Self {
            support: LagunaExactStorageSupport::RuntimeReady,
        }
    }

    pub(super) const fn two_level() -> Self {
        Self {
            support: LagunaExactStorageSupport::FutureExactKernel,
        }
    }

    #[must_use]
    pub const fn bits(&self) -> u32 {
        4
    }

    #[must_use]
    pub const fn group_size(&self) -> u32 {
        16
    }

    #[must_use]
    pub const fn support(&self) -> LagunaExactStorageSupport {
        self.support
    }
}

/// Exact block-FP8 storage profile awaiting a kernel that consumes block scales.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LagunaBlockFp8Profile {
    block_row_extent: u32,
    block_column_extent: u32,
}

impl LagunaBlockFp8Profile {
    pub(super) const fn evidenced() -> Self {
        Self {
            block_row_extent: 128,
            block_column_extent: 128,
        }
    }

    #[must_use]
    pub const fn support(&self) -> LagunaExactStorageSupport {
        LagunaExactStorageSupport::FutureExactKernel
    }

    #[must_use]
    pub const fn block_row_extent(&self) -> u32 {
        self.block_row_extent
    }

    #[must_use]
    pub const fn block_column_extent(&self) -> u32 {
        self.block_column_extent
    }
}

impl LagunaDirectAffineStorageDescriptor {
    pub(super) fn new(
        default_profile: LagunaAffineProfile,
        module_overrides: BTreeMap<String, LagunaAffineProfile>,
    ) -> Self {
        Self {
            default_profile,
            module_overrides,
        }
    }

    /// Returns the affine profile inherited by modules without an override.
    #[must_use]
    pub const fn default_profile(&self) -> LagunaAffineProfile {
        self.default_profile
    }

    /// Returns the number of canonical module overrides.
    #[must_use]
    pub fn module_override_count(&self) -> usize {
        self.module_overrides.len()
    }

    /// Returns a canonical module's explicit profile or the default profile.
    #[must_use]
    pub fn profile_for_module(&self, canonical_module_name: &str) -> LagunaAffineProfile {
        self.module_overrides
            .get(canonical_module_name)
            .copied()
            .unwrap_or(self.default_profile)
    }

    /// Returns explicit wrapper-free overrides for exact executable-module resolution.
    pub(crate) const fn module_overrides(&self) -> &BTreeMap<String, LagunaAffineProfile> {
        &self.module_overrides
    }
}

/// Canonical storage forms decidable from configuration without tensor inventory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LagunaStorageDescriptor {
    Unquantized,
    DirectAffine(LagunaDirectAffineStorageDescriptor),
    NativeNvfp4(LagunaNvfp4Profile),
    Compressed(LagunaCompressedStorageDescriptor),
}

impl LagunaStorageDescriptor {
    pub(crate) fn has_fp8_kv_cache(&self) -> bool {
        matches!(self, Self::Compressed(descriptor) if descriptor.kv_cache().is_some())
    }
}
