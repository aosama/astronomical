//! The operating system a native build targets, read from its target triple.
//!
//! The native runtime builds with a different toolchain on each operating
//! system: Xcode and the macOS SDK on macOS, the host's C and C++ compilers on
//! Linux. The identity must fingerprint the toolchain that actually compiles
//! the archives, so the probe set follows the target's operating system. The
//! choice is made from the target triple, never from the process that runs
//! the probes, so `cargo xtask native-fingerprint` and the Cargo build script
//! agree for the same triple.

/// An operating system the pinned native runtime can be built for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeTargetPlatform {
    /// Apple silicon macOS: the Metal backend, built with Xcode.
    MacOs,
    /// Linux: MLX's CPU backend, built with the host's C and C++ compilers.
    /// Android is not Linux here: it has its own C library and toolchain.
    Linux,
}

impl NativeTargetPlatform {
    /// Reads the operating system from a Rust target triple.
    ///
    /// # Errors
    ///
    /// Returns [`UnsupportedTargetPlatform`] for any operating system other
    /// than macOS or Linux, so an unsupported target fails before a single
    /// toolchain probe runs.
    pub fn of_target_triple(target_triple: &str) -> Result<Self, UnsupportedTargetPlatform> {
        let triple_segments: Vec<&str> = target_triple.trim().split('-').collect();
        if triple_segments.contains(&"darwin") && triple_segments.contains(&"apple") {
            Ok(Self::MacOs)
        } else if triple_segments.contains(&"linux") && !triple_segments.contains(&"android") {
            Ok(Self::Linux)
        } else {
            Err(UnsupportedTargetPlatform {
                target_triple: target_triple.to_owned(),
            })
        }
    }
}

/// A target triple whose operating system has no native recipe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedTargetPlatform {
    pub target_triple: String,
}

impl std::fmt::Display for UnsupportedTargetPlatform {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "the native runtime builds only for macOS or Linux targets, got {:?}",
            self.target_triple
        )
    }
}

impl std::error::Error for UnsupportedTargetPlatform {}
