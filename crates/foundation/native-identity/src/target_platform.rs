//! The operating system a native build targets, read from its target triple.
//!
//! The native runtime builds with a different toolchain on each operating
//! system: Xcode and the macOS SDK on macOS, the host's C and C++ compilers on
//! Linux. The identity must fingerprint the toolchain that actually compiles
//! the archives, so the probe set follows the target's operating system. The
//! operating system is read from the target triple, never from the process
//! that runs the probes, so `cargo xtask native-fingerprint` and the Cargo
//! build script agree for the same triple. The one host fact on top of it is
//! the GPU backend: a Linux target builds MLX's CUDA backend when the build
//! host can ([`NativeTargetPlatform::with_host_gpu_backend`]).

/// An operating system the pinned native runtime can be built for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeTargetPlatform {
    /// Apple silicon macOS: the Metal backend, built with Xcode.
    MacOs,
    /// Linux: MLX's CPU backend, built with the host's C and C++ compilers.
    /// Android is not Linux here: it has its own C library and toolchain.
    Linux,
    /// Linux with MLX's CUDA backend beside the CPU one: chosen for a Linux
    /// target when the build host has the CUDA toolkit and an NVIDIA GPU
    /// (see `linux_gpu_backend`). Everything that holds for `Linux` holds
    /// here too, plus the CUDA compiler, libraries and kernel headers.
    LinuxCuda,
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

impl NativeTargetPlatform {
    /// The platform recipe a native build for `target_triple` uses on this
    /// host: the triple's operating system, with the host's GPU backend
    /// applied ([`Self::with_host_gpu_backend`]).
    ///
    /// # Errors
    ///
    /// Returns [`UnsupportedTargetPlatform`] as [`Self::of_target_triple`]
    /// does.
    pub fn for_build(target_triple: &str) -> Result<Self, UnsupportedTargetPlatform> {
        Ok(Self::of_target_triple(target_triple)?.with_host_gpu_backend())
    }

    /// The recipe with this build host's GPU backend: a Linux target is
    /// promoted to [`Self::LinuxCuda`] when the host can build MLX's CUDA
    /// backend. The identity probes and the build orchestration both promote
    /// through this one function, so they always agree on the recipe.
    #[must_use]
    pub fn with_host_gpu_backend(self) -> Self {
        let has_cuda_build_target =
            crate::linux_gpu_backend::detected_cuda_build_target().is_some();
        self.with_gpu_backend(has_cuda_build_target)
    }

    /// The recipe once the host's GPU backend is known: a Linux target with
    /// a CUDA build target becomes [`Self::LinuxCuda`]; nothing else changes.
    #[must_use]
    pub const fn with_gpu_backend(self, has_cuda_build_target: bool) -> Self {
        match self {
            Self::Linux if has_cuda_build_target => Self::LinuxCuda,
            other_platform => other_platform,
        }
    }

    /// Whether the recipe builds for Linux, with or without CUDA.
    #[must_use]
    pub const fn is_linux(self) -> bool {
        matches!(self, Self::Linux | Self::LinuxCuda)
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
