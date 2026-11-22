//! Every difference between the macOS and Linux native build recipes.
//!
//! # Why one module
//!
//! The pinned runtime builds the Metal backend with Xcode on macOS and MLX's
//! CPU backend with the host's compilers on Linux (issue #2424). Keeping each
//! difference here, behind one [`NativeTargetPlatform`] value, keeps
//! operating-system checks out of the compile, store and link executors, and
//! gives later backends (Vulkan, CUDA, Windows) one place to add a recipe.
//! The macOS recipe is the one every existing store entry was built with, so
//! its arguments and products must stay exactly as they are.
//!
//! # Only native builds are supported
//!
//! CMake builds for the machine it runs on. A build for another operating
//! system would publish archives the target cannot link, so the platform is
//! accepted only when the target triple's operating system is the host's. On
//! Linux nothing pins the architecture or the C library either, so a Linux
//! target must be the host's exact triple. (macOS pins `arm64` itself, and its
//! accepted triples are left exactly as they were.)

use std::{error::Error, path::PathBuf, process::Command};

pub use native_identity::target_platform::NativeTargetPlatform;

/// Where a Linux build installs MLX's own CMake package, relative to the CMake
/// build directory. MLX is built in-tree, so its install step runs on its
/// subproject alone after the compile.
pub const MLX_PACKAGE_INSTALL_RELATIVE_DIRECTORY: &str = "mlx-package";
/// MLX's subproject build directory inside the CMake build directory.
pub const MLX_SUBPROJECT_BUILD_RELATIVE_DIRECTORY: &str = "_deps/mlx-build";
/// The exported targets file of MLX's installed CMake package, relative to a
/// store entry. It records the libraries MLX was configured to link, which a
/// Linux link reads back (see `native_link_directives`).
pub const MLX_TARGETS_RELATIVE_PATH: &str = "share/cmake/MLX/MLXTargets.cmake";

/// A product a platform's build publishes into the store beside the static
/// archives and headers every platform shares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlatformNativeProduct {
    /// Where CMake writes the product, relative to the CMake build directory.
    pub build_relative_path: &'static str,
    /// Where the store publishes it, relative to the entry directory.
    pub entry_relative_path: &'static str,
    pub description: &'static str,
}

impl PlatformNativeProduct {
    /// The product's entry path and description, as store validation checks
    /// them.
    #[must_use]
    pub const fn entry_path_and_description(&self) -> (&'static str, &'static str) {
        (self.entry_relative_path, self.description)
    }
}

const MACOS_NATIVE_PRODUCTS: &[PlatformNativeProduct] = &[PlatformNativeProduct {
    build_relative_path: "_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib",
    entry_relative_path: "share/mlx/mlx.metallib",
    description: "published MLX metallib",
}];

const LINUX_NATIVE_PRODUCTS: &[PlatformNativeProduct] = &[PlatformNativeProduct {
    build_relative_path: "mlx-package/share/cmake/MLX/MLXTargets.cmake",
    entry_relative_path: MLX_TARGETS_RELATIVE_PATH,
    description: "published MLX CMake package targets",
}];

/// Resolves the platform to build for and refuses a build the host cannot
/// produce archives for.
///
/// # Errors
///
/// Fails for a target operating system without a native recipe, when the
/// target's operating system differs from the host's, or for a Linux target
/// that is not the host's exact triple.
pub fn native_build_platform(
    target_triple: &str,
    host_triple: &str,
) -> Result<NativeTargetPlatform, Box<dyn Error>> {
    native_build_platform_for_host(target_triple, host_triple, host_platform())
}

/// The host decision as a pure function, so every refusal is testable on
/// any machine.
pub fn native_build_platform_for_host(
    target_triple: &str,
    host_triple: &str,
    host_platform: Option<NativeTargetPlatform>,
) -> Result<NativeTargetPlatform, Box<dyn Error>> {
    let target_platform = NativeTargetPlatform::of_target_triple(target_triple)?;
    if host_platform != Some(target_platform) {
        return Err(format!(
            "the native runtime builds only for the host's operating system; {target_triple} targets {target_platform:?} but the host is {host_platform:?}"
        )
        .into());
    }
    let is_foreign_linux_triple =
        target_platform == NativeTargetPlatform::Linux && target_triple != host_triple;
    if is_foreign_linux_triple {
        return Err(format!(
            "a Linux native runtime builds only for the host's own triple {host_triple}, not {target_triple}"
        )
        .into());
    }
    Ok(target_platform)
}

/// The operating system this process runs on. Build scripts and the
/// standalone tool are compiled for the host, so this is the build host.
pub const fn host_platform() -> Option<NativeTargetPlatform> {
    if cfg!(target_os = "macos") {
        Some(NativeTargetPlatform::MacOs)
    } else if cfg!(target_os = "linux") {
        Some(NativeTargetPlatform::Linux)
    } else {
        None
    }
}

/// Whether a platform installs MLX's CMake package after the compile. Linux
/// does, so its link reads the libraries MLX chose from MLX's own published
/// package; the Metal build links a fixed framework set and needs none.
pub const fn installs_mlx_package(native_target_platform: NativeTargetPlatform) -> bool {
    matches!(native_target_platform, NativeTargetPlatform::Linux)
}

/// Environment variables the platform's native build identity reads, beyond
/// the ones every build script declares. Cargo must re-run the build script
/// when any of them changes, or a new BLAS would keep linking the old entry.
/// Linux hashes the BLAS search settings; macOS reads none of them. `PATH` is
/// left out: the identity already records the `cc` and `c++` it finds, and
/// rustup and IDEs alter `PATH` between processes, which would re-run the
/// build script with nothing changed.
pub const fn identity_environment_variables(
    native_target_platform: NativeTargetPlatform,
) -> &'static [&'static str] {
    match native_target_platform {
        NativeTargetPlatform::MacOs => &[],
        NativeTargetPlatform::Linux => &LINUX_IDENTITY_ENVIRONMENT_VARIABLES,
    }
}

const LINUX_IDENTITY_ENVIRONMENT_VARIABLES: [&str; 10] =
    native_identity::probes::BLAS_SEARCH_VARIABLES;

/// The products a platform publishes beyond the shared archives and headers.
pub const fn platform_native_products(
    native_target_platform: NativeTargetPlatform,
) -> &'static [PlatformNativeProduct] {
    match native_target_platform {
        NativeTargetPlatform::MacOs => MACOS_NATIVE_PRODUCTS,
        NativeTargetPlatform::Linux => LINUX_NATIVE_PRODUCTS,
    }
}

/// The CMake configure arguments that select the compilers and, on macOS,
/// the SDK and architecture.
///
/// macOS resolves the toolchain through `xcrun` rather than trusting `PATH`
/// so the compilers are the ones the identity's Xcode probes measured. Linux
/// names `cc` and `c++`, the exact commands the identity's Linux probes run,
/// so CMake resolves the same compilers from `PATH`.
///
/// # Errors
///
/// Fails on macOS when `xcrun` cannot resolve a tool or the SDK.
pub fn compiler_configure_arguments(
    native_target_platform: NativeTargetPlatform,
) -> Result<Vec<String>, Box<dyn Error>> {
    match native_target_platform {
        NativeTargetPlatform::MacOs => {
            let clang_compiler_path = discover_xcrun_path(&["--find", "clang"])?;
            let clang_cxx_compiler_path = discover_xcrun_path(&["--find", "clang++"])?;
            let macos_sdk_path = discover_xcrun_path(&["--sdk", "macosx", "--show-sdk-path"])?;
            Ok(vec![
                format!("-DCMAKE_C_COMPILER={}", clang_compiler_path.display()),
                format!("-DCMAKE_CXX_COMPILER={}", clang_cxx_compiler_path.display()),
                format!("-DCMAKE_OSX_SYSROOT={}", macos_sdk_path.display()),
                "-DCMAKE_OSX_ARCHITECTURES=arm64".to_owned(),
            ])
        }
        NativeTargetPlatform::Linux => Ok(linux_compiler_configure_arguments()),
    }
}

/// The Linux half of [`compiler_configure_arguments`], which runs no tool.
pub fn linux_compiler_configure_arguments() -> Vec<String> {
    vec![
        "-DCMAKE_C_COMPILER=cc".to_owned(),
        "-DCMAKE_CXX_COMPILER=c++".to_owned(),
    ]
}

pub(crate) fn discover_xcrun_path(arguments: &[&str]) -> Result<PathBuf, Box<dyn Error>> {
    let path_output = Command::new("xcrun").args(arguments).output()?;
    if !path_output.status.success() {
        return Err(format!("xcrun failed to resolve {}", arguments.join(" ")).into());
    }
    let path_text = String::from_utf8(path_output.stdout)?;
    let resolved_path = PathBuf::from(path_text.trim());
    if !resolved_path.is_absolute() {
        return Err(format!(
            "xcrun reported a non-absolute path for {}",
            arguments.join(" ")
        )
        .into());
    }
    Ok(resolved_path)
}

#[cfg(test)]
mod tests;
