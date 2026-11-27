//! Every difference between the macOS and Linux native build recipes.
//!
//! # Why one module
//!
//! The pinned runtime builds the Metal backend with Xcode on macOS, MLX's
//! CPU backend with the host's compilers on Linux (issue #2424), and MLX's
//! CUDA backend beside it on a Linux host with the CUDA toolkit and an NVIDIA
//! GPU (the `LinuxCuda` recipe, issue #2699). Keeping each
//! difference here, behind one [`NativeTargetPlatform`] value, keeps
//! operating-system checks out of the compile, store and link executors, and
//! gives later backends (Vulkan, Windows) one place to add a recipe.
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
//!
//! # Changing a recipe
//!
//! The configure arguments and published products here are executor code,
//! outside the native build identity's source pathspecs. Changing either for
//! any platform must bump `native-build-store-schema-version`, as the identity
//! policy in `native-identity` requires, or existing store entries built with
//! the old recipe would be reused.

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

/// Whether a published product is one file or a folder of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeProductKind {
    File,
    /// Copied whole, symlinks refused, like the MLX headers.
    Directory,
}

/// A product a platform's build publishes into the store beside the static
/// archives and headers every platform shares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlatformNativeProduct {
    /// Where CMake writes the product, relative to the CMake build directory.
    pub build_relative_path: &'static str,
    /// Where the store publishes it, relative to the entry directory.
    pub entry_relative_path: &'static str,
    pub description: &'static str,
    pub kind: NativeProductKind,
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
    kind: NativeProductKind::File,
}];

const MLX_TARGETS_PRODUCT: PlatformNativeProduct = PlatformNativeProduct {
    build_relative_path: "mlx-package/share/cmake/MLX/MLXTargets.cmake",
    entry_relative_path: MLX_TARGETS_RELATIVE_PATH,
    description: "published MLX CMake package targets",
    kind: NativeProductKind::File,
};

const LINUX_NATIVE_PRODUCTS: &[PlatformNativeProduct] = &[MLX_TARGETS_PRODUCT];

/// The CUDA and cuDNN libraries to link, one linker input per line, which the
/// native build resolves because MLX's package names them only as CMake
/// targets (`native/cuda_runtime_facts.cmake`).
pub const CUDA_LINK_LIBRARIES_RELATIVE_PATH: &str = "share/astronomical/cuda-link-libraries.txt";
/// The CUDA runtime headers' folder, for MLX's run-time CUDA compiler.
pub const CUDA_TOOLKIT_INCLUDE_DIRECTORY_RELATIVE_PATH: &str =
    "share/astronomical/cuda-toolkit-include-directory.txt";

/// The JIT header folder as a literal, so each header product's path is
/// built from it with `concat!` instead of repeating it.
macro_rules! cuda_jit_headers_relative_directory {
    () => {
        "share/mlx/cuda-jit-headers"
    };
}

/// The CUTLASS, CuTe and CCCL headers MLX installs for its run-time CUDA
/// compiler, which a statically linked MLX cannot find beside itself, so the
/// runtime names this folder to MLX (`mlx_cuda_set_jit_header_directories`).
pub const CUDA_JIT_HEADERS_RELATIVE_DIRECTORY: &str = cuda_jit_headers_relative_directory!();

const LINUX_CUDA_NATIVE_PRODUCTS: &[PlatformNativeProduct] = &[
    MLX_TARGETS_PRODUCT,
    PlatformNativeProduct {
        build_relative_path: "cuda/link-libraries.txt",
        entry_relative_path: CUDA_LINK_LIBRARIES_RELATIVE_PATH,
        description: "published CUDA link libraries",
        kind: NativeProductKind::File,
    },
    PlatformNativeProduct {
        build_relative_path: "cuda/toolkit-include-directory.txt",
        entry_relative_path: CUDA_TOOLKIT_INCLUDE_DIRECTORY_RELATIVE_PATH,
        description: "published CUDA toolkit include directory",
        kind: NativeProductKind::File,
    },
    PlatformNativeProduct {
        build_relative_path: "mlx-package/include/cute",
        entry_relative_path: concat!(cuda_jit_headers_relative_directory!(), "/cute"),
        description: "published CuTe headers for MLX's run-time CUDA compiler",
        kind: NativeProductKind::Directory,
    },
    PlatformNativeProduct {
        build_relative_path: "mlx-package/include/cutlass",
        entry_relative_path: concat!(cuda_jit_headers_relative_directory!(), "/cutlass"),
        description: "published CUTLASS headers for MLX's run-time CUDA compiler",
        kind: NativeProductKind::Directory,
    },
    PlatformNativeProduct {
        build_relative_path: "mlx-package/include/cccl",
        entry_relative_path: concat!(cuda_jit_headers_relative_directory!(), "/cccl"),
        description: "published CCCL headers for MLX's run-time CUDA compiler",
        kind: NativeProductKind::Directory,
    },
];

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
        .map(NativeTargetPlatform::with_host_gpu_backend)
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
    native_target_platform.is_linux()
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
    platform_recipe_facts(native_target_platform).identity_environment_variables
}

const BLAS_SEARCH_VARIABLE_COUNT: usize = native_identity::probes::BLAS_SEARCH_VARIABLES.len();
const CUDA_SEARCH_VARIABLE_COUNT: usize = native_identity::probes::CUDA_SEARCH_VARIABLES.len();

const LINUX_IDENTITY_ENVIRONMENT_VARIABLES: [&str; BLAS_SEARCH_VARIABLE_COUNT] =
    native_identity::probes::BLAS_SEARCH_VARIABLES;

/// A CUDA build also hashes the CUDA toolkit search settings.
const LINUX_CUDA_IDENTITY_ENVIRONMENT_VARIABLES: [&str;
    BLAS_SEARCH_VARIABLE_COUNT + CUDA_SEARCH_VARIABLE_COUNT] = concatenate_variable_names(
    native_identity::probes::BLAS_SEARCH_VARIABLES,
    native_identity::probes::CUDA_SEARCH_VARIABLES,
);

const fn concatenate_variable_names(
    blas_variables: [&'static str; BLAS_SEARCH_VARIABLE_COUNT],
    cuda_variables: [&'static str; CUDA_SEARCH_VARIABLE_COUNT],
) -> [&'static str; BLAS_SEARCH_VARIABLE_COUNT + CUDA_SEARCH_VARIABLE_COUNT] {
    let mut variable_names = [""; BLAS_SEARCH_VARIABLE_COUNT + CUDA_SEARCH_VARIABLE_COUNT];
    let mut variable_index = 0;
    while variable_index < blas_variables.len() {
        variable_names[variable_index] = blas_variables[variable_index];
        variable_index += 1;
    }
    let mut cuda_index = 0;
    while cuda_index < cuda_variables.len() {
        variable_names[variable_index + cuda_index] = cuda_variables[cuda_index];
        cuda_index += 1;
    }
    variable_names
}

/// The products a platform publishes beyond the shared archives and headers.
pub const fn platform_native_products(
    native_target_platform: NativeTargetPlatform,
) -> &'static [PlatformNativeProduct] {
    platform_recipe_facts(native_target_platform).native_products
}

/// What one platform recipe fingerprints from the environment and publishes
/// beyond what every recipe shares, kept in one table so a recipe is one
/// entry and every per-recipe lookup reads the same row.
struct PlatformRecipeFacts {
    identity_environment_variables: &'static [&'static str],
    native_products: &'static [PlatformNativeProduct],
}

const MACOS_RECIPE_FACTS: PlatformRecipeFacts = PlatformRecipeFacts {
    identity_environment_variables: &[],
    native_products: MACOS_NATIVE_PRODUCTS,
};
const LINUX_RECIPE_FACTS: PlatformRecipeFacts = PlatformRecipeFacts {
    identity_environment_variables: &LINUX_IDENTITY_ENVIRONMENT_VARIABLES,
    native_products: LINUX_NATIVE_PRODUCTS,
};
const LINUX_CUDA_RECIPE_FACTS: PlatformRecipeFacts = PlatformRecipeFacts {
    identity_environment_variables: &LINUX_CUDA_IDENTITY_ENVIRONMENT_VARIABLES,
    native_products: LINUX_CUDA_NATIVE_PRODUCTS,
};

const fn platform_recipe_facts(
    native_target_platform: NativeTargetPlatform,
) -> &'static PlatformRecipeFacts {
    match native_target_platform {
        NativeTargetPlatform::MacOs => &MACOS_RECIPE_FACTS,
        NativeTargetPlatform::Linux => &LINUX_RECIPE_FACTS,
        NativeTargetPlatform::LinuxCuda => &LINUX_CUDA_RECIPE_FACTS,
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
        NativeTargetPlatform::MacOs => macos_compiler_configure_arguments(),
        NativeTargetPlatform::Linux => Ok(linux_compiler_configure_arguments()),
        NativeTargetPlatform::LinuxCuda => host_linux_cuda_compiler_configure_arguments(),
    }
}

/// The macOS half of [`compiler_configure_arguments`]: Xcode's compilers and
/// SDK, resolved through `xcrun`, for arm64.
fn macos_compiler_configure_arguments() -> Result<Vec<String>, Box<dyn Error>> {
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

/// The Linux CUDA half of [`compiler_configure_arguments`], for this host's
/// detected CUDA build target.
fn host_linux_cuda_compiler_configure_arguments() -> Result<Vec<String>, Box<dyn Error>> {
    let cuda_build_target = native_identity::linux_gpu_backend::detected_cuda_build_target()
        .ok_or("the Linux CUDA recipe was chosen, but this host has no CUDA build target")?;
    Ok(linux_cuda_compiler_configure_arguments(cuda_build_target))
}

/// The Linux half of [`compiler_configure_arguments`], which runs no tool.
pub fn linux_compiler_configure_arguments() -> Vec<String> {
    vec![
        "-DCMAKE_C_COMPILER=cc".to_owned(),
        "-DCMAKE_CXX_COMPILER=c++".to_owned(),
    ]
}

/// The Linux CUDA recipe's configure arguments: the Linux compilers, then
/// MLX's CUDA backend with the detected CUDA compiler, the host C++ compiler
/// as its host compiler (the one the identity probed), and the detected GPU
/// architectures.
pub fn linux_cuda_compiler_configure_arguments(
    cuda_build_target: &native_identity::linux_gpu_backend::CudaBuildTarget,
) -> Vec<String> {
    let mut configure_arguments = linux_compiler_configure_arguments();
    configure_arguments.extend([
        "-DASTRONOMICAL_MLX_BUILD_CUDA=ON".to_owned(),
        format!(
            "-DCMAKE_CUDA_COMPILER={}",
            cuda_build_target.cuda_compiler_path.display()
        ),
        "-DCMAKE_CUDA_HOST_COMPILER=c++".to_owned(),
        format!(
            "-DMLX_CUDA_ARCHITECTURES={}",
            cuda_build_target.cuda_architectures
        ),
    ]);
    configure_arguments
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
