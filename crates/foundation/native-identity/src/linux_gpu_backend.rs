//! Which GPU backend a Linux native build compiles MLX with: CUDA when the
//! build host has the CUDA toolkit and an NVIDIA GPU, otherwise the CPU
//! backend alone (issue #2699).
//!
//! # Why detected, not configured
//!
//! The choice follows what the machine can build and run, the way the
//! compilers do: a host with `nvcc` on `PATH` and a GPU that `nvidia-smi`
//! reports builds the CUDA backend; any other Linux host keeps the CPU
//! backend it always had. No environment variable or setting is involved, and
//! the identity fingerprints what was detected (see `probes`), so installing
//! or upgrading the toolkit, or moving to another GPU, keys a new store entry
//! instead of reusing one built for something else.
//!
//! # Why hosts below MLX's floor keep the CPU backend
//!
//! MLX's CUDA backend needs CUDA 12 or later and a GPU of compute capability
//! 7.5 or later. A host with an older toolkit or GPU builds the CPU backend
//! it always built, instead of choosing a CUDA build that cannot succeed;
//! with several GPUs, only those MLX supports are compiled for.
//!
//! # Why the GPU's own architecture
//!
//! MLX compiles its CUDA kernels for the architectures it is given, and by
//! default asks the GPU of the build host. The same answer is read here
//! through `nvidia-smi`, so the identity can record it and the build can
//! name it explicitly rather than leave the query to MLX's configure.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

/// The oldest CUDA toolkit MLX's CUDA backend builds with.
const MINIMUM_CUDA_COMPILER_MAJOR_VERSION: u32 = 12;
/// The oldest GPU architecture MLX's CUDA backend runs on (compute
/// capability 7.5, Turing).
const MINIMUM_CUDA_ARCHITECTURE_NUMBER: u32 = 75;

/// What a Linux CUDA build needs to know: the CUDA compiler CMake is given
/// and the architectures MLX compiles its kernels for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaBuildTarget {
    pub cuda_compiler_path: PathBuf,
    /// MLX's `MLX_CUDA_ARCHITECTURES` value, such as `86` or `86;90a`.
    pub cuda_architectures: String,
}

/// The Linux build host's CUDA build target, detected once per process; `None`
/// when the host lacks the toolkit or a GPU.
pub fn detected_cuda_build_target() -> Option<&'static CudaBuildTarget> {
    static DETECTED: OnceLock<Option<CudaBuildTarget>> = OnceLock::new();
    DETECTED.get_or_init(detect_cuda_build_target).as_ref()
}

fn detect_cuda_build_target() -> Option<CudaBuildTarget> {
    let path_directories = std::env::var_os("PATH")?;
    let cuda_compiler_path = find_executable_in(std::env::split_paths(&path_directories), "nvcc")?;
    let cuda_compiler_version_output = Command::new(&cuda_compiler_path)
        .arg("--version")
        .output()
        .ok()?;
    let cuda_compiler_major_version = cuda_compiler_major_version_from_version_output(
        &String::from_utf8_lossy(&cuda_compiler_version_output.stdout),
    )?;
    if cuda_compiler_major_version < MINIMUM_CUDA_COMPILER_MAJOR_VERSION {
        return None;
    }
    let gpu_query_output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !gpu_query_output.status.success() {
        return None;
    }
    let cuda_architectures = cuda_architectures_from_compute_capabilities(
        &String::from_utf8_lossy(&gpu_query_output.stdout),
    )?;
    Some(CudaBuildTarget {
        cuda_compiler_path,
        cuda_architectures,
    })
}

/// The CUDA compiler's major version from `nvcc --version`, read from its
/// `release 13.0, V13.0.88` line. `None` when no such line parses, so an
/// unrecognized compiler keeps the CPU backend.
pub fn cuda_compiler_major_version_from_version_output(version_output: &str) -> Option<u32> {
    for version_line in version_output.lines() {
        let Some((_, release_text)) = version_line.split_once("release ") else {
            continue;
        };
        let (major_text, _) = release_text.split_once('.')?;
        return major_text.trim().parse().ok();
    }
    None
}

/// The first `executable_name` among `search_directories` that is a file.
pub fn find_executable_in(
    search_directories: impl IntoIterator<Item = PathBuf>,
    executable_name: &str,
) -> Option<PathBuf> {
    for search_directory in search_directories {
        let candidate_path = search_directory.join(executable_name);
        if is_file(&candidate_path) {
            return Some(candidate_path);
        }
    }
    None
}

fn is_file(candidate_path: &Path) -> bool {
    candidate_path
        .metadata()
        .is_ok_and(|metadata| metadata.is_file())
}

/// MLX's architecture list from `nvidia-smi`'s compute capabilities, one per
/// GPU (`8.6`): each becomes `86`, and from 9.0 on the architecture-specific
/// form (`90a`) MLX itself chooses for the native GPU. Repeated GPUs are
/// listed once, in the order first seen, and GPUs older than MLX supports
/// are left out. `None` when no line is a supported compute capability, so a
/// host whose driver reports nothing, or only older GPUs, keeps the CPU
/// backend.
pub fn cuda_architectures_from_compute_capabilities(gpu_query_output: &str) -> Option<String> {
    let mut cuda_architectures: Vec<String> = Vec::new();
    for compute_capability_line in gpu_query_output.lines() {
        let Some(cuda_architecture) = cuda_architecture_of(compute_capability_line.trim()) else {
            continue;
        };
        if !cuda_architectures.contains(&cuda_architecture) {
            cuda_architectures.push(cuda_architecture);
        }
    }
    if cuda_architectures.is_empty() {
        None
    } else {
        Some(cuda_architectures.join(";"))
    }
}

fn cuda_architecture_of(compute_capability: &str) -> Option<String> {
    let (major_text, minor_text) = compute_capability.split_once('.')?;
    let major_version: u32 = major_text.parse().ok()?;
    let minor_version: u32 = minor_text.parse().ok()?;
    let architecture_number = major_version * 10 + minor_version;
    if architecture_number < MINIMUM_CUDA_ARCHITECTURE_NUMBER {
        return None;
    }
    // MLX appends `a` from 9.0 on (mlx/backend/cuda/CMakeLists.txt) to use the
    // architecture-specific features of the GPU it builds for.
    let architecture_suffix = if architecture_number >= 90 { "a" } else { "" };
    Some(format!("{architecture_number}{architecture_suffix}"))
}

/// The standard folders, after the CUDA toolkit's own, where CMake's search
/// for `cudnn.h` (MLX's `FindCUDNN`) finds a system cuDNN package.
const SYSTEM_CUDNN_INCLUDE_DIRECTORIES: [&str; 2] = ["/usr/local/include", "/usr/include"];

/// The cuDNN version MLX's configure will compile its cuDNN frontend against,
/// as `major.minor.patch`, from the first `cudnn_version.h` in the folders
/// MLX's `FindCUDNN` searches, in its order: the CUDA toolkit's include
/// folder beside `cuda_compiler_path`, then the system folders. `None` when
/// no cuDNN is installed, in which case MLX's configure stops the build.
pub fn installed_cudnn_version(cuda_compiler_path: &Path) -> Option<String> {
    let mut include_directories: Vec<PathBuf> = Vec::new();
    if let Some(cuda_toolkit_directory) = cuda_compiler_path.parent().and_then(Path::parent) {
        include_directories.push(cuda_toolkit_directory.join("include"));
    }
    for system_include_directory in SYSTEM_CUDNN_INCLUDE_DIRECTORIES {
        include_directories.push(PathBuf::from(system_include_directory));
    }
    for include_directory in include_directories {
        let Ok(version_header_text) =
            std::fs::read_to_string(include_directory.join("cudnn_version.h"))
        else {
            continue;
        };
        return cudnn_version_from_version_header(&version_header_text);
    }
    None
}

/// `major.minor.patch` from the `CUDNN_MAJOR`, `CUDNN_MINOR` and
/// `CUDNN_PATCHLEVEL` definitions of `cudnn_version.h`; `None` when any is
/// missing or not a number.
pub fn cudnn_version_from_version_header(version_header_text: &str) -> Option<String> {
    let major_version = defined_number(version_header_text, "CUDNN_MAJOR")?;
    let minor_version = defined_number(version_header_text, "CUDNN_MINOR")?;
    let patch_level = defined_number(version_header_text, "CUDNN_PATCHLEVEL")?;
    Some(format!("{major_version}.{minor_version}.{patch_level}"))
}

fn defined_number(header_text: &str, macro_name: &str) -> Option<u32> {
    for header_line in header_text.lines() {
        let mut line_words = header_line.split_whitespace();
        if line_words.next() == Some("#define") && line_words.next() == Some(macro_name) {
            return line_words.next()?.parse().ok();
        }
    }
    None
}
