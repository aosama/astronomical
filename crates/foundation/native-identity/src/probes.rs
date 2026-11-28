//! Full-mode toolchain probes: the hosted toolchain identity folded into
//! the native build fingerprint. The probe set follows the target's operating
//! system (see [`crate::target_platform`]): macOS probes Xcode, the SDK and
//! Xcode's Clang; Linux probes the `cc` and `c++` compilers CMake uses there,
//! and a Linux CUDA build also the CUDA compiler, the GPU architectures and
//! the CUDA toolkit search settings.
//! Both probe CMake and rustc. Digest semantics are carried over from the
//! shell era byte-for-byte so `github_actions_cache` keys stay meaningful: whole-output probes
//! hash raw stdout (trailing newline included), composite probes hash the
//! re-composed `version=…`/`build=…` text, and overrides hash the override
//! text plus one trailing newline.

use std::process::Command;
use std::time::Instant;

use crate::target_platform::NativeTargetPlatform;
use crate::{FingerprintRequest, NativeIdentityError, digest_bytes};

/// Toolchain identity inputs, explicit per caller. Fixtures inject these
/// fields directly; production callers supply only the target triple so the
/// identity fingerprints the toolchain CMake actually uses.
#[derive(Debug, Default, Clone)]
pub struct ToolchainOverrides {
    pub xcode: Option<String>,
    pub sdk: Option<String>,
    pub clang: Option<String>,
    /// Linux only: the whole `cc --version` output.
    pub c_compiler: Option<String>,
    /// Linux only: the whole `c++ --version` output.
    pub cxx_compiler: Option<String>,
    /// Linux only: the CMake search settings that decide which BLAS and
    /// LAPACK MLX's configure finds, as `NAME=value` lines.
    pub blas_search: Option<String>,
    /// Linux only: whether the host builds MLX's CUDA backend. `None` asks
    /// the host (`linux_gpu_backend`); fixtures say it outright so their
    /// identity never depends on the machine running them.
    pub has_cuda_build_target: Option<bool>,
    /// Linux CUDA only: the whole `nvcc --version` output.
    pub cuda_compiler: Option<String>,
    /// Linux CUDA only: MLX's CUDA architecture list, such as `86`.
    pub cuda_architectures: Option<String>,
    /// Linux CUDA only: the CUDA toolkit search settings, as `NAME=value`
    /// lines.
    pub cuda_search: Option<String>,
    /// Linux CUDA only: the installed cuDNN's `major.minor.patch`.
    pub cudnn_version: Option<String>,
    pub cmake: Option<String>,
    pub rustc: Option<String>,
    pub target_triple: Option<String>,
    pub build_type: Option<String>,
}

impl ToolchainOverrides {
    pub fn none() -> Self {
        Self::default()
    }
}

/// One captured toolchain-probe value: the exact bytes that feed the identity
/// digest, plus whether the value came from an explicit override instead of
/// the live toolchain. `--explain` surfaces these so an identity flap is
/// attributable in one log read.
#[derive(Debug, PartialEq, Eq)]
pub struct ProbeValue {
    pub name: &'static str,
    pub value: String,
    pub from_override: bool,
}

/// Captures the toolchain inputs in the fixed order the shell era
/// established: target, build type, then the target platform's toolchain
/// probes. This is the single capture path — the identity manifest digests
/// exactly these values, so an explained value can never diverge from the
/// hashed one.
pub fn toolchain_probe_values(
    request: &FingerprintRequest<'_>,
) -> Result<Vec<ProbeValue>, NativeIdentityError> {
    let target_triple = request
        .overrides
        .target_triple
        .as_ref()
        .filter(|target_triple| !target_triple.trim().is_empty())
        .cloned()
        .ok_or(NativeIdentityError::MissingTargetTriple)?;
    let build_type = request
        .overrides
        .build_type
        .clone()
        .unwrap_or_else(|| "Release".to_owned());
    if build_type != "Release" {
        return Err(NativeIdentityError::UnsupportedBuildType(build_type));
    }
    let compiler_probe_values = platform_compiler_probe_values(&target_triple, &request.overrides)?;

    let mut probe_values = vec![
        ProbeValue {
            name: "target",
            value: target_triple,
            from_override: true,
        },
        ProbeValue {
            name: "build-type",
            value: build_type,
            from_override: true,
        },
    ];
    probe_values.extend(compiler_probe_values);
    probe_values.push(composite_cmake_probe_value(
        request.overrides.cmake.as_deref(),
    )?);
    probe_values.push(whole_output_probe_value(
        "rustc",
        request.overrides.rustc.as_deref(),
        "rustc",
        &["-vV"],
    )?);
    Ok(probe_values)
}

/// Captures the compiler probes of the target triple's operating system. An
/// unsupported operating system fails here, before any probe runs.
fn platform_compiler_probe_values(
    target_triple: &str,
    overrides: &ToolchainOverrides,
) -> Result<Vec<ProbeValue>, NativeIdentityError> {
    let target_platform = build_platform(target_triple, overrides)?;
    let mut compiler_probe_values = Vec::new();
    push_platform_compiler_probe_values(target_platform, overrides, &mut compiler_probe_values)?;
    Ok(compiler_probe_values)
}

/// The compiler probes of `target_platform`'s recipe; a CUDA build adds
/// its CUDA probes to the Linux ones.
fn push_platform_compiler_probe_values(
    target_platform: NativeTargetPlatform,
    overrides: &ToolchainOverrides,
    probe_values: &mut Vec<ProbeValue>,
) -> Result<(), NativeIdentityError> {
    match target_platform {
        NativeTargetPlatform::MacOs => push_macos_compiler_probe_values(overrides, probe_values),
        NativeTargetPlatform::Linux => push_linux_compiler_probe_values(overrides, probe_values),
        NativeTargetPlatform::LinuxCuda => {
            push_linux_compiler_probe_values(overrides, probe_values)?;
            push_cuda_probe_values(overrides, probe_values)
        }
    }
}

/// The recipe the build will use: the host's answer
/// ([`NativeTargetPlatform::for_build`]) unless a fixture states it.
fn build_platform(
    target_triple: &str,
    overrides: &ToolchainOverrides,
) -> Result<NativeTargetPlatform, NativeIdentityError> {
    let target_platform = match overrides.has_cuda_build_target {
        Some(has_cuda_build_target) => NativeTargetPlatform::of_target_triple(target_triple)
            .map_err(unsupported_platform_error)?
            .with_gpu_backend(has_cuda_build_target),
        None => {
            NativeTargetPlatform::for_build(target_triple).map_err(unsupported_platform_error)?
        }
    };
    Ok(target_platform)
}

fn unsupported_platform_error(
    unsupported_platform: crate::target_platform::UnsupportedTargetPlatform,
) -> NativeIdentityError {
    NativeIdentityError::Probe(unsupported_platform.to_string())
}

/// The CUDA probes of a Linux CUDA build:
///
/// - the whole `nvcc --version` output of the detected compiler, hashed for
///   the same reason the C compilers' is;
/// - the architectures, which decide which kernels the archive holds;
/// - the toolkit search settings, which decide which toolkit CMake finds,
///   whose files the store entry then links by path;
/// - the cuDNN version, because MLX compiles its cuDNN frontend against the
///   installed cuDNN's headers, so an upgrade must build a new entry.
fn push_cuda_probe_values(
    overrides: &ToolchainOverrides,
    probe_values: &mut Vec<ProbeValue>,
) -> Result<(), NativeIdentityError> {
    let cuda_compiler_program = match overrides.cuda_compiler {
        Some(_) => String::new(),
        None => required_cuda_build_target()?
            .cuda_compiler_path
            .display()
            .to_string(),
    };
    probe_values.push(whole_output_probe_value(
        "cuda-compiler",
        overrides.cuda_compiler.as_deref(),
        &cuda_compiler_program,
        &["--version"],
    )?);
    let cuda_architectures = match overrides.cuda_architectures.as_deref() {
        Some(override_value) => override_value.to_owned(),
        None => required_cuda_build_target()?.cuda_architectures.clone(),
    };
    probe_values.push(ProbeValue {
        name: "cuda-architectures",
        value: format!("{cuda_architectures}\n"),
        from_override: overrides.cuda_architectures.is_some(),
    });
    probe_values.push(search_settings_probe_value(
        "cuda-search",
        &CUDA_SEARCH_VARIABLES,
        overrides.cuda_search.as_deref(),
    ));
    let cudnn_version = match overrides.cudnn_version.as_deref() {
        Some(override_value) => override_value.to_owned(),
        None => match crate::linux_gpu_backend::installed_cudnn_version(
            &required_cuda_build_target()?.cuda_compiler_path,
        ) {
            Some(installed_cudnn_version) => installed_cudnn_version,
            None => "absent".to_owned(),
        },
    };
    probe_values.push(ProbeValue {
        name: "cudnn-version",
        value: format!("{cudnn_version}\n"),
        from_override: overrides.cudnn_version.is_some(),
    });
    Ok(())
}

/// The host's CUDA build target, which a CUDA probe without an override
/// reads; an error when the host has none.
fn required_cuda_build_target()
-> Result<&'static crate::linux_gpu_backend::CudaBuildTarget, NativeIdentityError> {
    crate::linux_gpu_backend::detected_cuda_build_target().ok_or(NativeIdentityError::Probe(
        "no CUDA build target on this host".to_owned(),
    ))
}

/// The macOS compiler probes: Xcode, the macOS SDK, and Xcode's Clang. Their
/// names, order and digest semantics are hashed bytes shared with every
/// existing macOS store entry, so they must not change.
fn push_macos_compiler_probe_values(
    overrides: &ToolchainOverrides,
    probe_values: &mut Vec<ProbeValue>,
) -> Result<(), NativeIdentityError> {
    probe_values.push(whole_output_probe_value(
        "xcode",
        overrides.xcode.as_deref(),
        "xcodebuild",
        &["-version"],
    )?);
    probe_values.push(composite_sdk_probe_value(overrides.sdk.as_deref())?);
    probe_values.push(composite_clang_probe_value(overrides.clang.as_deref())?);
    Ok(())
}

/// The Linux compiler probes. CMake configures Linux builds with the `cc`
/// and `c++` found on `PATH`, so those exact commands are probed. The whole
/// `--version` output is hashed because GCC's `-dumpversion` prints only the
/// major version, which would let a compiler update reuse stale archives.
fn push_linux_compiler_probe_values(
    overrides: &ToolchainOverrides,
    probe_values: &mut Vec<ProbeValue>,
) -> Result<(), NativeIdentityError> {
    probe_values.push(whole_output_probe_value(
        "c-compiler",
        overrides.c_compiler.as_deref(),
        "cc",
        &["--version"],
    )?);
    probe_values.push(whole_output_probe_value(
        "cxx-compiler",
        overrides.cxx_compiler.as_deref(),
        "c++",
        &["--version"],
    )?);
    probe_values.push(blas_search_probe_value(overrides.blas_search.as_deref()));
    Ok(())
}

/// The settings MLX's Linux configure consults to find BLAS and LAPACK:
/// CMake's own search paths, FindBLAS and FindLAPACK's vendor and root hints
/// (including MKL's), and the link and pkg-config paths FindBLAS searches.
/// `LD_LIBRARY_PATH` is left out on purpose: Cargo rewrites it for every
/// build script and `cargo run`, so hashing it would key a new store entry
/// per process.
/// macOS links Accelerate, so none of these apply there.
pub const BLAS_SEARCH_VARIABLES: [&str; 10] = [
    "CMAKE_PREFIX_PATH",
    "CMAKE_INCLUDE_PATH",
    "CMAKE_LIBRARY_PATH",
    "BLA_VENDOR",
    "BLAS_HOME",
    "BLAS_ROOT",
    "LAPACK_ROOT",
    "MKLROOT",
    "LIBRARY_PATH",
    "PKG_CONFIG_PATH",
];

/// The environment settings CMake's CUDAToolkit consults to find the CUDA
/// toolkit (`CUDA_PATH` is the one Linux distributions set). MLX's FindCUDNN
/// reads CMake variables, not the environment, so which cuDNN is used is
/// fingerprinted by its version instead.
pub const CUDA_SEARCH_VARIABLES: [&str; 3] = ["CUDA_PATH", "CUDAToolkit_ROOT", "CUDACXX"];

/// A Linux store entry records the BLAS and LAPACK files MLX's configure
/// found, and links them by path. Pointing CMake at another BLAS must
/// therefore build a new entry rather than reuse one whose recorded files
/// are elsewhere, so the settings that steer that search are hashed. Unset
/// settings hash as empty values, which keeps the probe present and stable.
fn blas_search_probe_value(override_text: Option<&str>) -> ProbeValue {
    search_settings_probe_value("blas-search", &BLAS_SEARCH_VARIABLES, override_text)
}

/// The `NAME=value` lines of `search_variables`, unset ones as empty values.
fn search_settings_probe_value(
    probe_name: &'static str,
    search_variables: &[&str],
    override_text: Option<&str>,
) -> ProbeValue {
    let probe_output = match override_text {
        Some(override_value) => format!("{override_value}\n"),
        None => {
            let mut search_setting_lines = String::new();
            for search_variable in search_variables {
                let search_value = std::env::var(search_variable).unwrap_or_default();
                search_setting_lines.push_str(&format!("{search_variable}={search_value}\n"));
            }
            search_setting_lines
        }
    };
    ProbeValue {
        name: probe_name,
        value: probe_output,
        from_override: override_text.is_some(),
    }
}

/// Assembles the toolchain section of the identity manifest in the fixed
/// order the shell era established: target, build type, then the target
/// platform's toolchain probes.
pub fn toolchain_manifest_lines(
    request: &FingerprintRequest<'_>,
) -> Result<String, NativeIdentityError> {
    let probe_values = toolchain_probe_values(request)?;
    Ok(manifest_lines_from_probe_values(&probe_values))
}

/// Folds already-captured probe values into the manifest lines. This is the
/// only place probe values become manifest lines, so the hashed bytes and
/// the `--explain` bytes share one composition by construction.
pub fn manifest_lines_from_probe_values(probe_values: &[ProbeValue]) -> String {
    let mut manifest_lines = String::new();
    for probe_entry in probe_values {
        // The `compatibility` line label is hashed bytes carried over from
        // the shell era; rewording it would change every identity and
        // invalidate every native build held in the `astro_native_build_cache`, so it stays despite the
        // toolchain vocabulary everywhere else.
        manifest_lines.push_str(&format!(
            "compatibility\t{}\t{}\n",
            probe_entry.name,
            digest_bytes(probe_entry.value.as_bytes())
        ));
    }
    manifest_lines
}

fn whole_output_probe_value(
    probe_name: &'static str,
    override_text: Option<&str>,
    program_name: &str,
    program_arguments: &[&str],
) -> Result<ProbeValue, NativeIdentityError> {
    let probe_started_at = Instant::now();
    let probe_output = match override_text {
        Some(override_value) => format!("{override_value}\n"),
        None => {
            let probe_command = Command::new(program_name)
                .args(program_arguments)
                .output()
                .map_err(|error| {
                    NativeIdentityError::Probe(format!(
                        "{probe_name} identity probe failed to start: {error}"
                    ))
                })?;
            if !probe_command.status.success() {
                return Err(NativeIdentityError::Probe(format!(
                    "{probe_name} identity probe exited with {}",
                    probe_command.status
                )));
            }
            String::from_utf8_lossy(&probe_command.stdout).into_owned()
        }
    };
    require_non_empty_probe(probe_name, &probe_output)?;
    eprintln!(
        "[astro-native-build-cache-fingerprint] probe={probe_name} elapsed_seconds={}",
        probe_started_at.elapsed().as_secs()
    );
    Ok(ProbeValue {
        name: probe_name,
        value: probe_output,
        from_override: override_text.is_some(),
    })
}

fn composite_sdk_probe_value(
    override_text: Option<&str>,
) -> Result<ProbeValue, NativeIdentityError> {
    let probe_started_at = Instant::now();
    let probe_output = match override_text {
        Some(override_value) => format!("{override_value}\n"),
        None => {
            let sdk_version = substituted_value(
                "xcrun",
                &["--sdk", "macosx", "--show-sdk-version"],
                "sdk version",
            )?;
            let sdk_build_version = substituted_value(
                "xcrun",
                &["--sdk", "macosx", "--show-sdk-build-version"],
                "sdk build",
            )?;
            format!("version={sdk_version}\nbuild={sdk_build_version}\n")
        }
    };
    require_non_empty_probe("sdk", &probe_output)?;
    eprintln!(
        "[astro-native-build-cache-fingerprint] probe=sdk elapsed_seconds={}",
        probe_started_at.elapsed().as_secs()
    );
    Ok(ProbeValue {
        name: "sdk",
        value: probe_output,
        from_override: override_text.is_some(),
    })
}

fn composite_clang_probe_value(
    override_text: Option<&str>,
) -> Result<ProbeValue, NativeIdentityError> {
    let probe_started_at = Instant::now();
    let probe_output = match override_text {
        Some(override_value) => format!("{override_value}\n"),
        None => {
            let clang_version =
                substituted_value("xcrun", &["clang", "-dumpversion"], "clang version")?;
            let clang_target =
                substituted_value("xcrun", &["clang", "-dumpmachine"], "clang target")?;
            format!("version={clang_version}\ntarget={clang_target}\n")
        }
    };
    require_non_empty_probe("clang", &probe_output)?;
    eprintln!(
        "[astro-native-build-cache-fingerprint] probe=clang elapsed_seconds={}",
        probe_started_at.elapsed().as_secs()
    );
    Ok(ProbeValue {
        name: "clang",
        value: probe_output,
        from_override: override_text.is_some(),
    })
}

/// CMake is the one identity tool the runner image provides unpinned (Xcode
/// comes from the pinned `DEVELOPER_DIR` and rustc from rust-toolchain.toml),
/// so its patch version rolls with every image rollout. A patch release does
/// not change codegen, but it did rotate the identity and force a cold native
/// rebuild per rollout. Only the major.minor line is hashed; overrides pass
/// through verbatim like every other probe.
fn composite_cmake_probe_value(
    override_text: Option<&str>,
) -> Result<ProbeValue, NativeIdentityError> {
    let probe_started_at = Instant::now();
    let probe_output = match override_text {
        Some(override_value) => format!("{override_value}\n"),
        None => {
            let cmake_version_output = substituted_value("cmake", &["--version"], "cmake version")?;
            format!("{}\n", coarsened_cmake_version_line(&cmake_version_output)?)
        }
    };
    require_non_empty_probe("cmake", &probe_output)?;
    eprintln!(
        "[astro-native-build-cache-fingerprint] probe=cmake elapsed_seconds={}",
        probe_started_at.elapsed().as_secs()
    );
    Ok(ProbeValue {
        name: "cmake",
        value: probe_output,
        from_override: override_text.is_some(),
    })
}

/// Reduces one `cmake --version` first line ("cmake version 4.4.3") to its
/// major.minor identity ("version=4.4"), dropping the patch version and the
/// vendor banner. Anything unparseable fails the probe rather than silently
/// hashing a value that may flap between runner images.
fn coarsened_cmake_version_line(
    whole_cmake_version_output: &str,
) -> Result<String, NativeIdentityError> {
    let first_line = whole_cmake_version_output
        .lines()
        .next()
        .unwrap_or_default()
        .trim();
    let version_text = first_line
        .strip_prefix("cmake version ")
        .ok_or_else(|| {
            NativeIdentityError::Probe(format!(
                "cmake identity probe printed an unrecognized version line: {first_line:?}"
            ))
        })?
        .trim();
    let mut version_segments = version_text.split('.');
    let major_segment = version_segments.next().unwrap_or_default();
    let minor_segment = version_segments.next().unwrap_or_default();
    if major_segment.is_empty() || minor_segment.is_empty() {
        return Err(NativeIdentityError::Probe(format!(
            "cmake identity probe printed a version without major.minor: {first_line:?}"
        )));
    }
    Ok(format!("version={major_segment}.{minor_segment}"))
}

/// Test-only exposure so unit tests can pin the coarsening contract
/// without invoking a real `cmake` binary.
#[cfg(test)]
pub(crate) fn coarsened_cmake_version_line_for_tests(
    whole_cmake_version_output: &str,
) -> Result<String, NativeIdentityError> {
    coarsened_cmake_version_line(whole_cmake_version_output)
}

/// Command substitution semantics: trailing newlines are stripped from the
/// captured value before the caller re-composes the probe text, matching
/// `$(...)` in the shell era exactly.
fn substituted_value(
    program_name: &str,
    program_arguments: &[&str],
    probe_description: &str,
) -> Result<String, NativeIdentityError> {
    let probe_command = Command::new(program_name)
        .args(program_arguments)
        .output()
        .map_err(|error| {
            NativeIdentityError::Probe(format!(
                "{probe_description} probe failed to start: {error}"
            ))
        })?;
    if !probe_command.status.success() {
        return Err(NativeIdentityError::Probe(format!(
            "{probe_description} probe exited with {}",
            probe_command.status
        )));
    }
    Ok(String::from_utf8_lossy(&probe_command.stdout)
        .trim_end_matches('\n')
        .to_owned())
}

fn require_non_empty_probe(
    probe_name: &str,
    probe_output: &str,
) -> Result<(), NativeIdentityError> {
    if probe_output.is_empty() {
        Err(NativeIdentityError::Probe(format!(
            "native toolchain input is empty: {probe_name}"
        )))
    } else {
        Ok(())
    }
}
