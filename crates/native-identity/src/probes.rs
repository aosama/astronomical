//! Full-mode compatibility probes: the hosted toolchain identity folded into
//! the native build fingerprint. Digest semantics are carried over from the
//! shell era byte-for-byte so cache keys stay meaningful: whole-output probes
//! hash raw stdout (trailing newline included), composite probes hash the
//! re-composed `version=…`/`build=…` text, and overrides hash the override
//! text plus one trailing newline.

use std::process::Command;
use std::time::Instant;

use crate::{FingerprintRequest, NativeIdentityError, digest_bytes};

/// Toolchain identity inputs, explicit per caller. `None` means "probe the
/// real toolchain". The environment-backed constructor preserves the
/// historical `ASTRONOMICAL_NATIVE_IDENTITY_*` variables for CI and fixture
/// workflows; the Cargo build scripts use `none()` so a production build
/// always fingerprints the toolchain CMake actually uses.
#[derive(Debug, Default, Clone)]
pub struct CompatibilityOverrides {
    pub xcode: Option<String>,
    pub sdk: Option<String>,
    pub clang: Option<String>,
    pub cmake: Option<String>,
    pub rustc: Option<String>,
    pub target_triple: Option<String>,
    pub build_type: Option<String>,
}

impl CompatibilityOverrides {
    pub fn none() -> Self {
        Self::default()
    }

    pub fn from_environment() -> Self {
        let environment_value = |variable_name: &str| {
            std::env::var(variable_name)
                .ok()
                .filter(|value| !value.is_empty())
        };
        Self {
            xcode: environment_value("ASTRONOMICAL_NATIVE_IDENTITY_XCODE"),
            sdk: environment_value("ASTRONOMICAL_NATIVE_IDENTITY_SDK"),
            clang: environment_value("ASTRONOMICAL_NATIVE_IDENTITY_CLANG"),
            cmake: environment_value("ASTRONOMICAL_NATIVE_IDENTITY_CMAKE"),
            rustc: environment_value("ASTRONOMICAL_NATIVE_IDENTITY_RUSTC"),
            target_triple: environment_value("ASTRONOMICAL_NATIVE_IDENTITY_TARGET")
                .or_else(|| environment_value("TARGET")),
            build_type: environment_value("ASTRONOMICAL_NATIVE_BUILD_TYPE"),
        }
    }
}

/// Assembles the compatibility manifest lines in the fixed order the shell
/// era established: target, build type, then the five toolchain probes.
pub fn compatibility_manifest_lines(
    request: &FingerprintRequest<'_>,
) -> Result<String, NativeIdentityError> {
    let target_triple = request
        .overrides
        .target_triple
        .clone()
        .ok_or(NativeIdentityError::MissingTargetTriple)?;
    let build_type = request
        .overrides
        .build_type
        .clone()
        .unwrap_or_else(|| "Release".to_owned());
    if build_type != "Release" {
        return Err(NativeIdentityError::UnsupportedBuildType(build_type));
    }

    let mut manifest_lines = format!("compatibility\ttarget\t{target_triple}\n");
    manifest_lines.push_str(&format!("compatibility\tbuild-type\t{build_type}\n"));

    let probe_names = ["xcode", "sdk", "clang", "cmake", "rustc"];
    for probe_name in probe_names {
        let probe_started_at = Instant::now();
        let probe_digest = match probe_name {
            "xcode" => override_or_whole_output(
                request.overrides.xcode.as_deref(),
                "xcodebuild",
                &["-version"],
                "xcode",
            )?,
            "sdk" => composite_sdk_identity(request.overrides.sdk.as_deref())?,
            "clang" => composite_clang_identity(request.overrides.clang.as_deref())?,
            "cmake" => override_or_whole_output(
                request.overrides.cmake.as_deref(),
                "cmake",
                &["--version"],
                "cmake",
            )?,
            _ => override_or_whole_output(
                request.overrides.rustc.as_deref(),
                "rustc",
                &["-vV"],
                "rustc",
            )?,
        };
        manifest_lines.push_str(&format!("compatibility\t{probe_name}\t{probe_digest}\n"));
        eprintln!(
            "[native-build-cache-fingerprint] probe={probe_name} elapsed_seconds={}",
            probe_started_at.elapsed().as_secs()
        );
    }
    Ok(manifest_lines)
}

fn override_or_whole_output(
    override_text: Option<&str>,
    program_name: &str,
    program_arguments: &[&str],
    probe_name: &str,
) -> Result<String, NativeIdentityError> {
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
    Ok(digest_bytes(probe_output.as_bytes()))
}

fn composite_sdk_identity(override_text: Option<&str>) -> Result<String, NativeIdentityError> {
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
    Ok(digest_bytes(probe_output.as_bytes()))
}

fn composite_clang_identity(override_text: Option<&str>) -> Result<String, NativeIdentityError> {
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
    Ok(digest_bytes(probe_output.as_bytes()))
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
            "native compatibility input is empty: {probe_name}"
        )))
    } else {
        Ok(())
    }
}
