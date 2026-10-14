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

/// Captures the compatibility inputs in the fixed order the shell era
/// established: target, build type, then the five toolchain probes. This is
/// the single capture path — the identity manifest digests exactly these
/// values, so an explained value can never diverge from the hashed one.
pub fn compatibility_probe_values(
    request: &FingerprintRequest<'_>,
) -> Result<Vec<ProbeValue>, NativeIdentityError> {
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
    probe_values.push(whole_output_probe_value(
        "xcode",
        request.overrides.xcode.as_deref(),
        "xcodebuild",
        &["-version"],
    )?);
    probe_values.push(composite_sdk_probe_value(request.overrides.sdk.as_deref())?);
    probe_values.push(composite_clang_probe_value(
        request.overrides.clang.as_deref(),
    )?);
    probe_values.push(whole_output_probe_value(
        "cmake",
        request.overrides.cmake.as_deref(),
        "cmake",
        &["--version"],
    )?);
    probe_values.push(whole_output_probe_value(
        "rustc",
        request.overrides.rustc.as_deref(),
        "rustc",
        &["-vV"],
    )?);
    Ok(probe_values)
}

/// Assembles the compatibility manifest lines in the fixed order the shell
/// era established: target, build type, then the five toolchain probes.
pub fn compatibility_manifest_lines(
    request: &FingerprintRequest<'_>,
) -> Result<String, NativeIdentityError> {
    let probe_values = compatibility_probe_values(request)?;
    Ok(manifest_lines_from_probe_values(&probe_values))
}

/// Folds already-captured probe values into the manifest lines. This is the
/// only place probe values become manifest lines, so the hashed bytes and
/// the `--explain` bytes share one composition by construction.
pub fn manifest_lines_from_probe_values(probe_values: &[ProbeValue]) -> String {
    let mut manifest_lines = String::new();
    for probe_entry in probe_values {
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
        "[native-build-cache-fingerprint] probe={probe_name} elapsed_seconds={}",
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
        "[native-build-cache-fingerprint] probe=sdk elapsed_seconds={}",
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
        "[native-build-cache-fingerprint] probe=clang elapsed_seconds={}",
        probe_started_at.elapsed().as_secs()
    );
    Ok(ProbeValue {
        name: "clang",
        value: probe_output,
        from_override: override_text.is_some(),
    })
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
