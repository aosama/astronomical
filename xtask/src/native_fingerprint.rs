//! `cargo xtask native-fingerprint` — CLI face of the native build identity
//! library for every caller that runs outside a Cargo build script (CI
//! steps, provisioning pipelines, packaging). The Cargo build scripts call
//! the library in-process instead (issue #1250).

use std::path::{Path, PathBuf};
use std::process::Command;

use astronomical_native_identity::probes::{ProbeValue, ToolchainOverrides};
use astronomical_native_identity::{
    FingerprintRequest, IdentityScope, NativeBuildProfile, native_build_identity_with_probe_values,
};

#[cfg(test)]
#[path = "native_fingerprint_tests.rs"]
mod native_fingerprint_tests;

pub fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    let Some(options) = parse_arguments(repository_root, arguments)? else {
        print_usage();
        return Ok(());
    };
    let cargo_target = std::env::var("TARGET").ok();
    let request = options.request(cargo_target.as_deref(), probe_rustc_host)?;
    let (identity, probe_values) =
        native_build_identity_with_probe_values(&request).map_err(|error| error.to_string())?;
    if options.explain {
        // These are the exact values the identity was hashed from — one
        // capture, no second pass. Values are flattened to one line each so
        // a CI log stays greppable: a native-cache flap is then attributable
        // by reading seven lines instead of diffing runner images.
        eprintln!("[native-fingerprint] identity={identity}");
        for probe_entry in &probe_values {
            eprintln!(
                "[native-fingerprint] probe={} source={} value={}",
                probe_entry.name,
                probe_source_text(probe_entry),
                one_line_text(&probe_entry.value)
            );
        }
    }
    println!("{identity}");
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
struct FingerprintOptions {
    repository_root: PathBuf,
    profile: NativeBuildProfile,
    scope: IdentityScope,
    explain: bool,
    target_triple: Option<String>,
}

impl FingerprintOptions {
    fn request(
        &self,
        cargo_target: Option<&str>,
        rustc_host: impl FnOnce() -> Result<String, String>,
    ) -> Result<FingerprintRequest<'_>, String> {
        let target_triple = if self.scope == IdentityScope::Full {
            Some(resolve_target_triple(
                self.target_triple.as_deref(),
                cargo_target,
                rustc_host,
            )?)
        } else {
            None
        };
        Ok(FingerprintRequest {
            repository_root: &self.repository_root,
            profile: self.profile,
            scope: self.scope,
            overrides: ToolchainOverrides {
                target_triple,
                ..ToolchainOverrides::none()
            },
        })
    }
}

fn parse_arguments(
    repository_root: &Path,
    arguments: &[String],
) -> Result<Option<FingerprintOptions>, String> {
    let mut source_only = false;
    let mut explain = false;
    let mut profile_text: Option<String> = None;
    let mut positional_root: Option<String> = None;
    let mut target_triple: Option<String> = None;
    let mut argument_index = 0;
    while argument_index < arguments.len() {
        let argument = &arguments[argument_index];
        match argument.as_str() {
            "--source-only" => source_only = true,
            "--explain" => explain = true,
            "--profile" => {
                let profile_value = option_value(arguments, argument_index, "--profile")?;
                profile_text = Some(profile_value.clone());
                argument_index += 1;
            }
            "--target" => {
                let target_value = option_value(arguments, argument_index, "--target")?;
                if target_triple.replace(target_value.clone()).is_some() {
                    return Err("only one --target may be given".to_owned());
                }
                argument_index += 1;
            }
            "--help" | "-h" => return Ok(None),
            other => {
                if other.starts_with("--") {
                    return Err(format!("unsupported native fingerprint option: {other}"));
                }
                if positional_root.replace(other.to_owned()).is_some() {
                    return Err("only one repository root may be given".to_owned());
                }
            }
        }
        argument_index += 1;
    }
    let profile_text = profile_text.ok_or("--profile is required")?;
    let profile =
        NativeBuildProfile::from_profile_text(&profile_text).map_err(|error| error.to_string())?;
    let scope = if source_only {
        IdentityScope::SourceOnly
    } else {
        IdentityScope::Full
    };
    Ok(Some(FingerprintOptions {
        repository_root: positional_root
            .map(PathBuf::from)
            .unwrap_or_else(|| repository_root.to_path_buf()),
        profile,
        scope,
        explain,
        target_triple,
    }))
}

fn option_value<'a>(
    arguments: &'a [String],
    argument_index: usize,
    option_name: &str,
) -> Result<&'a String, String> {
    arguments
        .get(argument_index + 1)
        .filter(|option_value| !option_value.trim().is_empty() && !option_value.starts_with('-'))
        .ok_or_else(|| format!("{option_name} requires a value"))
}

fn resolve_target_triple(
    explicit_target: Option<&str>,
    cargo_target: Option<&str>,
    rustc_host: impl FnOnce() -> Result<String, String>,
) -> Result<String, String> {
    match explicit_target.or(cargo_target.filter(|target_triple| !target_triple.is_empty())) {
        Some(target_triple) if target_triple.trim().is_empty() => {
            Err("the native build target triple must not be blank".to_owned())
        }
        Some(target_triple) => Ok(target_triple.to_owned()),
        None => rustc_host(),
    }
}

fn probe_rustc_host() -> Result<String, String> {
    let rustc_output = Command::new("rustc")
        .arg("-vV")
        .output()
        .map_err(|error| format!("rustc host probe failed to start: {error}"))?;
    if !rustc_output.status.success() {
        return Err(format!(
            "rustc host probe exited with {}",
            rustc_output.status
        ));
    }
    host_from_rustc_version(&String::from_utf8_lossy(&rustc_output.stdout))
}

fn host_from_rustc_version(rustc_version_output: &str) -> Result<String, String> {
    rustc_version_output
        .lines()
        .find_map(|version_line| version_line.strip_prefix("host: "))
        .map(str::trim)
        .filter(|host_triple| !host_triple.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| "rustc host probe did not report a target triple".to_owned())
}

fn probe_source_text(probe_entry: &ProbeValue) -> &'static str {
    if probe_entry.from_override {
        "override"
    } else {
        "live"
    }
}

fn one_line_text(multi_line_value: &str) -> String {
    multi_line_value
        .split('\n')
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" | ")
}

fn print_usage() {
    eprintln!(
        "usage: cargo xtask native-fingerprint [--source-only] [--explain] [--target TARGET_TRIPLE] --profile core|core+memory-contract [repository-root]"
    );
}
