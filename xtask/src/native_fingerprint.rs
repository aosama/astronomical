//! `cargo xtask native-fingerprint` — CLI face of the native build identity
//! library for every caller that runs outside a Cargo build script (CI
//! steps, provisioning pipelines, packaging). The Cargo build scripts call
//! the library in-process instead (issue #1250).

use std::path::Path;

use astronomical_native_identity::probes::CompatibilityOverrides;
use astronomical_native_identity::{
    FingerprintRequest, IdentityScope, NativeBuildProfile, native_build_identity,
};

pub fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    let mut source_only = false;
    let mut profile_text: Option<String> = None;
    let mut positional_root: Option<String> = None;
    let mut argument_index = 0;
    while argument_index < arguments.len() {
        let argument = &arguments[argument_index];
        match argument.as_str() {
            "--source-only" => source_only = true,
            "--profile" => {
                let profile_value = arguments.get(argument_index + 1).ok_or(
                    "--profile requires a value: expected core or core+memory-contract".to_owned(),
                )?;
                profile_text = Some(profile_value.clone());
                argument_index += 1;
            }
            "--help" | "-h" => {
                print_usage();
                return Ok(());
            }
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
    let effective_root = positional_root
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| repository_root.to_path_buf());
    let identity = native_build_identity(&FingerprintRequest {
        repository_root: &effective_root,
        profile,
        scope,
        // The historical environment contract stays: fixture overrides and
        // CI-provided target triples flow through the same variables the
        // shell era defined.
        overrides: CompatibilityOverrides::from_environment(),
    })
    .map_err(|error| error.to_string())?;
    println!("{identity}");
    Ok(())
}

fn print_usage() {
    eprintln!(
        "usage: cargo xtask native-fingerprint [--source-only] --profile core|core+memory-contract [repository-root]"
    );
}
