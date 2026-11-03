//! Command-line surface of `cargo xtask provision-bindgen-headers`, kept
//! flag-for-flag with the retired shell entry point so documented commands
//! and CI steps keep working.

use astronomical_native_identity::NativeBuildProfile;

pub(crate) const PROVISION_USAGE: &str = "usage: cargo xtask provision-bindgen-headers [--cache-dir ABSOLUTE_PATH] [--native-build-store-dir PATH] [--manifest FILE] [--patch-list FILE] [--profile core|core+memory-contract] [--target TARGET_TRIPLE] [--verify-headers | --print-patch-list]";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProvisionMode {
    /// Extract, patch, and publish the headers (or reuse a valid extraction).
    Provision,
    /// Compare the extracted MLX-C headers with a completed native build.
    VerifyHeaders,
    /// Print the parsed patch pipeline and exit.
    PrintPatchList,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProvisionOptions {
    pub(crate) cache_directory: Option<String>,
    pub(crate) native_build_store_directory: Option<String>,
    /// Tooling and tests substitute fixture manifests and patch lists so
    /// no contract depends on the real pins.
    pub(crate) explicit_manifest_path: Option<String>,
    pub(crate) explicit_patch_list_path: Option<String>,
    pub(crate) native_build_profile: NativeBuildProfile,
    pub(crate) target_triple: Option<String>,
    pub(crate) mode: ProvisionMode,
}

pub(crate) fn parse_provision_options(arguments: &[String]) -> Result<ProvisionOptions, String> {
    let mut provision_options = ProvisionOptions {
        cache_directory: None,
        native_build_store_directory: None,
        explicit_manifest_path: None,
        explicit_patch_list_path: None,
        native_build_profile: NativeBuildProfile::Core,
        target_triple: None,
        mode: ProvisionMode::Provision,
    };
    let mut argument_position = 0;
    while argument_position < arguments.len() {
        argument_position += apply_argument(&mut provision_options, arguments, argument_position)?;
    }
    Ok(provision_options)
}

/// Applies the flag at `argument_position` and returns how many arguments it
/// consumed.
fn apply_argument(
    provision_options: &mut ProvisionOptions,
    arguments: &[String],
    argument_position: usize,
) -> Result<usize, String> {
    let flag = arguments[argument_position].as_str();
    match flag {
        "--verify-headers" => select_mode(provision_options, ProvisionMode::VerifyHeaders)?,
        "--print-patch-list" => select_mode(provision_options, ProvisionMode::PrintPatchList)?,
        "--cache-dir"
        | "--native-build-store-dir"
        | "--manifest"
        | "--patch-list"
        | "--profile"
        | "--target" => {
            let Some(flag_value) = arguments.get(argument_position + 1) else {
                return Err(format!("{flag} requires a value\n{PROVISION_USAGE}"));
            };
            apply_valued_flag(provision_options, flag, flag_value)?;
            return Ok(2);
        }
        _ => return Err(format!("unrecognized argument: {flag}\n{PROVISION_USAGE}")),
    }
    Ok(1)
}

fn select_mode(
    provision_options: &mut ProvisionOptions,
    requested_mode: ProvisionMode,
) -> Result<(), String> {
    if provision_options.mode != ProvisionMode::Provision
        && provision_options.mode != requested_mode
    {
        return Err(format!(
            "--verify-headers and --print-patch-list are exclusive\n{PROVISION_USAGE}"
        ));
    }
    provision_options.mode = requested_mode;
    Ok(())
}

fn apply_valued_flag(
    provision_options: &mut ProvisionOptions,
    flag: &str,
    flag_value: &str,
) -> Result<(), String> {
    if flag_value.is_empty() {
        return Err(format!(
            "{flag} requires a non-empty value\n{PROVISION_USAGE}"
        ));
    }
    match flag {
        "--cache-dir" => provision_options.cache_directory = Some(flag_value.to_owned()),
        "--native-build-store-dir" => {
            provision_options.native_build_store_directory = Some(flag_value.to_owned())
        }
        "--manifest" => provision_options.explicit_manifest_path = Some(flag_value.to_owned()),
        "--patch-list" => provision_options.explicit_patch_list_path = Some(flag_value.to_owned()),
        "--target" => provision_options.target_triple = Some(flag_value.to_owned()),
        _ => match NativeBuildProfile::from_profile_text(flag_value) {
            Ok(native_build_profile) => {
                provision_options.native_build_profile = native_build_profile
            }
            Err(_) => return Err(format!("unsupported native build profile: {flag_value}")),
        },
    }
    Ok(())
}
