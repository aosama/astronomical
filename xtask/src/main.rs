//! Repository tooling driven through `cargo xtask <command>`.
//!
//! Each subcommand is a typed port of the shell tool it replaces; the output
//! lines and exit codes are kept compatible so local habits, hooks, and CI
//! steps survive the move unchanged.

mod acceptance_feature_lint;
#[cfg(test)]
mod acceptance_feature_lint_tests;
mod app_bundle;
mod audit_test_declarations;
#[cfg(test)]
mod audit_test_declarations_tests;
mod bindgen_headers;
mod bootstrap_native_dependencies;
#[cfg(test)]
mod bootstrap_native_dependencies_tests;
mod check_architecture;
mod check_architecture_family_layer;
#[cfg(test)]
mod check_architecture_family_layer_tests;
mod check_architecture_recognition_crates;
#[cfg(test)]
mod check_architecture_recognition_crates_tests;
#[cfg(test)]
mod check_architecture_test_support;
#[cfg(test)]
mod check_architecture_tests;
mod check_duplication;
#[cfg(test)]
mod check_duplication_tests;
mod classify_scope;
#[cfg(test)]
mod classify_scope_tests;
mod commit_identity;
#[cfg(test)]
mod commit_identity_tests;
mod convert_kev;
mod convert_laya;
mod coverage_floor;
#[cfg(test)]
mod coverage_floor_tests;
mod crap_baseline;
#[cfg(test)]
mod crap_baseline_tests;
mod disposable_target;
#[cfg(test)]
mod disposable_target_tests;
mod function_length_budget;
#[cfg(test)]
mod function_length_budget_tests;
mod github_actions_cache_prune;
#[cfg(test)]
mod github_actions_cache_prune_tests;
mod host_platforms;
#[cfg(test)]
mod host_platforms_tests;
mod install_verification_tools;
#[cfg(test)]
mod install_verification_tools_tests;
mod measure_speed;
mod menu_contracts;
#[cfg(test)]
mod menu_contracts_tests;
mod model_files;
mod model_loading_changed;
#[cfg(test)]
mod model_loading_changed_tests;
mod model_serving_default_features;
#[cfg(test)]
mod model_serving_default_features_tests;
mod native_dependency_manifest;
#[cfg(test)]
mod native_dependency_manifest_tests;
mod native_fingerprint;
mod native_store_paths;
#[cfg(test)]
mod native_store_paths_tests;
mod notices;
#[cfg(test)]
mod notices_tests;
mod opencode_plugin_release;
#[cfg(test)]
mod opencode_plugin_release_tests;
mod prewarm_native_build;
#[cfg(test)]
mod prewarm_native_build_tests;
#[path = "disposable_target/signal_edge.rs"]
mod process_signals;
mod release;
mod release_speed;
mod replay_expert_misses;
mod retired_native_scan;
#[cfg(test)]
mod retired_native_scan_tests;
mod rust_test_selection;
#[cfg(test)]
mod sccache_save_tests;
mod site_layout;
#[cfg(test)]
mod site_layout_tests;
mod site_link_check;
#[cfg(test)]
mod site_link_check_tests;
mod site_pages;
mod skill_site;
#[cfg(test)]
mod skill_site_tests;
mod small_mac_matrix;
#[cfg(test)]
mod small_mac_matrix_tests;
mod source_size;
mod step_log;
mod test_channel_isolation;
#[cfg(test)]
mod test_channel_isolation_tests;
mod test_lanes;
mod test_run;
mod thin_talk_assets;
#[cfg(test)]
mod thin_talk_assets_tests;
mod tier_pyramid;
#[cfg(test)]
mod tier_pyramid_tests;
mod tool_pins;
#[cfg(test)]
mod tool_pins_tests;
mod usage_report;
#[cfg(test)]
mod usage_report_tests;
mod verified_tree;
mod version_only_bump;
#[cfg(test)]
mod version_only_bump_tests;

mod sccache_save;

pub(crate) use resolve_repository_root as workspace_root_from_manifest_directory;

use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut arguments: Vec<String> = std::env::args().skip(1).collect();
    let repository_root_override = take_repository_root_override(&mut arguments);
    let resolve_root = move || repository_root_override.unwrap_or_else(resolve_repository_root);
    if let Some(command_name) = arguments.first()
        && APP_AND_RELEASE_COMMANDS.contains(&command_name.as_str())
    {
        return run_command(run_app_and_release_command(
            command_name,
            &resolve_root(),
            &arguments[1..],
        ));
    }
    match arguments.first().map(String::as_str) {
        Some("test-run") => test_run::run(&resolve_root(), &arguments[1..]),
        Some("mlx-c-coverage-inventory") => {
            test_run::refresh_coverage_inventory(&resolve_root(), &arguments[1..])
        }
        Some("disposable-target") => disposable_target::run(&resolve_root(), &arguments[1..]),
        Some("scan-retired-native-output") => {
            retired_native_scan::run(&resolve_root(), &arguments[1..])
        }
        Some("tier-pyramid") => run_command(tier_pyramid::run(
            &resolve_root(),
            arguments.contains(&"--update-baseline".to_owned()),
        )),
        Some("source-size") => run_command(source_size::run(&resolve_root(), &arguments[1..])),
        Some("coverage-floor") => {
            run_command(coverage_floor::run(&resolve_root(), &arguments[1..]))
        }
        Some("crap-baseline") => run_command(crap_baseline::run(&resolve_root(), &arguments[1..])),
        Some("notices") => run_command(notices::run(
            &resolve_root(),
            arguments.contains(&"--check".to_owned()),
        )),
        Some("native-fingerprint") => {
            run_command(native_fingerprint::run(&resolve_root(), &arguments[1..]))
        }
        Some("commit-identity") => run_command(commit_identity::run(&arguments[1..])),
        Some("classify-scope") => run_command(classify_scope::run(&resolve_root())),
        Some("check-architecture") => run_command(check_architecture::run(&resolve_root())),
        Some("audit-test-declarations") => {
            run_command(audit_test_declarations::run(&resolve_root()))
        }
        Some("prune-caches") => run_command(github_actions_cache_prune::run(
            arguments[1..].iter().cloned(),
        )),
        Some("save-sccache") => run_command(sccache_save::run(&arguments[1..])),
        Some("install-verification-tools") => run_command(install_verification_tools::run(
            arguments[1..].iter().cloned(),
        )),
        Some("bootstrap-native-dependencies") => run_command(bootstrap_native_dependencies::run(
            &resolve_root(),
            &arguments[1..],
        )),
        _ => run_model_conversion_command(&arguments),
    }
}

const USAGE: &str = "usage: cargo xtask [--repository-root PATH] app dev | app instance status|start|stop|restart | app package --channel stable|app-store [--signing-identity NAME] | app smoke-test --app-bundle PATH [--real-model MODEL_ID] | app info-plist --channel development|stable|app-store | prewarm-native-build --profile core|core+memory-contract [--profile ...] | release <step> (see scripts/release/README.md) | provision-bindgen-headers [--verify-headers | --print-patch-list] [--cache-dir PATH] [--native-build-store-dir PATH] [--manifest FILE] [--patch-list FILE] [--profile NAME] [--target TRIPLE] | test-run --package NAME (--test TARGET | --lib | --tests) [--features FEATURES] [--release | --profile NAME] [--configuration PATH] [--filter NAME] [--exact] [--ignored] | mlx-c-coverage-inventory | disposable-target --lane NAME -- COMMAND [ARGUMENT ...] | scan-retired-native-output [--target-directory PATH] [--fail-on-found] [EXTRA_SCAN_ROOT ...] | tier-pyramid [--update-baseline] | source-size [--staged] [root] | notices [--check] | native-fingerprint [--source-only] [--explain] [--cache-environment] [--target TARGET_TRIPLE] --profile core|core+memory-contract [root] | commit-identity --staged|--pre-push | classify-scope | check-architecture | check-duplication [--write] [--against REF] [--top N] | model-loading-changed PREVIOUS_TAG | check-test-channel-isolation | menu-contracts | lint-acceptance-features | function-length-budget | model-serving-default-features | vendor-thin-talk-assets [--verify-only] | audit-test-declarations | prune-caches [--dry-run] [--repo owner/name] | save-sccache <primary-cache-key> | install-verification-tools [--prefix DIR] | bootstrap-native-dependencies [--cache-dir ABSOLUTE_PATH] [--verify] | convert-laya --checkpoint DIRECTORY --output DIRECTORY | convert-kev --adapter DIRECTORY --base DIRECTORY --output DIRECTORY | replay-expert-misses EVIDENCE_DIRECTORY | opencode-plugin-release-check [--require-on-main] TAG | site-layout [--check] | check-site-links | skill-site [--check] | usage-report [--repo owner/name] [--catalog PATH] | small-mac-matrix [--base-url URL] [--model ID]... [--mac-gb N]... | measure-speed [--base-url URL] [--rounds N] [--temperature T] [--measured-on TEXT] [--reference-set] [--write-ledger] [--write-catalog] [MODEL_ID...] | release-speed-table --previous VERSION|FILE --current VERSION|FILE | selected-cargo --selection all|none|PACKAGES ALIAS [ARGUMENT ...] | lane-cargo ALIAS [ARGUMENT ...]";

/// The commands that convert a model checkpoint into the layout Astronomical
/// serves, dispatched apart from `main` so its match stays within the
/// complexity budget. Anything else is not a command, and gets the usage.
fn run_model_conversion_command(arguments: &[String]) -> ExitCode {
    match arguments.first().map(String::as_str) {
        Some("convert-laya") => run_command(convert_laya::run(&arguments[1..])),
        Some("convert-kev") => run_command(convert_kev::run(&arguments[1..])),
        _ => run_code_shape_command(arguments),
    }
}

/// The ratchets over the shape of the code itself, dispatched apart from the
/// commands above so neither match grows. Like the conversion commands before
/// it, it resolves the repository root from the checkout: the hooks run it
/// from inside the checkout they guard.
fn run_code_shape_command(arguments: &[String]) -> ExitCode {
    match arguments.first().map(String::as_str) {
        Some("check-duplication") => run_command(check_duplication::run(
            &resolve_repository_root(),
            &arguments[1..],
        )),
        _ => run_measurement_command(arguments),
    }
}

/// The measurement reports that read evidence a probe saved, dispatched apart
/// from the commands above so neither match grows. Anything else is not a
/// command, and gets the usage.
fn run_measurement_command(arguments: &[String]) -> ExitCode {
    match arguments.first().map(String::as_str) {
        Some("replay-expert-misses") => run_command(replay_expert_misses::run(&arguments[1..])),
        _ => run_app_release_command(arguments),
    }
}

/// The checks that gate releasing the app on the direct download, dispatched
/// apart from the commands above so neither match grows. Like the commands
/// after it, it resolves the repository root from the checkout: the release
/// skill runs it from the checkout root.
fn run_app_release_command(arguments: &[String]) -> ExitCode {
    match arguments.first().map(String::as_str) {
        Some("model-loading-changed") => run_command(model_loading_changed::run(
            &resolve_repository_root(),
            &arguments[1..],
        )),
        _ => run_integration_release_command(arguments),
    }
}

/// The checks that gate releasing an integration published apart from the
/// app, dispatched apart from the commands above so neither match grows.
/// Anything else is not a command, and gets the usage. Like the conversion and
/// measurement commands before it, it resolves the repository root from the
/// checkout and ignores `--repository-root`: the publish workflow runs it from
/// the checkout root, and threading the override through every dispatcher
/// would widen `main`'s match for no caller.
fn run_integration_release_command(arguments: &[String]) -> ExitCode {
    match arguments.first().map(String::as_str) {
        Some("opencode-plugin-release-check") => run_command(opencode_plugin_release::run(
            &resolve_repository_root(),
            &arguments[1..],
        )),
        _ => run_project_report_command(arguments),
    }
}

/// Reports about the project rather than checks of it, dispatched apart from
/// the commands above so neither match grows. Anything else is not a
/// command, and gets the usage.
fn run_project_report_command(arguments: &[String]) -> ExitCode {
    match arguments.first().map(String::as_str) {
        Some("usage-report") => run_command(usage_report::run(
            &resolve_repository_root(),
            &arguments[1..],
        )),
        _ => run_ci_selection_command(arguments),
    }
}

/// The macOS build's narrowed cargo runs, dispatched apart from the commands
/// above so neither match grows.
fn run_ci_selection_command(arguments: &[String]) -> ExitCode {
    match arguments.first().map(String::as_str) {
        Some("selected-cargo") => run_command(rust_test_selection::run_selected_cargo(
            &resolve_repository_root(),
            &arguments[1..],
        )),
        _ => run_test_lane_command(arguments),
    }
}

/// The test lanes built from each package's declared lanes, dispatched apart
/// from the commands above so neither match grows.
fn run_test_lane_command(arguments: &[String]) -> ExitCode {
    match arguments.first().map(String::as_str) {
        Some("lane-cargo") => {
            run_command(test_lanes::run(&resolve_repository_root(), &arguments[1..]))
        }
        _ => run_pre_release_probe_command(arguments),
    }
}

/// Probes run by hand before a release, dispatched apart from the commands
/// above so neither match grows; anything else goes on to the release notes'
/// commands.
fn run_pre_release_probe_command(arguments: &[String]) -> ExitCode {
    match arguments.first().map(String::as_str) {
        Some("small-mac-matrix") => run_command(small_mac_matrix::run(
            &resolve_repository_root(),
            &arguments[1..],
        )),
        Some("measure-speed") => run_command(measure_speed::run(
            &resolve_repository_root(),
            &arguments[1..],
        )),
        _ => run_release_notes_command(arguments),
    }
}

/// Commands that write part of a release's notes, dispatched apart from the
/// probes so neither match grows. Anything else is not a command, and gets
/// the usage.
fn run_release_notes_command(arguments: &[String]) -> ExitCode {
    match arguments.first().map(String::as_str) {
        Some("release-speed-table") => run_command(release_speed::run(
            &resolve_repository_root(),
            &arguments[1..],
        )),
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

/// The repository root a caller pinned, removing `--repository-root PATH`
/// from `arguments` so subcommands never see it.
///
/// The root is honored before any working-directory walk: subcommands invoked
/// from outside a workspace checkout (the disposable-target journeys run from
/// temporary sandboxes) must not depend on walking up from the working
/// directory. The explicit flag wins over the journey's exported root.
fn take_repository_root_override(arguments: &mut Vec<String>) -> Option<std::path::PathBuf> {
    let mut root_flag_position = None;
    for (argument_position, argument) in arguments.iter().enumerate() {
        if argument == "--repository-root" {
            root_flag_position = Some(argument_position);
            break;
        }
    }
    if let Some(root_position) = root_flag_position
        && let Some(root_path) = arguments.get(root_position + 1)
    {
        let repository_root = std::path::PathBuf::from(root_path);
        arguments.drain(root_position..=root_position + 1);
        return Some(repository_root);
    }
    match std::env::var(disposable_target::DISPOSABLE_REPOSITORY_ROOT_ENV) {
        Ok(journey_root) if !journey_root.is_empty() => {
            Some(std::path::PathBuf::from(journey_root))
        }
        _ => None,
    }
}

/// The app, release, native-provisioning, and repository-check commands
/// that replaced shell scripts, dispatched apart from `main` so its match
/// stays within the complexity budget.
const APP_AND_RELEASE_COMMANDS: [&str; 13] = [
    "app",
    "release",
    "provision-bindgen-headers",
    "prewarm-native-build",
    "menu-contracts",
    "lint-acceptance-features",
    "function-length-budget",
    "model-serving-default-features",
    "check-test-channel-isolation",
    "vendor-thin-talk-assets",
    "site-layout",
    "check-site-links",
    "skill-site",
];

fn run_app_and_release_command(
    command_name: &str,
    repository_root: &Path,
    command_arguments: &[String],
) -> Result<(), String> {
    match command_name {
        "app" => app_bundle::run(repository_root, command_arguments),
        "release" => release::run(repository_root, command_arguments),
        "provision-bindgen-headers" => bindgen_headers::run(repository_root, command_arguments),
        "prewarm-native-build" => prewarm_native_build::run(repository_root, command_arguments),
        "menu-contracts" => menu_contracts::run(repository_root, command_arguments),
        "vendor-thin-talk-assets" => thin_talk_assets::run(repository_root, command_arguments),
        _ => run_repository_check_command(command_name, repository_root, command_arguments),
    }
}

/// The repository checks among `APP_AND_RELEASE_COMMANDS`, split out so that
/// adding a check does not grow the dispatcher above past its complexity
/// baseline.
fn run_repository_check_command(
    command_name: &str,
    repository_root: &Path,
    command_arguments: &[String],
) -> Result<(), String> {
    match command_name {
        "lint-acceptance-features" => {
            acceptance_feature_lint::run_where_buildable(repository_root, command_arguments)
        }
        "site-layout" => site_layout::run(repository_root, command_arguments),
        "check-site-links" => site_link_check::run(repository_root, command_arguments),
        _ => run_lint_or_site_check(command_name, repository_root, command_arguments),
    }
}

/// The host-aware lints, or else the skill-site and isolation checks,
/// split out so the dispatcher above stays at its complexity baseline.
fn run_lint_or_site_check(
    command_name: &str,
    repository_root: &Path,
    command_arguments: &[String],
) -> Result<(), String> {
    match command_name {
        "function-length-budget" => function_length_budget::run(repository_root, command_arguments),
        "model-serving-default-features" => {
            model_serving_default_features::run(repository_root, command_arguments)
        }
        _ => run_skill_site_or_isolation_check(command_name, repository_root, command_arguments),
    }
}

/// `skill-site`, or else the test-channel isolation check, split out so the
/// dispatcher above stays at its complexity baseline.
fn run_skill_site_or_isolation_check(
    command_name: &str,
    repository_root: &Path,
    command_arguments: &[String],
) -> Result<(), String> {
    if command_name == "skill-site" {
        return skill_site::run(repository_root, command_arguments);
    }
    test_channel_isolation::run(repository_root, command_arguments)
}

fn run_command(result: Result<(), String>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

fn resolve_repository_root() -> std::path::PathBuf {
    // cargo run sets CARGO_MANIFEST_DIR; a directly executed binary (the
    // contract tests invoke target/debug/xtask under a narrowed PATH) falls
    // back to walking up from the working directory to the workspace root.
    if let Ok(manifest_directory) = std::env::var("CARGO_MANIFEST_DIR") {
        return std::path::PathBuf::from(manifest_directory)
            .join("..")
            .canonicalize()
            .expect("the repository root should resolve above the xtask package");
    }
    let start_directory =
        std::env::current_dir().expect("the working directory should be readable");
    let mut candidate_directory: Option<std::path::PathBuf> = Some(start_directory);
    while let Some(candidate) = candidate_directory {
        let is_workspace_root = std::fs::read_to_string(candidate.join("Cargo.toml"))
            .map(|manifest_text| {
                manifest_text
                    .lines()
                    .any(|line| line.trim() == "[workspace]")
            })
            .unwrap_or(false);
        if is_workspace_root {
            return candidate;
        }
        candidate_directory = candidate.parent().map(Path::to_path_buf);
    }
    panic!("no workspace root found above the working directory");
}
