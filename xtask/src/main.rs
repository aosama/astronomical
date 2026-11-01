//! Repository tooling driven through `cargo xtask <command>`.
//!
//! Each subcommand is a typed port of the shell tool it replaces; the output
//! lines and exit codes are kept compatible so local habits, hooks, and CI
//! steps survive the move unchanged.

mod audit_test_declarations;
#[cfg(test)]
mod audit_test_declarations_tests;
mod bootstrap_native_dependencies;
#[cfg(test)]
mod bootstrap_native_dependencies_tests;
mod cache_prune;
#[cfg(test)]
mod cache_prune_tests;
mod check_architecture;
#[cfg(test)]
mod check_architecture_tests;
mod classify_scope;
#[cfg(test)]
mod classify_scope_tests;
mod commit_identity;
#[cfg(test)]
mod commit_identity_tests;
mod coverage_floor;
#[cfg(test)]
mod coverage_floor_tests;
mod disposable_target;
#[cfg(test)]
mod disposable_target_tests;
mod install_verification_tools;
#[cfg(test)]
mod install_verification_tools_tests;
mod native_dependency_manifest;
#[cfg(test)]
mod native_dependency_manifest_tests;
mod native_fingerprint;
mod notices;
#[cfg(test)]
mod notices_tests;
#[path = "disposable_target/signal_edge.rs"]
mod process_signals;
mod retired_native_scan;
#[cfg(test)]
mod retired_native_scan_tests;
#[cfg(test)]
mod sccache_save_tests;
mod source_size;
mod test_run;
mod tier_pyramid;

mod sccache_save;

pub(crate) use resolve_repository_root as workspace_root_from_manifest_directory;

use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut arguments: Vec<String> = std::env::args().skip(1).collect();
    let repository_root_override = take_repository_root_override(&mut arguments);
    let resolve_root = move || repository_root_override.unwrap_or_else(resolve_repository_root);
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
        Some("prune-caches") => run_command(cache_prune::run(arguments[1..].iter().cloned())),
        Some("save-sccache") => run_command(sccache_save::run(&arguments[1..])),
        Some("install-verification-tools") => run_command(install_verification_tools::run(
            arguments[1..].iter().cloned(),
        )),
        Some("bootstrap-native-dependencies") => run_command(bootstrap_native_dependencies::run(
            &resolve_root(),
            &arguments[1..],
        )),
        _ => {
            eprintln!(
                "usage: cargo xtask [--repository-root PATH] test-run --package NAME (--test TARGET | --lib | --tests) [--features FEATURES] [--release | --profile NAME] [--configuration PATH] [--filter NAME] [--exact] [--ignored] | mlx-c-coverage-inventory | disposable-target --lane NAME -- COMMAND [ARGUMENT ...] | scan-retired-native-output [--target-directory PATH] [--fail-on-found] [EXTRA_SCAN_ROOT ...] | tier-pyramid [--update-baseline] | source-size [--staged] [root] | notices [--check] | native-fingerprint [--source-only] [--explain] [--target TARGET_TRIPLE] --profile core|core+memory-contract [root] | commit-identity --staged|--pre-push | classify-scope | check-architecture | audit-test-declarations | prune-caches [--dry-run] [--repo owner/name] | save-sccache <primary-cache-key> | install-verification-tools [--prefix DIR] | bootstrap-native-dependencies [--cache-dir ABSOLUTE_PATH] [--verify]"
            );
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
