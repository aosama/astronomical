//! Repository tooling driven through `cargo xtask <command>`.
//!
//! Each subcommand is a typed port of the shell tool it replaces; the output
//! lines and exit codes are kept compatible so local habits, hooks, and CI
//! steps survive the move unchanged.

mod classify_scope;
#[cfg(test)]
mod classify_scope_tests;
mod native_fingerprint;
mod notices;
mod source_size;
mod tier_pyramid;

use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut arguments: Vec<String> = std::env::args().skip(1).collect();
    let mut repository_root = resolve_repository_root();
    if let Some(root_position) = arguments
        .iter()
        .position(|argument| argument == "--repository-root")
    {
        if let Some(root_path) = arguments.get(root_position + 1) {
            repository_root = std::path::PathBuf::from(root_path);
            arguments.drain(root_position..=root_position + 1);
        }
    }
    match arguments.first().map(String::as_str) {
        Some("tier-pyramid") => run_command(tier_pyramid::run(
            &repository_root,
            arguments.contains(&"--update-baseline".to_owned()),
        )),
        Some("source-size") => run_command(source_size::run(&repository_root, &arguments[1..])),
        Some("notices") => run_command(notices::run(
            &repository_root,
            arguments.contains(&"--check".to_owned()),
        )),
        Some("native-fingerprint") => {
            run_command(native_fingerprint::run(&repository_root, &arguments[1..]))
        }
        Some("classify-scope") => run_command(classify_scope::run(&repository_root)),
        _ => {
            eprintln!(
                "usage: cargo xtask [--repository-root PATH] tier-pyramid [--update-baseline] | source-size [--staged] [root] | notices [--check] | native-fingerprint [--source-only] --profile core|core+memory-contract [root] | classify-scope"
            );
            ExitCode::from(2)
        }
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
