//! Locates workspace-built binaries for whole-product journeys.
//!
//! `CARGO_BIN_EXE_<name>` is only set for binaries of the package running the
//! test, so a root-level E2E suite that drives binaries from other packages
//! resolves them through the shared target directory instead.

use std::path::{Path, PathBuf};

/// Returns the executable path of a binary built into the workspace's shared
/// target directory.
///
/// Panics with the exact build command when the binary is missing, so a
/// skipped build fails loudly at the point of use instead of surfacing as a
/// confusing spawn error deep inside a journey. A relative `CARGO_TARGET_DIR`
/// resolves against the test's working directory, which cargo sets to the
/// test package's root.
pub fn workspace_binary(binary_name: &str) -> PathBuf {
    let test_working_directory =
        std::env::current_dir().expect("the test working directory should be readable");
    let workspace_root = find_workspace_root(&test_working_directory).unwrap_or_else(|| {
        panic!(
            "no workspace root found above {}",
            test_working_directory.display()
        )
    });
    let target_root = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root.join("target"));
    resolve_workspace_binary(&target_root, binary_name)
}

fn find_workspace_root(start_directory: &Path) -> Option<PathBuf> {
    let mut candidate_directory = Some(start_directory);
    while let Some(candidate) = candidate_directory {
        let manifest_path = candidate.join("Cargo.toml");
        let is_workspace_root = std::fs::read_to_string(&manifest_path)
            .map(|manifest_text| {
                manifest_text
                    .lines()
                    .any(|line| line.trim() == "[workspace]")
            })
            .unwrap_or(false);
        if is_workspace_root {
            return Some(candidate.to_path_buf());
        }
        candidate_directory = candidate.parent();
    }
    None
}

fn resolve_workspace_binary(target_root: &Path, binary_name: &str) -> PathBuf {
    let binary_file_name = format!("{binary_name}{}", std::env::consts::EXE_SUFFIX);
    for profile_directory in ["debug", "release"] {
        let candidate = target_root.join(profile_directory).join(&binary_file_name);
        if candidate.is_file() {
            return candidate;
        }
    }
    panic!(
        "workspace binary '{binary_name}' was not found under {}; \
         build it first with: cargo build --bins -p astronomical-daemon",
        target_root.display()
    );
}

#[cfg(test)]
mod tests {
    use super::resolve_workspace_binary;

    #[test]
    fn resolves_a_built_debug_binary() {
        let target_root = tempfile::tempdir().expect("temporary target root should be created");
        let binary_path = target_root.path().join("debug").join("fixture-bin");
        std::fs::create_dir_all(binary_path.parent().expect("debug directory has a parent"))
            .expect("debug directory should be created");
        std::fs::write(&binary_path, b"").expect("fixture binary should be written");

        assert_eq!(
            resolve_workspace_binary(target_root.path(), "fixture-bin"),
            binary_path
        );
    }

    #[test]
    fn panics_with_the_build_command_when_the_binary_is_missing() {
        let target_root = tempfile::tempdir().expect("temporary target root should be created");
        let resolution = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            resolve_workspace_binary(target_root.path(), "missing-bin");
        }));

        let panic_message = resolution
            .err()
            .map(|payload| {
                payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| {
                        payload
                            .downcast_ref::<&str>()
                            .map(|message| message.to_string())
                    })
                    .expect("panic payload should be a string")
            })
            .expect("resolving a missing binary should panic");
        assert!(
            panic_message.contains("cargo build --bins -p astronomical-daemon"),
            "panic should name the build command, got: {panic_message}"
        );
    }
}
