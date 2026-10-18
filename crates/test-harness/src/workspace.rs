//! Locates workspace-built binaries for whole-product journeys.
//!
//! `CARGO_BIN_EXE_<name>` is only set for binaries of the package running the
//! test, so a root-level E2E suite that drives binaries from other packages
//! resolves them through the shared target directory instead.
//!
//! A journey launches the binary built with the same Cargo profile as the test
//! binary itself: `cargo test` spawns `target/debug/<name>` and
//! `cargo test --release` spawns `target/release/<name>`, with no fallback to
//! the other profile. Throughput journeys depend on that rule, because an
//! optimized journey that quietly launched an unoptimized worker would record
//! debug-build rates as release numbers. `ASTRONOMICAL_WORKSPACE_BINARY_PROFILE`
//! names another profile directory explicitly (for example `full-debug`) when a
//! journey must deliberately launch a binary built differently from itself.

use std::path::{Path, PathBuf};

/// Environment variable naming the profile directory under the target root
/// that workspace binaries are launched from, overriding the test binary's own
/// profile. The value is the directory name cargo uses: `debug`, `release`, or
/// a custom profile's name such as `full-debug`.
pub const WORKSPACE_BINARY_PROFILE_ENVIRONMENT_VARIABLE: &str =
    "ASTRONOMICAL_WORKSPACE_BINARY_PROFILE";

/// Returns the executable path of a binary built into the workspace's shared
/// target directory with the same profile as the running test binary (or the
/// profile named by `ASTRONOMICAL_WORKSPACE_BINARY_PROFILE`).
///
/// Panics with the exact build command when the binary is missing from that
/// profile directory, so a skipped build fails loudly at the point of use
/// instead of surfacing as a confusing spawn error deep inside a journey, and
/// so a release journey never measures a leftover debug build. A relative
/// `CARGO_TARGET_DIR` resolves against the test's working directory, which
/// cargo sets to the test package's root.
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
    let profile_directory = requested_profile_directory(
        std::env::var(WORKSPACE_BINARY_PROFILE_ENVIRONMENT_VARIABLE).ok(),
    );
    resolve_workspace_binary(&target_root, &profile_directory, binary_name)
}

/// The profile directory to launch binaries from: the explicit override when
/// it is set to a non-empty value, otherwise the running test binary's own
/// profile.
fn requested_profile_directory(profile_override: Option<String>) -> String {
    profile_override
        .filter(|profile_directory| !profile_directory.is_empty())
        .unwrap_or_else(|| test_binary_profile_directory().to_owned())
}

/// The profile directory the running test binary was built into.
///
/// `debug_assertions` is on for the dev and test profiles and off for release
/// and bench, and this crate is compiled with the same profile as the test
/// binary that links it, so the flag mirrors the test binary's profile without
/// reading cargo metadata at run time. A custom profile inherits the flag from
/// its parent and lands in its own directory, which is what the environment
/// override exists for.
fn test_binary_profile_directory() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
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

/// The two profile directories cargo creates without a custom profile; a
/// missing-binary panic reports a build found in the other one so the reader
/// learns that the profiles were mixed, not that the build was skipped.
const STANDARD_PROFILE_DIRECTORIES: [&str; 2] = ["debug", "release"];

fn resolve_workspace_binary(
    target_root: &Path,
    profile_directory: &str,
    binary_name: &str,
) -> PathBuf {
    let binary_file_name = format!("{binary_name}{}", std::env::consts::EXE_SUFFIX);
    let expected_binary_path = target_root.join(profile_directory).join(&binary_file_name);
    if expected_binary_path.is_file() {
        return expected_binary_path;
    }
    let other_profile_build_notice = STANDARD_PROFILE_DIRECTORIES
        .into_iter()
        .filter(|standard_profile_directory| *standard_profile_directory != profile_directory)
        .map(|other_profile_directory| {
            target_root
                .join(other_profile_directory)
                .join(&binary_file_name)
        })
        .filter(|other_profile_binary_path| other_profile_binary_path.is_file())
        .map(|other_profile_binary_path| {
            format!(
                " A build from another profile exists at {} and is not used; set \
                 {WORKSPACE_BINARY_PROFILE_ENVIRONMENT_VARIABLE} to launch it deliberately.",
                other_profile_binary_path.display()
            )
        })
        .collect::<String>();
    panic!(
        "workspace binary '{binary_name}' was not found at {}; build it with the \
         test binary's profile first by running `{}`.{other_profile_build_notice}",
        expected_binary_path.display(),
        build_command(profile_directory, binary_name)
    );
}

/// The cargo command that builds `binary_name` into `profile_directory`.
/// `--bin` selects the binary across every workspace member, so the command
/// stays correct for a binary of any package without the harness knowing the
/// package name.
fn build_command(profile_directory: &str, binary_name: &str) -> String {
    match profile_directory {
        "debug" => format!("cargo build --bin {binary_name}"),
        "release" => format!("cargo build --release --bin {binary_name}"),
        custom_profile => format!("cargo build --profile {custom_profile} --bin {binary_name}"),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        WORKSPACE_BINARY_PROFILE_ENVIRONMENT_VARIABLE, requested_profile_directory,
        resolve_workspace_binary, test_binary_profile_directory,
    };
    use std::path::{Path, PathBuf};

    fn write_fixture_binary(target_root: &Path, profile_directory: &str) -> PathBuf {
        let binary_path = target_root.join(profile_directory).join("fixture-bin");
        std::fs::create_dir_all(
            binary_path
                .parent()
                .expect("the profile directory has a parent"),
        )
        .expect("the profile directory should be created");
        std::fs::write(&binary_path, b"").expect("the fixture binary should be written");
        binary_path
    }

    fn panic_message_of(resolution: impl FnOnce() + std::panic::UnwindSafe) -> String {
        let panic_payload = std::panic::catch_unwind(resolution)
            .err()
            .expect("the resolution should panic");
        panic_payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| {
                panic_payload
                    .downcast_ref::<&str>()
                    .map(|message| message.to_string())
            })
            .expect("the panic payload should be a string")
    }

    #[test]
    fn should_resolve_the_binary_from_the_requested_profile_directory() {
        let target_root = tempfile::tempdir().expect("temporary target root should be created");
        let debug_binary_path = write_fixture_binary(target_root.path(), "debug");
        let release_binary_path = write_fixture_binary(target_root.path(), "release");

        assert_eq!(
            resolve_workspace_binary(target_root.path(), "debug", "fixture-bin"),
            debug_binary_path
        );
        assert_eq!(
            resolve_workspace_binary(target_root.path(), "release", "fixture-bin"),
            release_binary_path
        );
    }

    #[test]
    fn should_not_fall_back_to_a_debug_build_when_the_release_build_is_missing() {
        let target_root = tempfile::tempdir().expect("temporary target root should be created");
        let debug_binary_path = write_fixture_binary(target_root.path(), "debug");

        let panic_message = panic_message_of(|| {
            resolve_workspace_binary(target_root.path(), "release", "fixture-bin");
        });

        assert!(
            panic_message.contains("cargo build --release --bin fixture-bin"),
            "the panic should name the release build command, got: {panic_message}"
        );
        assert!(
            panic_message.contains(&debug_binary_path.display().to_string()),
            "the panic should point at the unused debug build, got: {panic_message}"
        );
        assert!(
            panic_message.contains(WORKSPACE_BINARY_PROFILE_ENVIRONMENT_VARIABLE),
            "the panic should name the override variable, got: {panic_message}"
        );
    }

    #[test]
    fn should_name_the_debug_build_command_when_the_debug_build_is_missing() {
        let target_root = tempfile::tempdir().expect("temporary target root should be created");

        let panic_message = panic_message_of(|| {
            resolve_workspace_binary(target_root.path(), "debug", "missing-bin");
        });

        assert!(
            panic_message.contains("cargo build --bin missing-bin"),
            "the panic should name the debug build command, got: {panic_message}"
        );
        assert!(
            !panic_message.contains(WORKSPACE_BINARY_PROFILE_ENVIRONMENT_VARIABLE),
            "no other-profile build exists, so the override must not be suggested, got: {panic_message}"
        );
    }

    #[test]
    fn should_name_the_custom_profile_build_command_for_an_overridden_profile() {
        let target_root = tempfile::tempdir().expect("temporary target root should be created");

        let panic_message = panic_message_of(|| {
            resolve_workspace_binary(target_root.path(), "full-debug", "fixture-bin");
        });

        assert!(
            panic_message.contains("cargo build --profile full-debug --bin fixture-bin"),
            "the panic should name the custom profile build command, got: {panic_message}"
        );
    }

    #[test]
    fn should_default_to_the_test_binary_profile_without_an_override() {
        let expected_profile_directory = if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        };

        assert_eq!(test_binary_profile_directory(), expected_profile_directory);
        assert_eq!(
            requested_profile_directory(None),
            expected_profile_directory
        );
        assert_eq!(
            requested_profile_directory(Some(String::new())),
            expected_profile_directory,
            "an empty override should behave like no override"
        );
    }

    #[test]
    fn should_prefer_the_override_profile_over_the_test_binary_profile() {
        assert_eq!(
            requested_profile_directory(Some("full-debug".to_owned())),
            "full-debug"
        );
        assert_eq!(
            requested_profile_directory(Some("release".to_owned())),
            "release"
        );
    }
}
