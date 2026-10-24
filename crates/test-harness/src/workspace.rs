//! Locates workspace-built binaries for whole-product journeys.
//!
//! `CARGO_BIN_EXE_<name>` is only set for binaries of the package running the
//! test, so a root-level E2E suite that drives binaries from other packages
//! resolves them through the shared target directory instead.
//!
//! An explicit test run uses the exact executables reported by Cargo. Plain
//! test invocations resolve siblings of their own Cargo profile directory,
//! including custom profiles and target triples, without selecting a different
//! profile or falling back to an older build.

use std::path::{Path, PathBuf};

use crate::test_run::TestRunContext;

/// Returns an explicitly built executable or a sibling of this test's profile.
///
/// Panics with the exact build command when the binary is missing from that
/// profile directory, so a skipped build fails loudly at the point of use
/// instead of surfacing as a confusing spawn error deep inside a journey, and
/// so a release journey never measures a leftover debug build.
pub fn workspace_binary(binary_name: &str) -> PathBuf {
    let test_working_directory =
        std::env::current_dir().expect("the test working directory should be readable");
    if let Some(run_context) = TestRunContext::read_from_directory(&test_working_directory)
        .expect("the explicit test-run context should load")
    {
        return resolve_run_binary(&run_context, binary_name);
    }
    let test_executable_path =
        std::env::current_exe().expect("the test executable path should resolve");
    let profile_directory = executing_profile_directory(&test_executable_path)
        .expect("the test executable should identify its Cargo profile");
    let profile_name = profile_directory
        .file_name()
        .and_then(|profile_name| profile_name.to_str())
        .expect("the Cargo profile directory should have a text name");
    resolve_workspace_binary(
        profile_directory
            .parent()
            .expect("the Cargo profile directory should have a parent"),
        profile_name,
        binary_name,
    )
}

fn resolve_run_binary(run_context: &TestRunContext, binary_name: &str) -> PathBuf {
    let executable_path = run_context
        .workspace_binaries
        .get(binary_name)
        .unwrap_or_else(|| {
            panic!("cargo xtask test-run did not build requested workspace binary '{binary_name}'")
        });
    assert!(
        executable_path.is_file(),
        "the explicitly built workspace binary is missing: {}",
        executable_path.display()
    );
    executable_path.clone()
}

fn executing_profile_directory(test_executable_path: &Path) -> Result<PathBuf, String> {
    let executable_directory = test_executable_path
        .parent()
        .ok_or_else(|| "the test executable has no parent directory".to_owned())?;
    if executable_directory
        .file_name()
        .is_some_and(|name| name == "deps")
    {
        return executable_directory
            .parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| "the test executable has no Cargo profile directory".to_owned());
    }
    // Cargo's artifact layout can place test executables under
    // profile/build/package/fingerprint/out instead of profile/deps.
    if executable_directory
        .file_name()
        .is_some_and(|name| name == "out")
    {
        for build_directory in executable_directory.ancestors().skip(2).take(2) {
            if build_directory
                .file_name()
                .is_some_and(|name| name == "build")
            {
                return build_directory
                    .parent()
                    .map(Path::to_path_buf)
                    .ok_or_else(|| {
                        "the test executable has no Cargo profile directory".to_owned()
                    });
            }
        }
    }
    Err(
        "the test executable does not identify its Cargo profile; use cargo xtask test-run"
            .to_owned(),
    )
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
                " A build from another profile exists at {} and is not used; \
                 cargo xtask test-run selects and builds an explicit profile.",
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
    use super::{executing_profile_directory, resolve_run_binary, resolve_workspace_binary};
    use crate::test_run::TestRunContext;
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
            panic_message.contains("cargo xtask test-run"),
            "the panic should name the explicit runner, got: {panic_message}"
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
            !panic_message.contains("A build from another profile"),
            "no other-profile build exists, so none should be suggested, got: {panic_message}"
        );
    }

    #[test]
    fn should_name_the_custom_profile_build_command() {
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
    fn should_resolve_custom_profiles_and_target_triples_from_the_test_executable() {
        let target_root = tempfile::tempdir().expect("target root");
        let profile_directory = target_root.path().join("target-triple/full-debug");
        assert_eq!(
            executing_profile_directory(&profile_directory.join("deps/test-executable"))
                .expect("the custom profile should resolve"),
            profile_directory
        );
    }

    #[test]
    fn should_reject_executables_that_do_not_identify_a_cargo_test_profile() {
        assert!(executing_profile_directory(Path::new("test-executable")).is_err());
        assert!(executing_profile_directory(Path::new("profile/test-executable")).is_err());
    }

    #[test]
    fn should_resolve_the_running_test_profile_and_cargo_build_output_layouts() {
        assert!(
            executing_profile_directory(
                &std::env::current_exe().expect("the test executable should resolve")
            )
            .is_ok()
        );
        let target_root = tempfile::tempdir().expect("target root");
        let profile_directory = target_root.path().join("full-debug");
        for test_executable_path in [
            profile_directory.join("build/package/fingerprint/out/test-executable"),
            profile_directory.join("build/package-fingerprint/out/test-executable"),
        ] {
            assert_eq!(
                executing_profile_directory(&test_executable_path).expect("build output profile"),
                profile_directory
            );
        }
    }

    #[test]
    fn should_launch_only_the_executable_recorded_by_the_explicit_runner() {
        let target_root = tempfile::tempdir().expect("target root");
        let executable_path = write_fixture_binary(target_root.path(), "full-debug");
        let mut run_context = TestRunContext::default();
        run_context
            .workspace_binaries
            .insert("fixture-bin".to_owned(), executable_path.clone());
        assert_eq!(
            resolve_run_binary(&run_context, "fixture-bin"),
            executable_path
        );
    }

    #[test]
    fn should_reject_missing_explicit_executables_without_falling_back() {
        let run_context = TestRunContext::default();
        let panic_message = panic_message_of(|| {
            resolve_run_binary(&run_context, "fixture-bin");
        });
        assert!(panic_message.contains("did not build requested workspace binary"));
    }
}
