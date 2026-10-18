//! The builder journey contracts: what one builder invocation is, what it
//! reports back, and the assertion helpers that read the produced bundle
//! the same way the real validator does.

use std::path::{Path, PathBuf};
use std::process::Command;

use super::sandbox::AppBuilderSandbox;

pub(crate) const SPARKLE_PUBLIC_ED25519_KEY: &str = "g0URwy+j86uDYcmOu0k/IUVWwCOSrGOPSoFnVoYQ9AQ=";
pub(crate) const EXPECTED_UPDATE_FEED_URL: &str =
    "https://example.github.io/astronomical/appcast.xml";
pub(crate) const APP_STORE_STATE_ROOT_FEATURE: &str =
    "--features astronomical-daemon/app-store-state-root";

pub(crate) struct BuilderJourney {
    pub(crate) channel: &'static str,
    pub(crate) signing_identity: Option<&'static str>,
    pub(crate) metallib_path: PathBuf,
    pub(crate) journey_log_stem: &'static str,
}

pub(crate) struct BuilderRunOutcome {
    pub(crate) succeeded: bool,
    pub(crate) standard_output_text: String,
    pub(crate) standard_error_text: String,
    pub(crate) codesign_log_path: PathBuf,
    pub(crate) cargo_target_record_path: PathBuf,
}

pub(crate) fn file_text(path: &Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("{} should be readable: {error}", path.display()))
}

/// Reads one Info.plist value the way the real validator does: plutil
/// extraction by key, with absence expressed as the extraction failing.
/// Absolute-path invocation bypasses the shimmed plutil on PATH.
pub(crate) fn plist_string(bundle_path: &Path, plist_key: &str) -> Option<String> {
    let extraction_output = Command::new("/usr/bin/plutil")
        .args(["-extract", plist_key, "raw", "-o", "-"])
        .arg(bundle_path.join("Contents/Info.plist"))
        .output()
        .expect("plutil should run in the test process");
    if !extraction_output.status.success() {
        return None;
    }
    Some(
        String::from_utf8_lossy(&extraction_output.stdout)
            .trim()
            .to_owned(),
    )
}

pub(crate) fn resource_text(bundle_path: &Path, resource_relative_path: &str) -> String {
    file_text(
        &bundle_path
            .join("Contents/Resources")
            .join(resource_relative_path),
    )
    .trim()
    .to_owned()
}

pub(crate) fn log_line_count_containing(log_text: &str, needle: &str) -> usize {
    log_text
        .lines()
        .filter(|log_line| log_line.contains(needle))
        .count()
}

pub(crate) fn preparatory_development_journey(sandbox: &AppBuilderSandbox) -> BuilderJourney {
    BuilderJourney {
        channel: "development",
        signing_identity: None,
        metallib_path: sandbox.fixture_metallib_path.clone(),
        journey_log_stem: "preparatory-development",
    }
}

/// The disposable-target record is written by `printf '%s\n'`, so the path
/// must be trimmed before existence is probed — an untrimmed path with its
/// trailing newline never exists and the cleanup contract would pass
/// vacuously even when the runner leaked the target.
pub(crate) fn assert_disposable_target_was_removed(outcome: &BuilderRunOutcome) {
    let recorded_cargo_target = file_text(&outcome.cargo_target_record_path)
        .trim()
        .to_owned();
    assert!(
        !recorded_cargo_target.is_empty(),
        "the fake cargo never recorded its disposable Cargo target"
    );
    assert!(
        !Path::new(&recorded_cargo_target).exists(),
        "the build retained its disposable Cargo target: {recorded_cargo_target}"
    );
}
