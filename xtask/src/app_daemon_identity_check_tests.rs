//! Contracts of the check that a bundle's daemon belongs to this build, for
//! the store bundle whose daemon cannot be started outside its app.

use super::build_identity::BuildIdentity;
use super::daemon_identity_check::executable_embeds_build_identity;

fn build_identity(application_version: &str, build_commit: &str) -> BuildIdentity {
    BuildIdentity {
        application_version: application_version.to_owned(),
        repository_url: "https://github.com/example-owner/astronomical".to_owned(),
        build_commit: build_commit.to_owned(),
        build_number: "4242".to_owned(),
        build_date_utc: "20261102".to_owned(),
        is_dirty: false,
    }
}

/// Bytes shaped like a compiled executable: text constants surrounded by
/// bytes that are not valid text.
fn executable_bytes_holding(text_constants: &[&str]) -> Vec<u8> {
    let mut executable_bytes = vec![0xCF, 0xFA, 0xED, 0xFE, 0x00, 0xFF];
    for text_constant in text_constants {
        executable_bytes.extend_from_slice(text_constant.as_bytes());
        executable_bytes.extend_from_slice(&[0x00, 0xFE, 0x80]);
    }
    executable_bytes
}

#[test]
fn should_accept_an_executable_that_holds_this_builds_commit_and_version() {
    let executable_bytes = executable_bytes_holding(&["49af41bd9bba", "0.2.88"]);

    assert!(executable_embeds_build_identity(
        &executable_bytes,
        &build_identity("0.2.88", "49af41bd9bba")
    ));
}

#[test]
fn should_refuse_an_executable_left_over_from_another_commit() {
    let executable_bytes = executable_bytes_holding(&["000000000000", "0.2.88"]);

    assert!(!executable_embeds_build_identity(
        &executable_bytes,
        &build_identity("0.2.88", "49af41bd9bba")
    ));
}

#[test]
fn should_refuse_an_executable_of_this_commit_that_reports_another_version() {
    let executable_bytes = executable_bytes_holding(&["49af41bd9bba", "0.2.87"]);

    assert!(!executable_embeds_build_identity(
        &executable_bytes,
        &build_identity("0.2.88", "49af41bd9bba")
    ));
}

#[test]
fn should_refuse_an_empty_executable_and_an_identity_with_nothing_to_look_for() {
    assert!(!executable_embeds_build_identity(
        &[],
        &build_identity("0.2.88", "49af41bd9bba")
    ));
    // An empty string is found in every file, so an identity that was never
    // filled in must not pass.
    let executable_bytes = executable_bytes_holding(&["49af41bd9bba", "0.2.88"]);
    assert!(!executable_embeds_build_identity(
        &executable_bytes,
        &build_identity("0.2.88", "")
    ));
    assert!(!executable_embeds_build_identity(
        &executable_bytes,
        &build_identity("", "49af41bd9bba")
    ));
}
