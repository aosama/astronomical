use astronomical_native_identity::NativeBuildProfile;

use super::prewarm_native_build::{ProgressRelay, parse_profiles};

fn arguments(argument_words: &[&str]) -> Vec<String> {
    let mut owned_arguments = Vec::new();
    for argument_word in argument_words {
        owned_arguments.push((*argument_word).to_owned());
    }
    owned_arguments
}

#[test]
fn should_accept_one_or_more_profiles_in_order() {
    assert_eq!(
        parse_profiles(&arguments(&["--profile", "core"])),
        Ok(vec![NativeBuildProfile::Core])
    );
    assert_eq!(
        parse_profiles(&arguments(&[
            "--profile",
            "core",
            "--profile",
            "core+memory-contract"
        ])),
        Ok(vec![
            NativeBuildProfile::Core,
            NativeBuildProfile::CoreMemoryContract
        ])
    );
}

#[test]
fn should_require_at_least_one_profile() {
    let parse_error =
        parse_profiles(&arguments(&[])).expect_err("an empty argument list must be refused");

    assert!(parse_error.contains("missing required argument --profile"));
}

#[test]
fn should_refuse_a_dangling_or_unknown_profile() {
    assert!(
        parse_profiles(&arguments(&["--profile"]))
            .expect_err("dangling")
            .contains("requires a native build profile name")
    );
    assert!(
        parse_profiles(&arguments(&["--profile", "full"]))
            .expect_err("unknown")
            .contains("unsupported profile: full")
    );
    assert!(
        parse_profiles(&arguments(&["--verbose"]))
            .expect_err("flag")
            .contains("unsupported argument: --verbose")
    );
}

#[test]
fn should_relay_only_the_progress_appended_since_the_last_read() {
    let progress_directory =
        tempfile::tempdir().expect("a temporary directory should be available");
    let progress_file_path = progress_directory.path().join("progress");
    let mut progress_relay = ProgressRelay::new(progress_file_path.clone());

    assert_eq!(
        progress_relay.read_new_text(),
        "",
        "a missing file relays nothing"
    );
    std::fs::write(&progress_file_path, "status=start\n")
        .expect("the progress file should be writable");
    assert_eq!(progress_relay.read_new_text(), "status=start\n");
    assert_eq!(
        progress_relay.read_new_text(),
        "",
        "nothing new was appended"
    );
    std::fs::write(&progress_file_path, "status=start\nstatus=complete\n").expect("append");
    assert_eq!(progress_relay.read_new_text(), "status=complete\n");
}
