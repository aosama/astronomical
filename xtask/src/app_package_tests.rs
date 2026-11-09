//! Release packaging contracts, tested through the typed seams of the
//! pipeline (signing plan, Cargo arguments, layout and plist checks, the
//! disposable-lane re-entry, smoke-test parsing) instead of running a
//! shell builder against fake tools. These replace the retired app-builder
//! journeys and the validator's shell contract suite.

use std::path::{Path, PathBuf};

use super::build_identity::BuildIdentity;
use super::bundle_assembly::{BundleInputs, assemble_bundle, missing_inputs};
use super::bundle_check::{hidden_entry_problems, layout_problems, plist_identity_problems};
use super::channel_profile::{Channel, ChannelProfile};
use super::code_signing::{
    APPLICATION_ENTITLEMENTS_PATH, INHERITED_ENTITLEMENTS_PATH, SigningRequest, signing_plan,
};
use super::command::{AppCommand, parse_app_command};
use super::info_plist::render_info_plist;
use super::package_pipeline::{PackageRequest, disposable_reentry_arguments};
use super::release_build_steps::release_cargo_arguments;
use super::smoke_test::{
    BundleInstanceIdentity, advertises_model, has_assistant_output, instance_identity_from_plist,
    status_identity_problem,
};

const DEVELOPER_ID: &str = "Developer ID Application: Example (ABCDE12345)";

fn arguments(argument_words: &[&str]) -> Vec<String> {
    let mut owned_arguments = Vec::new();
    for argument_word in argument_words {
        owned_arguments.push((*argument_word).to_owned());
    }
    owned_arguments
}

fn release_identity(is_dirty: bool) -> BuildIdentity {
    BuildIdentity {
        application_version: "0.2.88".to_owned(),
        repository_url: "https://github.com/example-owner/astronomical".to_owned(),
        build_commit: "49af41bd9bba".to_owned(),
        build_number: "4242".to_owned(),
        build_date_utc: "20261102".to_owned(),
        is_dirty,
    }
}

fn plan_for(
    channel: Channel,
    signing_identity: Option<&str>,
) -> Result<Vec<super::code_signing::SigningInvocation>, String> {
    let channel_profile = ChannelProfile::for_channel(channel);
    signing_plan(&SigningRequest {
        bundle_path: Path::new("/fixture/Astronomical.app"),
        channel_profile: &channel_profile,
        signing_identity,
        repository_root: Path::new("/repository"),
    })
}

fn invocation_arguments_for(
    signing_invocations: &[super::code_signing::SigningInvocation],
    code_object_suffix: &str,
) -> String {
    for signing_invocation in signing_invocations {
        if signing_invocation
            .code_object_path
            .ends_with(code_object_suffix)
        {
            return signing_invocation.codesign_arguments.join(" ");
        }
    }
    panic!("no signing invocation for {code_object_suffix}");
}

#[test]
fn should_sign_stable_with_developer_id_hardened_runtime_and_timestamp() {
    let signing_invocations =
        plan_for(Channel::Stable, Some(DEVELOPER_ID)).expect("a Developer ID plan is valid");

    assert_eq!(
        signing_invocations.len(),
        10,
        "Sparkle code, four executables, and the bundle"
    );
    for signing_invocation in &signing_invocations {
        assert!(
            signing_invocation
                .codesign_arguments
                .join(" ")
                .starts_with(&format!(
                    "--force --sign {DEVELOPER_ID} --options runtime --timestamp"
                )),
            "{signing_invocation:?}"
        );
        assert!(
            !signing_invocation
                .codesign_arguments
                .contains(&"--entitlements".to_owned())
        );
    }
}

#[test]
fn should_sign_unsigned_stable_candidates_ad_hoc() {
    let signing_invocations =
        plan_for(Channel::Stable, None).expect("an ad-hoc Stable candidate is valid");

    assert!(
        invocation_arguments_for(&signing_invocations, "Astronomical.app")
            .starts_with("--force --sign -")
    );
}

#[test]
fn should_assign_app_store_entitlements_and_keep_the_cli_bare() {
    let signing_invocations =
        plan_for(Channel::AppStore, Some(DEVELOPER_ID)).expect("an identity is supplied");

    assert_eq!(
        signing_invocations.len(),
        5,
        "four executables and the bundle, no Sparkle"
    );
    let application_entitlements =
        format!("--entitlements /repository/{APPLICATION_ENTITLEMENTS_PATH}");
    let inherited_entitlements =
        format!("--entitlements /repository/{INHERITED_ENTITLEMENTS_PATH}");
    assert!(
        invocation_arguments_for(&signing_invocations, "MacOS/astronomical-menu")
            .ends_with(&application_entitlements)
    );
    assert!(
        invocation_arguments_for(&signing_invocations, "MacOS/astronomicald")
            .ends_with(&inherited_entitlements)
    );
    assert!(
        invocation_arguments_for(&signing_invocations, "MacOS/astronomical-inference-worker")
            .ends_with(&inherited_entitlements)
    );
    assert!(
        invocation_arguments_for(&signing_invocations, "Astronomical.app")
            .ends_with(&application_entitlements)
    );
    assert!(
        invocation_arguments_for(&signing_invocations, "MacOS/astronomical")
            .ends_with("--preserve-metadata=entitlements")
    );
}

#[test]
fn should_refuse_an_app_store_build_without_a_signing_identity() {
    let plan_error =
        plan_for(Channel::AppStore, None).expect_err("store entitlements need a real signature");

    assert!(plan_error.contains("--signing-identity"), "{plan_error}");
}

#[test]
fn should_enable_the_app_store_state_root_feature_only_for_the_store() {
    let stable_arguments = release_cargo_arguments(
        &ChannelProfile::for_channel(Channel::Stable),
        "aarch64-apple-darwin",
        Path::new("/lane-fixture/app-release"),
    );
    let store_arguments = release_cargo_arguments(
        &ChannelProfile::for_channel(Channel::AppStore),
        "aarch64-apple-darwin",
        Path::new("/lane-fixture/app-release"),
    );

    assert_eq!(
        &stable_arguments[..4],
        ["build", "--release", "--target", "aarch64-apple-darwin"]
    );
    assert!(!stable_arguments.contains(&"--features".to_owned()));
    assert_eq!(
        &store_arguments[store_arguments.len() - 2..],
        ["--features", "astronomical-daemon/app-store-state-root"]
    );
    for binary_name in [
        "astronomicald",
        "astronomical-inference-worker",
        "astronomical",
    ] {
        assert!(
            stable_arguments.contains(&binary_name.to_owned()),
            "{binary_name}"
        );
    }
}

/// sccache keys objects on every `CARGO_*` variable, so the per-run lane
/// directory must reach Cargo as a flag for release objects to be reusable.
#[test]
fn should_pass_the_disposable_target_directory_as_a_flag() {
    let stable_arguments = release_cargo_arguments(
        &ChannelProfile::for_channel(Channel::Stable),
        "aarch64-apple-darwin",
        Path::new("/lane-fixture/app-release"),
    );

    assert_eq!(
        &stable_arguments[4..6],
        ["--target-dir", "/lane-fixture/app-release"]
    );
}

#[test]
fn should_parse_package_requests_and_refuse_development_or_malformed_flags() {
    assert_eq!(
        parse_app_command(&arguments(&[
            "package",
            "--channel",
            "stable",
            "--signing-identity",
            DEVELOPER_ID
        ])),
        Ok(AppCommand::Package(PackageRequest {
            channel: Channel::Stable,
            signing_identity: Some(DEVELOPER_ID.to_owned())
        }))
    );
    assert!(parse_app_command(&arguments(&["package", "--channel", "development"])).is_err());
    assert!(parse_app_command(&arguments(&["package"])).is_err());
    assert!(
        parse_app_command(&arguments(&["package", "--channel", "--signing-identity"])).is_err()
    );
    assert!(
        parse_app_command(&arguments(&[
            "package",
            "--channel",
            "stable",
            "--channel",
            "stable"
        ]))
        .is_err()
    );
    assert!(
        parse_app_command(&arguments(&["package", "--channel", "stable", "--dry-run"])).is_err()
    );
}

#[test]
fn should_reenter_the_app_release_lane_with_the_same_request() {
    let package_request = PackageRequest {
        channel: Channel::AppStore,
        signing_identity: Some(DEVELOPER_ID.to_owned()),
    };

    let reentry_arguments = disposable_reentry_arguments(
        Path::new("/repository"),
        &package_request,
        Path::new("/repository/target/debug/xtask"),
    );

    assert_eq!(
        reentry_arguments,
        arguments(&[
            "xtask",
            "disposable-target",
            "--lane",
            "app-release",
            "--",
            "/repository/target/debug/xtask",
            "--repository-root",
            "/repository",
            "app",
            "package",
            "--channel",
            "app-store",
            "--signing-identity",
            DEVELOPER_ID,
        ])
    );
}

/// Assembles a fixture bundle through the real assembly code so layout
/// checks run against what the pipeline actually writes.
fn assembled_fixture_bundle(channel: Channel) -> (tempfile::TempDir, PathBuf) {
    let fixture_directory = tempfile::tempdir().expect("a fixture directory");
    let fixture_root = fixture_directory.path();
    let channel_profile = ChannelProfile::for_channel(channel);
    let swift_build = fixture_root.join("apps/astronomical-menu/.build");
    write_fixture_file(
        &swift_build
            .join("release")
            .join(channel_profile.menu_swift_product),
        "menu",
        true,
    );
    write_fixture_file(
        &swift_build.join("release/ThinTalk_ThinTalkCanvas.bundle/index.html"),
        "<html>",
        false,
    );
    write_fixture_file(
        &swift_build.join("checkouts/Sparkle/LICENSE"),
        "sparkle",
        false,
    );
    write_fixture_file(
        &swift_build.join("release/Sparkle.framework/Versions/B/Sparkle"),
        "sparkle",
        false,
    );
    for notice_path in [
        "LICENSE",
        "third-party/THIRD_PARTY_NOTICES",
        "third-party/RUST_DEPENDENCY_NOTICES",
        "mlx.metallib",
        "icon.icns",
    ] {
        write_fixture_file(&fixture_root.join(notice_path), "notice", false);
    }
    for executable_name in [
        "astronomicald",
        "astronomical-inference-worker",
        "astronomical",
    ] {
        write_fixture_file(
            &fixture_root.join("release").join(executable_name),
            "binary",
            true,
        );
    }
    let bundle_inputs = BundleInputs::for_repository(
        fixture_root,
        &channel_profile,
        &fixture_root.join("release"),
        fixture_root.join("mlx.metallib"),
        fixture_root.join("icon.icns"),
    );
    assert!(
        missing_inputs(&bundle_inputs).is_empty(),
        "{:?}",
        missing_inputs(&bundle_inputs)
    );
    let bundle_path = fixture_root
        .join("out")
        .join(channel_profile.bundle_directory_name());
    let plist_text = render_info_plist(
        &channel_profile,
        &release_identity(false),
        "https://example-owner.github.io/astronomical/appcast.xml",
    );
    assemble_bundle_without_rpath(&bundle_path, &bundle_inputs, &plist_text, &channel_profile);
    (fixture_directory, bundle_path)
}

/// `install_name_tool` cannot edit a text fixture, so the Sparkle
/// framework input is dropped before assembly and copied afterwards; the
/// layout under test is otherwise exactly what `assemble_bundle` writes.
fn assemble_bundle_without_rpath(
    bundle_path: &Path,
    bundle_inputs: &BundleInputs,
    plist_text: &str,
    channel_profile: &ChannelProfile,
) {
    let mut rpath_free_inputs = bundle_inputs.clone();
    rpath_free_inputs.sparkle_framework = None;
    assemble_bundle(bundle_path, &rpath_free_inputs, plist_text)
        .expect("the fixture bundle should assemble");
    if channel_profile.ships_sparkle {
        write_fixture_file(
            &bundle_path.join("Contents/Frameworks/Sparkle.framework/Versions/B/Sparkle"),
            "sparkle",
            false,
        );
    }
}

fn write_fixture_file(file_path: &Path, file_text: &str, is_executable: bool) {
    std::fs::create_dir_all(file_path.parent().expect("a parent")).expect("fixture directory");
    std::fs::write(file_path, file_text).expect("fixture file");
    if is_executable {
        let executable_permissions = std::os::unix::fs::PermissionsExt::from_mode(0o755);
        std::fs::set_permissions(file_path, executable_permissions).expect("executable bit");
    }
}

#[test]
fn should_find_no_layout_problem_in_freshly_assembled_bundles() {
    for channel in [Channel::Development, Channel::Stable, Channel::AppStore] {
        let (_fixture_directory, bundle_path) = assembled_fixture_bundle(channel);

        assert!(
            layout_problems(&bundle_path, &ChannelProfile::for_channel(channel)).is_empty(),
            "{channel:?}"
        );
    }
}

#[test]
fn should_report_a_missing_metallib_executable_bit_and_stray_sparkle() {
    let (_fixture_directory, bundle_path) = assembled_fixture_bundle(Channel::Stable);
    std::fs::remove_file(bundle_path.join("Contents/Resources/share/mlx/mlx.metallib"))
        .expect("remove metallib");
    let plain_permissions = std::os::unix::fs::PermissionsExt::from_mode(0o644);
    std::fs::set_permissions(
        bundle_path.join("Contents/MacOS/astronomicald"),
        plain_permissions,
    )
    .expect("drop the executable bit");

    let stable_problems =
        layout_problems(&bundle_path, &ChannelProfile::for_channel(Channel::Stable));
    let store_problems = layout_problems(
        &bundle_path,
        &ChannelProfile::for_channel(Channel::AppStore),
    );

    assert!(
        stable_problems.contains(&"required bundled MLX AOT metallib is unavailable".to_owned()),
        "{stable_problems:?}"
    );
    assert!(
        stable_problems.contains(&"bundled executable is unavailable: astronomicald".to_owned()),
        "{stable_problems:?}"
    );
    assert!(
        store_problems
            .contains(&"an App Store bundle must not embed Sparkle or its license".to_owned()),
        "{store_problems:?}"
    );
}

/// The v0.2.88 leak: a git-ignored Playwright folder inside the Thin Talk
/// web resources was copied whole into the shipped bundle.
#[test]
fn should_report_a_leaked_playwright_folder_inside_the_canvas_resources() {
    let (_fixture_directory, bundle_path) = assembled_fixture_bundle(Channel::Stable);
    let leaked_folder =
        bundle_path.join("Contents/Resources/ThinTalk_ThinTalkCanvas.bundle/.playwright-cli");
    write_fixture_file(&leaked_folder.join("console.log"), "debug", false);

    let stable_problems =
        layout_problems(&bundle_path, &ChannelProfile::for_channel(Channel::Stable));

    assert_eq!(
        stable_problems,
        vec![
            "hidden entry inside the bundle: Contents/Resources/ThinTalk_ThinTalkCanvas.bundle/.playwright-cli"
                .to_owned()
        ]
    );
}

#[test]
fn should_report_a_hidden_file_at_any_depth_of_the_bundle() {
    let (_fixture_directory, bundle_path) = assembled_fixture_bundle(Channel::Development);
    write_fixture_file(&bundle_path.join("Contents/.DS_Store"), "finder", false);

    assert_eq!(
        hidden_entry_problems(&bundle_path),
        vec!["hidden entry inside the bundle: Contents/.DS_Store".to_owned()]
    );
}

/// Sparkle's framework links `Versions/Current`; following symlinks could
/// walk out of the bundle and blame it for files it does not contain.
#[test]
fn should_not_follow_a_symlink_out_of_the_bundle() {
    let (fixture_directory, bundle_path) = assembled_fixture_bundle(Channel::Stable);
    let outside_directory = fixture_directory.path().join("outside");
    write_fixture_file(&outside_directory.join(".hidden"), "outside", false);
    std::os::unix::fs::symlink(&outside_directory, bundle_path.join("Contents/linked"))
        .expect("symlink out of the bundle");

    assert!(hidden_entry_problems(&bundle_path).is_empty());
}

#[test]
fn should_cap_the_hidden_entry_report() {
    let (_fixture_directory, bundle_path) = assembled_fixture_bundle(Channel::Stable);
    for hidden_index in 0..15 {
        write_fixture_file(
            &bundle_path.join(format!("Contents/Resources/.stray-{hidden_index:02}")),
            "stray",
            false,
        );
    }

    assert_eq!(hidden_entry_problems(&bundle_path).len(), 10);
}

fn plist_document_for(channel: Channel, is_dirty: bool) -> serde_json::Value {
    let channel_profile = ChannelProfile::for_channel(channel);
    let mut plist_document = serde_json::json!({
        "AstronomicalChannel": channel.as_channel_name(),
        "CFBundleIdentifier": channel_profile.bundle_identifier,
        "CFBundleIconFile": "Astronomical.icns",
        "AstronomicalDaemonPort": channel_profile.daemon_port,
        "CFBundleVersion": "4242",
        "AstronomicalBuildDate": "20261102",
        "AstronomicalBuildDirty": is_dirty,
        "CFBundleURLTypes": [{"CFBundleURLSchemes": [channel_profile.chat_url_scheme]}]
    });
    if let Some(state_directory_name) = channel_profile.state_directory_name {
        plist_document["AstronomicalStateDirectoryName"] = serde_json::json!(state_directory_name);
    }
    if channel_profile.ships_sparkle {
        for sparkle_key in ["SUFeedURL", "SUPublicEDKey", "SUEnableAutomaticChecks"] {
            plist_document[sparkle_key] = serde_json::json!("present");
        }
    }
    plist_document
}

#[test]
fn should_accept_a_plist_that_matches_its_channel() {
    for channel in [Channel::Development, Channel::Stable, Channel::AppStore] {
        let identity_problems = plist_identity_problems(
            &plist_document_for(channel, false),
            &ChannelProfile::for_channel(channel),
        );

        assert!(
            identity_problems.is_empty(),
            "{channel:?}: {identity_problems:?}"
        );
    }
}

#[test]
fn should_refuse_a_dirty_release_and_a_plist_built_for_another_channel() {
    let stable_profile = ChannelProfile::for_channel(Channel::Stable);

    let dirty_problems =
        plist_identity_problems(&plist_document_for(Channel::Stable, true), &stable_profile);
    let wrong_channel_problems = plist_identity_problems(
        &plist_document_for(Channel::Development, false),
        &stable_profile,
    );
    let development_dirty_problems = plist_identity_problems(
        &plist_document_for(Channel::Development, true),
        &ChannelProfile::for_channel(Channel::Development),
    );

    assert!(
        dirty_problems.contains(&"a release bundle must not be dirty".to_owned()),
        "{dirty_problems:?}"
    );
    assert!(
        wrong_channel_problems.contains(&"bundle channel and daemon port disagree".to_owned()),
        "{wrong_channel_problems:?}"
    );
    assert!(
        development_dirty_problems.is_empty(),
        "Development builds may be dirty"
    );
}

#[test]
fn should_refuse_a_development_bundle_that_registers_the_stable_chat_scheme() {
    let mut development_plist = plist_document_for(Channel::Development, true);
    development_plist["CFBundleURLTypes"] =
        serde_json::json!([{"CFBundleURLSchemes": ["astronomical"]}]);

    let identity_problems = plist_identity_problems(
        &development_plist,
        &ChannelProfile::for_channel(Channel::Development),
    );

    assert!(
        identity_problems.contains(
            &"the bundle must register only the astronomical-dev URL scheme for the chat handoff"
                .to_owned()
        )
    );
}

#[test]
fn should_refuse_a_development_bundle_that_also_registers_the_stable_chat_scheme() {
    let mut development_plist = plist_document_for(Channel::Development, true);
    development_plist["CFBundleURLTypes"] =
        serde_json::json!([{"CFBundleURLSchemes": ["astronomical-dev", "astronomical"]}]);

    let identity_problems = plist_identity_problems(
        &development_plist,
        &ChannelProfile::for_channel(Channel::Development),
    );

    assert!(
        identity_problems.contains(
            &"the bundle must register only the astronomical-dev URL scheme for the chat handoff"
                .to_owned()
        )
    );
}

#[test]
fn should_refuse_a_stable_bundle_that_registers_the_development_chat_scheme() {
    let mut stable_plist = plist_document_for(Channel::Stable, false);
    stable_plist["CFBundleURLTypes"] =
        serde_json::json!([{"CFBundleURLSchemes": ["astronomical-dev"]}]);

    let identity_problems =
        plist_identity_problems(&stable_plist, &ChannelProfile::for_channel(Channel::Stable));

    assert!(
        identity_problems.contains(
            &"the bundle must register only the astronomical URL scheme for the chat handoff"
                .to_owned()
        )
    );
}

#[test]
fn should_refuse_update_keys_state_directories_and_missing_url_schemes_in_the_store_plist() {
    let mut store_plist = plist_document_for(Channel::AppStore, false);
    store_plist["SUFeedURL"] = serde_json::json!("https://example.invalid/appcast.xml");
    store_plist["AstronomicalStateDirectoryName"] = serde_json::json!(".astronomical");
    store_plist["CFBundleURLTypes"] = serde_json::json!([]);
    store_plist["AstronomicalBuildDate"] = serde_json::json!("2026-11-02");

    let identity_problems = plist_identity_problems(
        &store_plist,
        &ChannelProfile::for_channel(Channel::AppStore),
    );

    for expected_problem in [
        "an App Store bundle must not contain the SUFeedURL update key",
        "an App Store bundle must not embed a state directory name",
        "the bundle must register only the astronomical URL scheme for the chat handoff",
        "bundle build date must use YYYYMMDD",
    ] {
        assert!(
            identity_problems.contains(&expected_problem.to_owned()),
            "{expected_problem}: {identity_problems:?}"
        );
    }
}

#[test]
fn should_parse_smoke_test_requests() {
    assert!(matches!(
        parse_app_command(&arguments(&[
            "smoke-test",
            "--app-bundle",
            "/fixture/A.app",
            "--real-model",
            "qwen"
        ])),
        Ok(AppCommand::SmokeTest(_))
    ));
    assert!(parse_app_command(&arguments(&["smoke-test"])).is_err());
    assert!(parse_app_command(&arguments(&["smoke-test", "--app-bundle"])).is_err());
}

fn bundle_instance_identity() -> BundleInstanceIdentity {
    instance_identity_from_plist(&serde_json::json!({
        "AstronomicalChannel": "stable",
        "AstronomicalDaemonPort": 6732,
        "CFBundleShortVersionString": "0.2.88",
        "AstronomicalBuildCommit": "49af41bd9bba"
    }))
    .expect("the plist names its instance identity")
}

#[test]
fn should_accept_only_a_daemon_serving_the_bundle_identity() {
    let matching_status = serde_json::json!({"application": {"channel": "stable", "version": "0.2.88", "commit": "49af41bd9bba"}});
    let stale_status = serde_json::json!({"application": {"channel": "stable", "version": "0.2.87", "commit": "49af41bd9bba"}});

    assert_eq!(
        status_identity_problem(&matching_status, &bundle_instance_identity()),
        None
    );
    assert!(status_identity_problem(&stale_status, &bundle_instance_identity()).is_some());
    assert!(status_identity_problem(&serde_json::json!({}), &bundle_instance_identity()).is_some());
    assert!(
        instance_identity_from_plist(&serde_json::json!({"AstronomicalChannel": "stable"}))
            .is_err()
    );
}

#[test]
fn should_require_an_advertised_model_and_assistant_output_for_the_real_model_probe() {
    let models_document = serde_json::json!({"data": [{"id": "qwen-fixture"}]});

    assert!(advertises_model(&models_document, "qwen-fixture"));
    assert!(!advertises_model(&models_document, "absent-model"));
    assert!(!advertises_model(&serde_json::json!({}), "qwen-fixture"));
    assert!(has_assistant_output(
        &serde_json::json!({"choices": [{"message": {"content": "Romeo"}}]})
    ));
    assert!(has_assistant_output(
        &serde_json::json!({"choices": [{"message": {"content": "", "reasoning_content": "thinking"}}]})
    ));
    assert!(!has_assistant_output(
        &serde_json::json!({"choices": [{"message": {"content": ""}}]})
    ));
}
