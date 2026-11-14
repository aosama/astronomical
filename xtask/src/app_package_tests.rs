//! Release packaging contracts, tested through the typed seams of the
//! pipeline (signing plan, Cargo arguments, layout and plist checks, the
//! disposable-lane re-entry, smoke-test parsing) instead of running a
//! shell builder against fake tools. These replace the retired app-builder
//! journeys and the validator's shell contract suite.

use std::path::Path;

use super::build_identity::BuildIdentity;
use super::bundle_check::plist_identity_problems;
use super::channel_profile::{Channel, ChannelProfile};
use super::code_signing::{
    APPLICATION_ENTITLEMENTS_PATH, INHERITED_ENTITLEMENTS_PATH, SigningRequest, signing_plan,
};
use super::command::{AppCommand, parse_app_command};
use super::info_plist::EXPORT_DECLARATION_KEY;
use super::package_pipeline::{PackageRequest, disposable_reentry_arguments};
use super::release_build_steps::release_cargo_arguments;
use super::smoke_test::{
    BundleInstanceIdentity, advertises_model, has_assistant_output, instance_identity_from_plist,
    status_identity_problem,
};

pub(super) const DEVELOPER_ID: &str = "Developer ID Application: Example (ABCDE12345)";

fn arguments(argument_words: &[&str]) -> Vec<String> {
    let mut owned_arguments = Vec::new();
    for argument_word in argument_words {
        owned_arguments.push((*argument_word).to_owned());
    }
    owned_arguments
}

pub(super) fn release_identity(is_dirty: bool) -> BuildIdentity {
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
        12,
        "Sparkle code, the agent and its bundle, four executables, and the bundle"
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
fn should_assign_app_store_entitlements_and_sign_no_terminal_command() {
    let signing_invocations =
        plan_for(Channel::AppStore, Some(DEVELOPER_ID)).expect("an identity is supplied");

    assert_eq!(
        signing_invocations.len(),
        6,
        "the agent and its bundle, three executables, and the bundle; no Sparkle, no terminal command"
    );
    // Rendered beside the bundle at signing time, because it names the app
    // group of this signing's team.
    let application_entitlements =
        "--entitlements /fixture/Astronomical.application.entitlements".to_owned();
    let inherited_entitlements =
        format!("--entitlements /repository/{INHERITED_ENTITLEMENTS_PATH}");
    // The main app and the agent are each a sandboxed application; the agent's
    // bundle is sealed with the same profile as its executable.
    for sandboxed_application_suffix in [
        "MacOS/astronomical-app",
        "MacOS/astronomical-menu",
        "LoginItems/Astronomical Agent.app",
    ] {
        assert!(
            invocation_arguments_for(&signing_invocations, sandboxed_application_suffix)
                .ends_with(&application_entitlements),
            "{sandboxed_application_suffix}"
        );
    }
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
    // The terminal command could only be signed without the sandbox, so a
    // store bundle has none to sign (issue #2183).
    for signing_invocation in &signing_invocations {
        assert!(
            !signing_invocation
                .code_object_path
                .ends_with("MacOS/astronomical"),
            "{signing_invocation:?}"
        );
    }
}

#[test]
fn should_keep_signing_the_terminal_command_bare_for_the_direct_channels() {
    let signing_invocations =
        plan_for(Channel::Stable, Some(DEVELOPER_ID)).expect("an identity is supplied");

    let terminal_command_arguments =
        invocation_arguments_for(&signing_invocations, "MacOS/astronomical");

    assert!(
        !terminal_command_arguments.contains("--entitlements "),
        "{terminal_command_arguments}"
    );
}

#[test]
fn should_bundle_the_terminal_command_for_every_channel_but_the_store() {
    assert_eq!(
        ChannelProfile::for_channel(Channel::Stable).bundled_rust_executable_names(),
        [
            "astronomicald",
            "astronomical-inference-worker",
            "astronomical"
        ]
    );
    assert_eq!(
        ChannelProfile::for_channel(Channel::Development).bundled_rust_executable_names(),
        [
            "astronomicald",
            "astronomical-inference-worker",
            "astronomical"
        ]
    );
    assert_eq!(
        ChannelProfile::for_channel(Channel::AppStore).bundled_rust_executable_names(),
        ["astronomicald", "astronomical-inference-worker"]
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
    with_export_answer(plist_document, channel)
}

/// Only a store bundle is uploaded, so only its plist carries the answer.
fn with_export_answer(
    mut plist_document: serde_json::Value,
    channel: Channel,
) -> serde_json::Value {
    if channel == Channel::AppStore {
        plist_document[EXPORT_DECLARATION_KEY] = serde_json::json!(false);
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

/// A store bundle's channel is `app-store`, but its engine is the Stable
/// instance and says so; the smoke test once passed the channel name to the
/// engine, which refused it (issue #2190).
#[test]
fn should_expect_the_stable_instance_from_a_sandboxed_store_bundle() {
    let store_identity = instance_identity_from_plist(&serde_json::json!({
        "AstronomicalChannel": "app-store",
        "AstronomicalDaemonPort": 6732,
        "CFBundleShortVersionString": "0.2.88",
        "AstronomicalBuildCommit": "49af41bd9bba"
    }))
    .expect("the plist names its instance identity");
    let stable_engine_status = serde_json::json!({"application": {"channel": "stable", "version": "0.2.88", "commit": "49af41bd9bba"}});
    let misnamed_engine_status = serde_json::json!({"application": {"channel": "app-store", "version": "0.2.88", "commit": "49af41bd9bba"}});

    assert_eq!(store_identity.runtime_instance_name, "stable");
    assert!(store_identity.is_sandboxed);
    assert_eq!(
        status_identity_problem(&stable_engine_status, &store_identity),
        None
    );
    assert!(status_identity_problem(&misnamed_engine_status, &store_identity).is_some());
}

#[test]
fn should_map_each_direct_channel_to_its_own_unsandboxed_instance() {
    let stable_identity = bundle_instance_identity();
    let development_identity = instance_identity_from_plist(&serde_json::json!({
        "AstronomicalChannel": "development",
        "AstronomicalDaemonPort": 6733,
        "CFBundleShortVersionString": "0.2.88",
        "AstronomicalBuildCommit": "49af41bd9bba"
    }))
    .expect("the plist names its instance identity");

    assert_eq!(stable_identity.runtime_instance_name, "stable");
    assert!(!stable_identity.is_sandboxed);
    assert_eq!(development_identity.runtime_instance_name, "development");
    assert!(!development_identity.is_sandboxed);
}

#[test]
fn should_refuse_a_bundle_whose_plist_names_an_unknown_channel() {
    let unknown_channel_result = instance_identity_from_plist(&serde_json::json!({
        "AstronomicalChannel": "nightly",
        "AstronomicalDaemonPort": 6732,
        "CFBundleShortVersionString": "0.2.88",
        "AstronomicalBuildCommit": "49af41bd9bba"
    }));

    assert!(unknown_channel_result.is_err());
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

/// App Store Connect refuses a helper that only inherits its sandbox: the
/// first store package failed validation with error 90296 until the engine
/// and the worker also declared the sandbox itself. Exactly these two keys
/// are allowed, because any further entitlement on an inheriting helper makes
/// macOS refuse to start it.
#[test]
fn should_sign_spawned_store_helpers_as_sandboxed_and_inheriting_and_nothing_else() {
    let repository_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the xtask crate sits in the repository root");
    let inherited_entitlements_text =
        std::fs::read_to_string(repository_root.join(INHERITED_ENTITLEMENTS_PATH))
            .expect("the inherited entitlements file is checked in");

    assert!(inherited_entitlements_text.contains("<key>com.apple.security.app-sandbox</key>"));
    assert!(inherited_entitlements_text.contains("<key>com.apple.security.inherit</key>"));
    assert_eq!(inherited_entitlements_text.matches("<key>").count(), 2);
    assert_eq!(inherited_entitlements_text.matches("<true/>").count(), 2);
}

/// The chat app attaches files and saves pictures and exports through the
/// system's Open and Save dialogs, which a sandboxed app is shown only with
/// the user-selected file entitlement. The first store build lacked it and
/// "Save as PDF" opened nothing (issue #2198). The engine and the worker must
/// not carry it: they never open a person's files, and an inheriting helper
/// with any further entitlement is refused by macOS.
#[test]
fn should_let_the_store_applications_use_files_the_person_selects_and_not_the_helpers() {
    let repository_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the xtask crate sits in the repository root");
    let application_entitlements_text =
        std::fs::read_to_string(repository_root.join(APPLICATION_ENTITLEMENTS_PATH))
            .expect("the application entitlements file is checked in");
    let inherited_entitlements_text =
        std::fs::read_to_string(repository_root.join(INHERITED_ENTITLEMENTS_PATH))
            .expect("the inherited entitlements file is checked in");
    let file_entitlement_key = "<key>com.apple.security.files.user-selected.read-write</key>";

    assert!(application_entitlements_text.contains(file_entitlement_key));
    assert!(!inherited_entitlements_text.contains(file_entitlement_key));
    assert!(!application_entitlements_text.contains("user-selected.read-only"));
}
