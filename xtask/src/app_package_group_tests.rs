//! Contracts of the app group an App Store bundle's two sandboxed
//! applications share: how it is named after the signing team, where it is
//! written, and what a bundle must carry for the two to see one state.

use std::path::Path;

use super::application_group::{
    APPLICATION_GROUP_PLIST_KEY, application_group_entitlements, application_group_identifier,
    stamp_application_group, team_identifier_of,
};
use super::bundle_check::application_group_problems;
use super::channel_profile::{Channel, ChannelProfile};
use super::code_signing::{SigningRequest, rendered_application_entitlements_path, signing_plan};

const STORE_IDENTITY: &str = "Apple Distribution: Example Corp (ABCDE12345)";
const SANDBOX_PROFILE: &str = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<plist version=\"1.0\">\n<dict>\n    <key>com.apple.security.app-sandbox</key>\n    <true/>\n</dict>\n</plist>\n";

fn plist_documents(
    application_group: Option<&str>,
    agent_group: Option<&str>,
) -> (serde_json::Value, serde_json::Value) {
    let mut application_document = serde_json::json!({});
    let mut agent_document = serde_json::json!({});
    if let Some(application_group) = application_group {
        application_document[APPLICATION_GROUP_PLIST_KEY] = serde_json::json!(application_group);
    }
    if let Some(agent_group) = agent_group {
        agent_document[APPLICATION_GROUP_PLIST_KEY] = serde_json::json!(agent_group);
    }
    (application_document, agent_document)
}

#[test]
fn should_read_the_team_from_the_final_parentheses_of_an_identity() {
    assert_eq!(team_identifier_of(STORE_IDENTITY), Some("ABCDE12345"));
    assert_eq!(
        team_identifier_of("3rd Party Mac Developer Application: Example (Holdings) (ZYXWV98765)"),
        Some("ZYXWV98765"),
        "an earlier parenthesis in the name is not the team"
    );
}

#[test]
fn should_read_no_team_from_an_identity_that_does_not_end_with_one() {
    for identity_without_team in [
        "",
        "-",
        "Apple Distribution: Example Corp",
        "Apple Distribution: Example Corp (ABCDE1234)",
        "Apple Distribution: Example Corp (ABCDE123456)",
        "Apple Distribution: Example Corp (abcde12345)",
        "Apple Distribution: Example Corp (ABCDE 2345)",
        "Apple Distribution (ABCDE12345) Example Corp",
    ] {
        assert_eq!(
            team_identifier_of(identity_without_team),
            None,
            "{identity_without_team:?}"
        );
    }
}

#[test]
fn should_name_the_store_group_after_the_signing_team() {
    assert_eq!(
        application_group_identifier(
            &ChannelProfile::for_channel(Channel::AppStore),
            Some(STORE_IDENTITY)
        ),
        Ok(Some("ABCDE12345.astronomical".to_owned()))
    );
}

/// Unsandboxed applications already share the home directory, so a direct
/// channel has no group whoever signs it, or when nobody does.
#[test]
fn should_give_unsandboxed_channels_no_group() {
    for channel in [Channel::Development, Channel::Stable] {
        let channel_profile = ChannelProfile::for_channel(channel);
        assert_eq!(
            application_group_identifier(&channel_profile, Some(STORE_IDENTITY)),
            Ok(None)
        );
        assert_eq!(
            application_group_identifier(&channel_profile, None),
            Ok(None)
        );
    }
}

#[test]
fn should_refuse_a_store_build_whose_identity_names_no_team() {
    let store_profile = ChannelProfile::for_channel(Channel::AppStore);

    for unusable_identity in [None, Some("Apple Distribution: Example Corp")] {
        let refusal = application_group_identifier(&store_profile, unusable_identity)
            .expect_err("a store build without a team cannot name its app group");
        assert!(refusal.contains("ten-character Team ID"), "{refusal}");
    }
}

/// The plan is refused before anything is signed, so no half-signed bundle
/// is left behind by an identity that cannot name the group.
#[test]
fn should_refuse_a_store_signing_plan_whose_identity_names_no_team() {
    let store_profile = ChannelProfile::for_channel(Channel::AppStore);

    let signing_plan_outcome = signing_plan(&SigningRequest {
        bundle_path: Path::new("/fixture/Astronomical.app"),
        channel_profile: &store_profile,
        signing_identity: Some("Apple Distribution: Example Corp"),
        repository_root: Path::new("/repository"),
    });

    assert!(
        signing_plan_outcome
            .expect_err("a store plan needs a team")
            .contains("Team ID")
    );
}

#[test]
fn should_stamp_the_group_as_the_last_entry_of_a_plist() {
    let stamped_plist_text = stamp_application_group(
        "<plist version=\"1.0\">\n<dict>\n  <key>CFBundleName</key><string>Astronomical</string>\n</dict>\n</plist>\n",
        "ABCDE12345.astronomical",
    )
    .expect("a flat plist is stampable");

    assert!(stamped_plist_text.ends_with(
        "  <key>AstronomicalApplicationGroupIdentifier</key><string>ABCDE12345.astronomical</string>\n</dict>\n</plist>\n"
    ));
    assert!(stamped_plist_text.contains("<key>CFBundleName</key><string>Astronomical</string>"));
}

#[test]
fn should_refuse_to_stamp_text_that_is_not_a_plist_dictionary() {
    for unstampable_text in ["", "<plist><array/></plist>", "{\"json\": true}"] {
        assert!(
            stamp_application_group(unstampable_text, "ABCDE12345.astronomical").is_err(),
            "{unstampable_text:?}"
        );
    }
}

#[test]
fn should_add_group_membership_to_the_sandbox_profile_and_keep_the_rest() {
    let rendered_entitlements =
        application_group_entitlements(SANDBOX_PROFILE, "ABCDE12345.astronomical")
            .expect("the sandbox profile is a flat dictionary");

    assert!(rendered_entitlements.contains("<key>com.apple.security.app-sandbox</key>"));
    assert!(rendered_entitlements.contains(
        "<key>com.apple.security.application-groups</key>\n    <array>\n        <string>ABCDE12345.astronomical</string>\n    </array>\n</dict>"
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn should_render_entitlements_from_the_checked_in_profile_that_plutil_accepts() {
    let repository_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask sits in the repository root");
    let checked_in_profile = std::fs::read_to_string(
        repository_root.join(super::code_signing::APPLICATION_ENTITLEMENTS_PATH),
    )
    .expect("the checked-in sandbox profile is readable");
    let scratch_directory = tempfile::tempdir().expect("a temporary directory");
    let rendered_path = scratch_directory.path().join("rendered.entitlements");
    std::fs::write(
        &rendered_path,
        application_group_entitlements(&checked_in_profile, "ABCDE12345.astronomical")
            .expect("the checked-in profile is stampable"),
    )
    .expect("the rendered entitlements are writable");

    let lint_output = std::process::Command::new("plutil")
        .arg("-lint")
        .arg(&rendered_path)
        .output()
        .expect("plutil should run on macOS");

    assert!(lint_output.status.success());
}

/// Rendered beside the bundle, never inside it: a file inside would be sealed
/// into the product by the signature it is used for.
#[test]
fn should_render_entitlements_beside_the_bundle_not_inside_it() {
    let bundle_path = Path::new("/release/out/Astronomical.app");

    let rendered_path = rendered_application_entitlements_path(bundle_path);

    assert_eq!(
        rendered_path,
        Path::new("/release/out/Astronomical.application.entitlements")
    );
    assert!(!rendered_path.starts_with(bundle_path));
}

#[test]
fn should_accept_a_store_bundle_whose_two_plists_name_one_team_group() {
    let (application_document, agent_document) = plist_documents(
        Some("ABCDE12345.astronomical"),
        Some("ABCDE12345.astronomical"),
    );

    assert_eq!(
        application_group_problems(
            &application_document,
            &agent_document,
            &ChannelProfile::for_channel(Channel::AppStore)
        ),
        Vec::<String>::new()
    );
}

/// With different groups, or none, the chat app and the engine would each
/// keep a separate state, which is the defect the group exists to prevent.
#[test]
fn should_refuse_a_store_bundle_with_missing_mismatched_or_teamless_groups() {
    let store_profile = ChannelProfile::for_channel(Channel::AppStore);
    let problems_for = |application_group: Option<&str>, agent_group: Option<&str>| {
        let (application_document, agent_document) =
            plist_documents(application_group, agent_group);
        application_group_problems(&application_document, &agent_document, &store_profile)
    };

    assert_eq!(problems_for(None, None).len(), 1, "no group at all");
    assert_eq!(
        problems_for(Some("ABCDE12345.astronomical"), None),
        vec!["the agent and the main app must name the same app group".to_owned()]
    );
    assert_eq!(
        problems_for(
            Some("ABCDE12345.astronomical"),
            Some("ZYXWV98765.astronomical")
        ),
        vec!["the agent and the main app must name the same app group".to_owned()]
    );
    assert_eq!(
        problems_for(Some(".astronomical"), Some(".astronomical")).len(),
        1,
        "a group without its team prefix is not a group"
    );
    assert_eq!(
        problems_for(Some("ABCDE12345.other"), Some("ABCDE12345.other")).len(),
        1,
        "a group of another name is not this channel's"
    );
}

#[test]
fn should_refuse_an_unsandboxed_bundle_that_names_a_group() {
    for channel in [Channel::Development, Channel::Stable] {
        let channel_profile = ChannelProfile::for_channel(channel);
        let (grouped_application, ungrouped_agent) =
            plist_documents(Some("ABCDE12345.astronomical"), None);
        let (ungrouped_application, plain_agent) = plist_documents(None, None);

        assert_eq!(
            application_group_problems(&grouped_application, &ungrouped_agent, &channel_profile),
            vec!["an unsandboxed bundle must not name an app group".to_owned()]
        );
        assert!(
            application_group_problems(&ungrouped_application, &plain_agent, &channel_profile)
                .is_empty()
        );
    }
}
