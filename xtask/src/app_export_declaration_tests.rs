//! Contracts of the export-compliance answer an App Store bundle carries:
//! which plist is given it, and what the bundle check accepts in its place.

use super::build_identity::BuildIdentity;
use super::bundle_check::plist_identity_problems;
use super::channel_profile::{Channel, ChannelProfile};
use super::info_plist::{render_agent_info_plist, render_info_plist};

const EXPORT_ANSWER_KEY: &str = "ITSAppUsesNonExemptEncryption";
const MISSING_ANSWER_PROBLEM: &str =
    "an App Store bundle must declare ITSAppUsesNonExemptEncryption as false";

fn sample_build_identity() -> BuildIdentity {
    BuildIdentity {
        application_version: "0.2.88".to_owned(),
        repository_url: "https://github.com/example-owner/astronomical".to_owned(),
        build_commit: "49af41bd9bba".to_owned(),
        build_number: "4242".to_owned(),
        build_date_utc: "20261102".to_owned(),
        is_dirty: false,
    }
}

fn reports_missing_answer(export_answer: Option<serde_json::Value>, channel: Channel) -> bool {
    let mut plist_document = serde_json::json!({});
    if let Some(export_answer) = export_answer {
        plist_document[EXPORT_ANSWER_KEY] = export_answer;
    }
    plist_identity_problems(&plist_document, &ChannelProfile::for_channel(channel))
        .contains(&MISSING_ANSWER_PROBLEM.to_owned())
}

/// The store plist carries the answer so an upload is not held for
/// questions. The direct channels are never uploaded, and the agent is a
/// helper inside the uploaded app, so neither carries it.
#[test]
fn should_declare_the_export_answer_in_the_store_application_plist_only() {
    let rendered_answer = format!("<key>{EXPORT_ANSWER_KEY}</key><false/>");
    for channel in [Channel::Development, Channel::Stable, Channel::AppStore] {
        let channel_profile = ChannelProfile::for_channel(channel);
        let application_plist_text = render_info_plist(
            &channel_profile,
            &sample_build_identity(),
            "https://example-owner.github.io/astronomical/appcast.xml",
        );
        let agent_plist_text = render_agent_info_plist(&channel_profile, &sample_build_identity());

        assert_eq!(
            application_plist_text.contains(&rendered_answer),
            channel == Channel::AppStore,
            "{channel:?}"
        );
        assert!(!agent_plist_text.contains(EXPORT_ANSWER_KEY), "{channel:?}");
    }
}

#[test]
fn should_accept_a_store_plist_that_declares_the_answer_as_false() {
    assert!(!reports_missing_answer(
        Some(serde_json::json!(false)),
        Channel::AppStore
    ));
}

#[test]
fn should_refuse_a_store_plist_with_a_missing_or_different_export_answer() {
    assert!(reports_missing_answer(None, Channel::AppStore));
    assert!(reports_missing_answer(
        Some(serde_json::json!(true)),
        Channel::AppStore
    ));
}

/// App Store Connect reads the key as a boolean; the word in a string is not
/// the declaration.
#[test]
fn should_refuse_a_store_plist_whose_export_answer_is_text() {
    assert!(reports_missing_answer(
        Some(serde_json::json!("false")),
        Channel::AppStore
    ));
}

#[test]
fn should_not_ask_a_direct_bundle_for_the_export_answer() {
    for channel in [Channel::Development, Channel::Stable] {
        assert!(!reports_missing_answer(None, channel), "{channel:?}");
    }
}
