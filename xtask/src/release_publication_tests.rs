//! Publisher decisions, tested as pure functions over manifests, appcast
//! text, and `gh` JSON. These cover the retired shell suite's journeys:
//! resuming an interrupted draft, repairing a mismatched upload, refusing a
//! prerelease, matching the remote tag, and never touching GitHub while
//! preparing.

use std::path::{Path, PathBuf};

use super::command::{ReleaseCommand, parse_release_command};
use super::github_publication::{
    DraftAssetAction, TagObject, draft_asset_action, next_tag_object, release_state_problems,
    uploaded_asset_problems,
};
use super::publication_entry::resolve_credential_map;
use super::release_identity::{
    ReleaseIdentity, check_output_directory, parse_release_version, repository_slug,
};
use super::release_manifest::{ArtifactRecord, ReleaseManifest, record_artifact, verify_artifact};
use super::release_preparation::{appcast_problems, release_download_prefix};

fn arguments(argument_words: &[&str]) -> Vec<String> {
    let mut owned_arguments = Vec::new();
    for argument_word in argument_words {
        owned_arguments.push((*argument_word).to_owned());
    }
    owned_arguments
}

fn release_identity() -> ReleaseIdentity {
    ReleaseIdentity {
        release_tag: "v0.2.88".to_owned(),
        release_version: "0.2.88".to_owned(),
        release_commit: "0123456789abcdef".to_owned(),
        repository_slug: "example-owner/astronomical".to_owned(),
        update_feed_url: "https://example-owner.github.io/astronomical/appcast.xml".to_owned(),
        output_directory: PathBuf::from("/target/releases/v0.2.88"),
    }
}

fn artifact(artifact_name: &str, sha256: &str, size_bytes: u64) -> ArtifactRecord {
    ArtifactRecord {
        artifact_name: artifact_name.to_owned(),
        sha256: sha256.to_owned(),
        size_bytes,
    }
}

fn release_manifest() -> ReleaseManifest {
    ReleaseManifest {
        release_tag: "v0.2.88".to_owned(),
        release_commit: "0123456789abcdef".to_owned(),
        release_version: "0.2.88".to_owned(),
        disk_image: artifact("Astronomical-0.2.88-macOS-arm64.dmg", "aaaa", 1024),
        release_notes: artifact("Astronomical-0.2.88-macOS-arm64.md", "bbbb", 64),
        appcast: artifact("appcast.xml", "cccc", 512),
    }
}

fn uploaded_asset(digest: &str, size_bytes: u64, upload_state: &str) -> serde_json::Value {
    serde_json::json!({"name": "Astronomical-0.2.88-macOS-arm64.dmg", "digest": digest, "size": size_bytes, "state": upload_state})
}

#[test]
fn should_accept_only_numeric_semantic_version_tags() {
    assert_eq!(parse_release_version("v0.2.88"), Ok("0.2.88".to_owned()));
    for rejected_tag in [
        "0.2.88",
        "v0.2",
        "v0.2.88.1",
        "v0.2.x",
        "v.2.88",
        "v0..88",
        "",
    ] {
        assert!(
            parse_release_version(rejected_tag).is_err(),
            "{rejected_tag}"
        );
    }
}

#[test]
fn should_derive_the_repository_slug_from_the_workspace_url() {
    assert_eq!(
        repository_slug("https://github.com/example-owner/astronomical.git"),
        Ok("example-owner/astronomical".to_owned())
    );
    assert!(repository_slug("git@github.com:example-owner/astronomical.git").is_err());
}

#[test]
fn should_never_overwrite_a_prepared_directory_and_only_publish_an_existing_one() {
    let fixture_directory = tempfile::tempdir().expect("a fixture directory");
    let absent_directory = fixture_directory.path().join("v0.2.88");

    assert_eq!(check_output_directory(&absent_directory, false), Ok(()));
    assert!(
        check_output_directory(fixture_directory.path(), false)
            .expect_err("exists")
            .contains("already exists")
    );
    assert!(
        check_output_directory(&absent_directory, true)
            .expect_err("absent")
            .contains("unavailable")
    );
    assert_eq!(
        check_output_directory(fixture_directory.path(), true),
        Ok(())
    );
}

#[test]
fn should_round_trip_the_release_manifest_and_refuse_an_incomplete_one() {
    let manifest_text = release_manifest().to_json_text();

    assert_eq!(
        ReleaseManifest::from_json_text(&manifest_text),
        Ok(release_manifest())
    );
    assert!(ReleaseManifest::from_json_text("{}").is_err());
    assert!(ReleaseManifest::from_json_text("not json").is_err());
}

#[test]
fn should_verify_an_artifact_by_digest_and_size_and_name_what_drifted() {
    let fixture_directory = tempfile::tempdir().expect("a fixture directory");
    let disk_image = fixture_directory.path().join("Astronomical.dmg");
    std::fs::write(&disk_image, "reviewed bytes").expect("fixture DMG");
    let reviewed_record = record_artifact(&disk_image).expect("recordable");

    assert_eq!(
        verify_artifact(&disk_image, &reviewed_record, "DMG"),
        Ok(())
    );
    std::fs::write(&disk_image, "tampered bytes").expect("tamper");
    assert!(
        verify_artifact(&disk_image, &reviewed_record, "DMG")
            .expect_err("tampered")
            .contains("DMG digest")
    );
    assert!(
        verify_artifact(Path::new("/absent.dmg"), &reviewed_record, "DMG")
            .expect_err("missing")
            .contains("incomplete")
    );
}

#[test]
fn should_require_signed_archive_signed_feed_and_exact_release_url_in_the_appcast() {
    let download_prefix = release_download_prefix(&release_identity());
    let signed_appcast = format!(
        "<!-- sparkle-signatures:\nedSignature: x -->\n<enclosure url=\"{download_prefix}Astronomical-0.2.88-macOS-arm64.dmg\" sparkle:edSignature=\"y\"/>"
    );

    assert_eq!(
        download_prefix,
        "https://github.com/example-owner/astronomical/releases/download/v0.2.88/"
    );
    assert!(
        appcast_problems(
            &signed_appcast,
            &download_prefix,
            "Astronomical-0.2.88-macOS-arm64.dmg"
        )
        .is_empty()
    );
    assert_eq!(
        appcast_problems(
            "<rss/>",
            &download_prefix,
            "Astronomical-0.2.88-macOS-arm64.dmg"
        ),
        vec![
            "appcast archive signature is missing",
            "signed appcast envelope is missing",
            "appcast release URL is incorrect"
        ]
    );
}

#[test]
fn should_follow_annotated_tags_to_the_commit() {
    let annotated = serde_json::json!({"object": {"type": "tag", "sha": "tag-sha"}});
    let lightweight = serde_json::json!({"object": {"type": "commit", "sha": "0123456789abcdef"}});

    assert_eq!(
        next_tag_object(&annotated),
        Ok(TagObject::AnnotatedTag("tag-sha".to_owned()))
    );
    assert_eq!(
        next_tag_object(&lightweight),
        Ok(TagObject::Commit("0123456789abcdef".to_owned()))
    );
    assert!(next_tag_object(&serde_json::json!({"object": {"type": "tree", "sha": "x"}})).is_err());
}

#[test]
fn should_upload_into_an_empty_draft_and_keep_a_verified_upload() {
    assert_eq!(
        draft_asset_action(&serde_json::json!({"assets": []}), &release_manifest()),
        Ok(DraftAssetAction::Upload)
    );
    assert_eq!(
        draft_asset_action(
            &serde_json::json!({"assets": [uploaded_asset("sha256:aaaa", 1024, "uploaded")]}),
            &release_manifest()
        ),
        Ok(DraftAssetAction::KeepVerifiedUpload)
    );
}

/// The retired suite's digest-mismatch journey: a draft whose upload has the
/// wrong bytes or never finished is re-uploaded before publication.
#[test]
fn should_replace_a_mismatched_or_incomplete_draft_upload() {
    for mismatched_asset in [
        uploaded_asset("sha256:ffff", 1024, "uploaded"),
        uploaded_asset("sha256:aaaa", 1024, "starter"),
    ] {
        assert_eq!(
            draft_asset_action(
                &serde_json::json!({"assets": [mismatched_asset]}),
                &release_manifest()
            ),
            Ok(DraftAssetAction::ReplaceMismatchedUpload)
        );
    }
}

#[test]
fn should_refuse_a_draft_with_unexpected_assets() {
    let unexpected_assets = serde_json::json!({"assets": [
        uploaded_asset("sha256:aaaa", 1024, "uploaded"),
        {"name": "extra.zip", "digest": "sha256:0", "size": 1, "state": "uploaded"}
    ]});

    assert!(draft_asset_action(&unexpected_assets, &release_manifest()).is_err());
    assert!(
        uploaded_asset_problems(&unexpected_assets, &release_manifest())
            .contains(&"GitHub Release contains unexpected assets".to_owned())
    );
}

#[test]
fn should_publish_only_an_asset_with_the_reviewed_digest_size_and_state() {
    let verified_state =
        serde_json::json!({"assets": [uploaded_asset("sha256:aaaa", 1024, "uploaded")]});
    let drifted_state =
        serde_json::json!({"assets": [uploaded_asset("sha256:ffff", 999, "starter")]});

    assert!(uploaded_asset_problems(&verified_state, &release_manifest()).is_empty());
    assert_eq!(
        uploaded_asset_problems(&drifted_state, &release_manifest()),
        vec![
            "GitHub Release asset digest does not match prepared DMG",
            "GitHub Release asset size does not match prepared DMG",
            "GitHub Release asset upload is incomplete",
        ]
    );
    assert_eq!(
        uploaded_asset_problems(&serde_json::json!({"assets": []}), &release_manifest()),
        vec!["GitHub Release must contain exactly one expected DMG"]
    );
}

#[test]
fn should_resume_only_a_release_that_is_this_release_and_never_a_prerelease() {
    let matching_state = serde_json::json!({
        "tagName": "v0.2.88", "name": "Astronomical 0.2.88", "body": "Notes\n", "isPrerelease": false
    });
    let prerelease_state = serde_json::json!({
        "tagName": "v0.2.88", "name": "Astronomical 0.2.88", "body": "Notes", "isPrerelease": true
    });
    let conflicting_state = serde_json::json!({
        "tagName": "v0.2.87", "name": "Astronomical 0.2.87", "body": "Other", "isPrerelease": false
    });

    assert!(release_state_problems(&matching_state, &release_identity(), "Notes").is_empty());
    assert_eq!(
        release_state_problems(&prerelease_state, &release_identity(), "Notes"),
        vec!["Stable publication cannot use a prerelease GitHub Release"]
    );
    assert_eq!(
        release_state_problems(&conflicting_state, &release_identity(), "Notes").len(),
        3
    );
}

#[test]
fn should_resolve_exactly_one_developer_id_identity_and_its_team() {
    let one_identity = "  1) ABC \"Developer ID Application: Example (ABCDE12345)\"\n  2) DEF \"Apple Development: Example (ZZZZZ99999)\"\n";
    let two_identities = "  1) ABC \"Developer ID Application: Example (ABCDE12345)\"\n  2) DEF \"Developer ID Application: Other (FGHIJ67890)\"\n";

    assert_eq!(
        resolve_credential_map(one_identity),
        Ok((
            "Developer ID Application: Example (ABCDE12345)".to_owned(),
            "ABCDE12345".to_owned()
        ))
    );
    assert!(
        resolve_credential_map(two_identities)
            .expect_err("two")
            .contains("exactly one")
    );
    assert!(
        resolve_credential_map("  0 valid identities found\n")
            .expect_err("none")
            .contains("exactly one")
    );
    assert!(
        resolve_credential_map("  1) ABC \"Developer ID Application: Example (SHORT)\"\n")
            .expect_err("team")
            .contains("Team ID")
    );
}

#[test]
fn should_parse_prepare_and_publish_requests() {
    let prepare_command = parse_release_command(&arguments(&[
        "prepare-and-publish",
        "--tag",
        "v0.2.88",
        "--notes-file",
        "/notes.md",
        "--signing-identity",
        "Dev",
        "--team-id",
        "ABCDE12345",
        "--notary-profile",
        "Astronomical Notarization",
    ]));
    let publish_command = parse_release_command(&arguments(&[
        "prepare-signed",
        "--tag",
        "v0.2.88",
        "--publish",
        "--output-directory",
        "/out",
    ]));

    let Ok(ReleaseCommand::PrepareAndPublish(prepare_request)) = prepare_command else {
        panic!("prepare-and-publish should parse");
    };
    assert!(!prepare_request.is_publish);
    assert_eq!(
        prepare_request.team_identifier.as_deref(),
        Some("ABCDE12345")
    );
    let Ok(ReleaseCommand::PrepareSigned(publish_request)) = publish_command else {
        panic!("prepare-signed should parse");
    };
    assert!(publish_request.is_publish);
    assert_eq!(
        publish_request.output_directory,
        Some(PathBuf::from("/out"))
    );
    assert!(
        parse_release_command(&arguments(&[
            "prepare-signed",
            "--tag",
            "v0.2.88",
            "--signing-identity",
            "Dev"
        ]))
        .is_err()
    );
    assert!(
        parse_release_command(&arguments(&[
            "prepare-and-publish",
            "--notes-file",
            "/notes.md"
        ]))
        .is_err()
    );
}
