use std::{collections::BTreeMap, fs, net::SocketAddr, path::Path, sync::Arc};

use axum::{Router, extract::State, http::StatusCode, response::IntoResponse};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

use super::{
    SkillUpdateError, SkillUpdateOutcome, app_is_at_least, is_skill_file_path,
    update_skill_from_site,
};
use crate::astronomical_skill::{
    bundled_skill_version, downloaded_skill_version, install_astronomical_skill, skill_index_path,
};

const APP_VERSION: &str = "0.2.123";

/// A stand-in for the project's site: serves whatever paths it was given.
async fn serve_site(site_files: BTreeMap<String, Vec<u8>>) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener");
    let site_address: SocketAddr = listener.local_addr().expect("address");
    let application = Router::new()
        .fallback(serve_site_file)
        .with_state(Arc::new(site_files));
    tokio::spawn(async move {
        let _ = axum::serve(listener, application).await;
    });
    format!("http://{site_address}/skill")
}

async fn serve_site_file(
    State(site_files): State<Arc<BTreeMap<String, Vec<u8>>>>,
    request_uri: axum::http::Uri,
) -> axum::response::Response {
    match site_files.get(request_uri.path()) {
        Some(file_bytes) => file_bytes.clone().into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

fn digest_of(file_text: &str) -> String {
    lowercase_hex::encode_lowercase_hex(&Sha256::digest(file_text.as_bytes()))
}

/// Site files for a skill of `skill_version` with one index and one reference.
fn published_skill(skill_version: u64, minimum_app_version: &str) -> BTreeMap<String, Vec<u8>> {
    let index_text = format!("# Skill version {skill_version}\n");
    let reference_text = "# Drawing, improved\n";
    let manifest = serde_json::json!({
        "skill_version": skill_version,
        "minimum_app_version": minimum_app_version,
        "files": {
            "SKILL.md": digest_of(&index_text),
            "references/images.md": digest_of(reference_text),
        },
    });
    BTreeMap::from([
        (
            "/skill/manifest.json".to_owned(),
            manifest.to_string().into_bytes(),
        ),
        (
            "/skill/astronomical/SKILL.md".to_owned(),
            index_text.into_bytes(),
        ),
        (
            "/skill/astronomical/references/images.md".to_owned(),
            reference_text.as_bytes().to_vec(),
        ),
    ])
}

fn installed_skill_directory() -> TempDir {
    let skill_directory = TempDir::new().expect("scratch folder");
    install_astronomical_skill(skill_directory.path()).expect("compiled-in skill");
    skill_directory
}

async fn update(
    site_base_url: &str,
    skill_directory: &Path,
) -> Result<SkillUpdateOutcome, SkillUpdateError> {
    let http_client = super::skill_site_http_client(false).expect("client");
    update_skill_from_site(&http_client, site_base_url, skill_directory, APP_VERSION).await
}

#[tokio::test]
async fn should_install_a_newer_skill_and_keep_it_across_a_restart() {
    let newer_version = bundled_skill_version() + 1;
    let site_base_url = serve_site(published_skill(newer_version, APP_VERSION)).await;
    let skill_directory = installed_skill_directory();

    let outcome = update(&site_base_url, skill_directory.path())
        .await
        .expect("update");
    assert_eq!(
        outcome,
        SkillUpdateOutcome::Updated {
            from_version: bundled_skill_version(),
            to_version: newer_version,
        }
    );
    let index_path = skill_index_path(skill_directory.path());
    let downloaded_index_text = format!("# Skill version {newer_version}\n");
    assert_eq!(
        fs::read_to_string(&index_path).expect("index"),
        downloaded_index_text
    );
    // The whole folder was replaced: references the new skill lacks are gone.
    assert!(
        !skill_directory
            .path()
            .join("astronomical/references/speech.md")
            .exists()
    );
    assert!(
        !skill_directory
            .path()
            .join("astronomical.incoming")
            .exists()
    );
    assert!(
        !skill_directory
            .path()
            .join("astronomical.outgoing")
            .exists()
    );

    install_astronomical_skill(skill_directory.path()).expect("restart");
    assert_eq!(
        fs::read_to_string(&index_path).expect("index"),
        downloaded_index_text
    );
    let second_look = update(&site_base_url, skill_directory.path())
        .await
        .expect("second look");
    assert_eq!(
        second_look,
        SkillUpdateOutcome::AlreadyCurrent {
            installed_version: newer_version
        }
    );
}

#[tokio::test]
async fn should_change_nothing_when_the_site_has_the_same_or_an_older_skill() {
    let site_base_url = serve_site(published_skill(bundled_skill_version(), APP_VERSION)).await;
    let skill_directory = installed_skill_directory();
    let outcome = update(&site_base_url, skill_directory.path())
        .await
        .expect("look");
    assert_eq!(
        outcome,
        SkillUpdateOutcome::AlreadyCurrent {
            installed_version: bundled_skill_version()
        }
    );
    assert_eq!(downloaded_skill_version(skill_directory.path()), None);
}

#[tokio::test]
async fn should_keep_its_own_skill_when_the_published_one_needs_a_newer_app() {
    let site_base_url = serve_site(published_skill(bundled_skill_version() + 1, "0.3.0")).await;
    let skill_directory = installed_skill_directory();
    let outcome = update(&site_base_url, skill_directory.path())
        .await
        .expect("look");
    assert_eq!(
        outcome,
        SkillUpdateOutcome::NeedsNewerApp {
            minimum_app_version: "0.3.0".to_owned()
        }
    );
    assert!(
        fs::read_to_string(skill_index_path(skill_directory.path()))
            .expect("index")
            .starts_with("---\nname: astronomical\n")
    );
}

#[tokio::test]
async fn should_refuse_a_file_that_does_not_match_its_digest_and_keep_the_old_skill() {
    let mut site_files = published_skill(bundled_skill_version() + 1, APP_VERSION);
    site_files.insert(
        "/skill/astronomical/references/images.md".to_owned(),
        b"cut off mid-download".to_vec(),
    );
    let site_base_url = serve_site(site_files).await;
    let skill_directory = installed_skill_directory();

    let refusal = update(&site_base_url, skill_directory.path())
        .await
        .expect_err("digest mismatch");
    assert!(
        matches!(refusal, SkillUpdateError::DigestMismatch { .. }),
        "{refusal}"
    );
    assert!(
        skill_directory
            .path()
            .join("astronomical/references/speech.md")
            .exists()
    );
    assert!(
        !skill_directory
            .path()
            .join("astronomical.incoming")
            .exists()
    );
    assert_eq!(downloaded_skill_version(skill_directory.path()), None);
}

#[tokio::test]
async fn should_refuse_a_manifest_that_names_a_path_outside_the_skill() {
    let manifest = serde_json::json!({
        "skill_version": bundled_skill_version() + 1,
        "minimum_app_version": APP_VERSION,
        "files": { "SKILL.md": digest_of("x"), "../../config.json": digest_of("y") },
    });
    let site_files = BTreeMap::from([(
        "/skill/manifest.json".to_owned(),
        manifest.to_string().into_bytes(),
    )]);
    let site_base_url = serve_site(site_files).await;
    let skill_directory = installed_skill_directory();
    let refusal = update(&site_base_url, skill_directory.path())
        .await
        .expect_err("bad path");
    assert!(
        matches!(refusal, SkillUpdateError::UnusableManifest(_)),
        "{refusal}"
    );
}

#[tokio::test]
async fn should_report_a_site_without_a_manifest() {
    let site_base_url = serve_site(BTreeMap::new()).await;
    let skill_directory = installed_skill_directory();
    let refusal = update(&site_base_url, skill_directory.path())
        .await
        .expect_err("no manifest");
    assert!(
        matches!(refusal, SkillUpdateError::Refused { status: 404, .. }),
        "{refusal}"
    );
}

#[test]
fn should_accept_only_the_index_and_markdown_files_directly_under_references() {
    assert!(is_skill_file_path("SKILL.md"));
    assert!(is_skill_file_path("references/models-and-memory.md"));
    for refused_path in [
        "../SKILL.md",
        "references/../../config.json",
        "references/nested/topic.md",
        "references/.md",
        "references/Topic.md",
        "references/script.sh",
        "/etc/passwd",
        "",
    ] {
        assert!(
            !is_skill_file_path(refused_path),
            "{refused_path} must be refused"
        );
    }
}

#[test]
fn should_compare_app_versions_number_by_number() {
    assert!(app_is_at_least("0.2.123", "0.2.123"));
    assert!(app_is_at_least("0.2.130", "0.2.123"));
    assert!(app_is_at_least("0.10.0", "0.9.9"));
    assert!(!app_is_at_least("0.2.99", "0.2.123"));
    assert!(!app_is_at_least("0.2.123", "not-a-version"));
    assert!(!app_is_at_least("0.2.x", "0.2.1"));
}
