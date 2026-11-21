use std::{fs, path::Path};

use tempfile::TempDir;

use crate::skill_site::run;

fn repository_with_skill() -> TempDir {
    let repository = TempDir::new().expect("scratch repository");
    let references_directory = repository.path().join("skill/astronomical/references");
    fs::create_dir_all(&references_directory).expect("skill folder");
    fs::write(
        repository.path().join("skill/astronomical/SKILL.md"),
        "# Astronomical\n",
    )
    .expect("index");
    fs::write(references_directory.join("images.md"), "# Drawing\n").expect("reference");
    fs::write(
        repository.path().join("skill/release.json"),
        r#"{"skill_version": 3, "minimum_app_version": "0.2.123"}"#,
    )
    .expect("release file");
    repository
}

fn check_arguments() -> Vec<String> {
    vec!["--check".to_owned()]
}

fn site_manifest(repository_root: &Path) -> serde_json::Value {
    let manifest_text =
        fs::read_to_string(repository_root.join("site/skill/manifest.json")).expect("manifest");
    serde_json::from_str(&manifest_text).expect("manifest is JSON")
}

#[test]
fn should_publish_the_files_and_a_manifest_with_their_digests() {
    let repository = repository_with_skill();
    run(repository.path(), &[]).expect("publish");
    assert_eq!(
        fs::read_to_string(
            repository
                .path()
                .join("site/skill/astronomical/references/images.md")
        )
        .expect("published reference"),
        "# Drawing\n"
    );
    let manifest = site_manifest(repository.path());
    assert_eq!(manifest["skill_version"], 3);
    assert_eq!(manifest["minimum_app_version"], "0.2.123");
    assert_eq!(
        manifest["files"]["SKILL.md"],
        sha256_hex("# Astronomical\n")
    );
    assert!(manifest["files"]["references/images.md"].is_string());
    run(repository.path(), &check_arguments()).expect("a fresh publish is current");
}

fn sha256_hex(text: &str) -> String {
    use sha2::{Digest, Sha256};
    lowercase_hex::encode_lowercase_hex(&Sha256::digest(text.as_bytes()))
}

#[test]
fn should_fail_the_check_when_the_skill_changed_after_publishing() {
    let repository = repository_with_skill();
    run(repository.path(), &[]).expect("publish");
    fs::write(
        repository.path().join("skill/astronomical/SKILL.md"),
        "# Astronomical, edited\n",
    )
    .expect("edit the skill");
    let refusal = run(repository.path(), &check_arguments()).expect_err("out of date");
    assert!(
        refusal.contains("differs: site/skill/astronomical/SKILL.md"),
        "{refusal}"
    );
    assert!(
        refusal.contains("differs: site/skill/manifest.json"),
        "{refusal}"
    );
    assert!(refusal.contains("cargo xtask skill-site"), "{refusal}");
}

#[test]
fn should_remove_a_published_reference_the_skill_no_longer_has() {
    let repository = repository_with_skill();
    run(repository.path(), &[]).expect("publish");
    fs::remove_file(
        repository
            .path()
            .join("skill/astronomical/references/images.md"),
    )
    .expect("retire the reference");
    let refusal = run(repository.path(), &check_arguments()).expect_err("stale copy");
    assert!(
        refusal.contains("no longer in the skill: site/skill/astronomical/references/images.md")
    );
    run(repository.path(), &[]).expect("publish again");
    assert!(
        !repository
            .path()
            .join("site/skill/astronomical/references/images.md")
            .exists()
    );
    assert!(
        site_manifest(repository.path())["files"]
            .get("references/images.md")
            .is_none()
    );
}

#[test]
fn should_fail_the_check_before_anything_was_published() {
    let repository = repository_with_skill();
    assert!(run(repository.path(), &check_arguments()).is_err());
    assert!(!repository.path().join("site/skill").exists());
}

#[test]
fn should_refuse_a_release_file_without_a_version() {
    let repository = repository_with_skill();
    fs::write(
        repository.path().join("skill/release.json"),
        r#"{"minimum_app_version": "0.2.123"}"#,
    )
    .expect("release file");
    let refusal = run(repository.path(), &[]).expect_err("no version");
    assert!(refusal.contains("skill_version"), "{refusal}");
}

#[test]
fn should_refuse_an_unknown_argument() {
    let repository = repository_with_skill();
    let refusal = run(repository.path(), &["--force".to_owned()]).expect_err("unknown flag");
    assert!(refusal.starts_with("usage: cargo xtask skill-site"));
}
