use crate::version_only_bump::{VersionBumpTexts, is_version_only_bump, workspace_package_version};

const BASE_MANIFEST: &str = "[workspace]\nmembers = [\"apps/daemon\"]\n\n[workspace.package]\nedition = \"2024\"\nversion = \"0.2.88\"\n\n[workspace.dependencies]\nserde = { version = \"1\" }\n";

const BASE_LOCKFILE: &str = "# generated\nversion = 4\n\n[[package]]\nname = \"astronomical-daemon\"\nversion = \"0.2.88\"\ndependencies = [\n \"serde\",\n]\n\n[[package]]\nname = \"serde\"\nversion = \"1.0.200\"\nsource = \"registry+https://github.com/rust-lang/crates.io-index\"\nchecksum = \"abc\"\n";

fn bumped(text: &str) -> String {
    text.replace("version = \"0.2.88\"", "version = \"0.2.89\"")
}

fn owned_paths(changed_paths: &[&str]) -> Vec<String> {
    let mut owned = Vec::new();
    for changed_path in changed_paths {
        owned.push((*changed_path).to_owned());
    }
    owned
}

fn version_only(changed_paths: &[&str], head_manifest: &str, head_lockfile: &str) -> bool {
    is_version_only_bump(
        &owned_paths(changed_paths),
        &VersionBumpTexts {
            base_manifest: BASE_MANIFEST,
            head_manifest,
            base_lockfile: BASE_LOCKFILE,
            head_lockfile,
        },
    )
}

#[test]
fn should_recognise_a_pure_workspace_version_bump() {
    assert!(version_only(
        &[
            "Cargo.toml",
            "Cargo.lock",
            "third-party/RUST_DEPENDENCY_NOTICES.digest"
        ],
        &bumped(BASE_MANIFEST),
        &bumped(BASE_LOCKFILE),
    ));
}

#[test]
fn should_recognise_a_bump_that_leaves_the_notices_digest_alone() {
    assert!(version_only(
        &["Cargo.toml", "Cargo.lock"],
        &bumped(BASE_MANIFEST),
        &bumped(BASE_LOCKFILE),
    ));
}

#[test]
fn should_refuse_a_bump_that_also_changes_any_other_file() {
    for extra_path in [
        "apps/daemon/src/main.rs",
        "third-party/RUST_DEPENDENCY_NOTICES",
        "README.md",
    ] {
        assert!(
            !version_only(
                &["Cargo.toml", "Cargo.lock", extra_path],
                &bumped(BASE_MANIFEST),
                &bumped(BASE_LOCKFILE),
            ),
            "{extra_path}"
        );
    }
}

#[test]
fn should_refuse_a_lockfile_that_also_moves_a_third_party_crate() {
    let head_lockfile = bumped(BASE_LOCKFILE).replace("1.0.200", "1.0.201");

    assert!(!version_only(
        &["Cargo.toml", "Cargo.lock"],
        &bumped(BASE_MANIFEST),
        &head_lockfile,
    ));
}

#[test]
fn should_refuse_a_manifest_that_also_changes_a_dependency() {
    let head_manifest =
        bumped(BASE_MANIFEST).replace("serde = { version = \"1\" }", "serde = { version = \"2\" }");

    assert!(!version_only(
        &["Cargo.toml", "Cargo.lock"],
        &head_manifest,
        &bumped(BASE_LOCKFILE),
    ));
}

#[test]
fn should_refuse_a_lockfile_that_gains_or_loses_a_package() {
    let head_lockfile = format!(
        "{}\n[[package]]\nname = \"added\"\nversion = \"0.1.0\"\nsource = \"registry+x\"\n",
        bumped(BASE_LOCKFILE)
    );

    assert!(!version_only(
        &["Cargo.toml", "Cargo.lock"],
        &bumped(BASE_MANIFEST),
        &head_lockfile,
    ));
}

#[test]
fn should_refuse_when_the_version_did_not_change_or_only_one_file_changed() {
    assert!(!version_only(
        &["Cargo.toml", "Cargo.lock"],
        BASE_MANIFEST,
        BASE_LOCKFILE
    ));
    assert!(!version_only(
        &["Cargo.toml"],
        &bumped(BASE_MANIFEST),
        BASE_LOCKFILE
    ));
    assert!(!version_only(
        &[],
        &bumped(BASE_MANIFEST),
        &bumped(BASE_LOCKFILE)
    ));
}

#[test]
fn should_read_the_version_only_from_the_workspace_package_table() {
    assert_eq!(workspace_package_version(BASE_MANIFEST), Some("0.2.88"));
    assert_eq!(
        workspace_package_version("[package]\nversion = \"9.9.9\"\n"),
        None
    );
    assert_eq!(workspace_package_version(""), None);
}
