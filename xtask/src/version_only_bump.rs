//! Proves from file contents that a change is only a workspace version bump.
//!
//! A version bump touches `Cargo.toml` and `Cargo.lock`, which by path alone
//! look like a dependency change and send a pull request through every macOS
//! lane: about ten minutes to re-prove code that is identical to the commit
//! main already verified, apart from the `CARGO_PKG_VERSION` value. Paths
//! cannot tell a bump from a dependency edit, so this module compares the
//! base and head texts: the manifest may differ only in the
//! `[workspace.package]` version, the lockfile only in the version lines of
//! workspace packages, and the notices only in their digest memo. Anything
//! else, or anything it cannot parse, is not a version-only bump.

const WORKSPACE_MANIFEST_PATH: &str = "Cargo.toml";
const LOCKFILE_PATH: &str = "Cargo.lock";
/// The digest memo records `Cargo.toml` among its inputs, so a bump moves it
/// while the notices text, which lists third-party crates only, stays put.
const NOTICES_DIGEST_PATH: &str = "third-party/RUST_DEPENDENCY_NOTICES.digest";
const WORKSPACE_PACKAGE_HEADER: &str = "[workspace.package]";
const LOCKFILE_PACKAGE_SEPARATOR: &str = "\n[[package]]\n";

/// Base and head texts of the two files a version bump rewrites.
pub struct VersionBumpTexts<'text> {
    pub base_manifest: &'text str,
    pub head_manifest: &'text str,
    pub base_lockfile: &'text str,
    pub head_lockfile: &'text str,
}

/// True only when the changed paths are the manifest, the lockfile, and at
/// most the notices digest, and the texts prove a pure version change.
pub fn is_version_only_bump(changed_paths: &[String], bump_texts: &VersionBumpTexts<'_>) -> bool {
    if !changes_only_version_bump_paths(changed_paths) {
        return false;
    }
    let Some(old_version) = workspace_package_version(bump_texts.base_manifest) else {
        return false;
    };
    let Some(new_version) = workspace_package_version(bump_texts.head_manifest) else {
        return false;
    };
    old_version != new_version
        && manifest_differs_only_in_workspace_version(
            bump_texts.base_manifest,
            bump_texts.head_manifest,
            old_version,
            new_version,
        )
        && lockfile_differs_only_in_workspace_versions(
            bump_texts.base_lockfile,
            bump_texts.head_lockfile,
            old_version,
            new_version,
        )
}

/// Both version sources must change, and nothing but the digest memo may
/// change beside them; the notices text itself is a third-party artifact.
fn changes_only_version_bump_paths(changed_paths: &[String]) -> bool {
    let mut manifest_changed = false;
    let mut lockfile_changed = false;
    for changed_path in changed_paths {
        match changed_path.as_str() {
            WORKSPACE_MANIFEST_PATH => manifest_changed = true,
            LOCKFILE_PATH => lockfile_changed = true,
            NOTICES_DIGEST_PATH => {}
            _ => return false,
        }
    }
    manifest_changed && lockfile_changed
}

/// The `version = "…"` value of the manifest's `[workspace.package]` table.
pub fn workspace_package_version(manifest_text: &str) -> Option<&str> {
    let mut inside_workspace_package = false;
    for manifest_line in manifest_text.lines() {
        let trimmed_line = manifest_line.trim();
        if trimmed_line.starts_with('[') {
            inside_workspace_package = trimmed_line == WORKSPACE_PACKAGE_HEADER;
            continue;
        }
        if inside_workspace_package
            && let Some(quoted_version) = trimmed_line.strip_prefix("version = \"")
        {
            return quoted_version.strip_suffix('"');
        }
    }
    None
}

/// The manifest equals the base once its workspace version is replaced.
fn manifest_differs_only_in_workspace_version(
    base_manifest: &str,
    head_manifest: &str,
    old_version: &str,
    new_version: &str,
) -> bool {
    let old_version_line = format!("version = \"{old_version}\"");
    let new_version_line = format!("version = \"{new_version}\"");
    if base_manifest.matches(&old_version_line).count() != 1 {
        return false;
    }
    base_manifest.replacen(&old_version_line, &new_version_line, 1) == head_manifest
}

/// Every lockfile package matches its base counterpart, except workspace
/// packages (no `source`, so path members), whose version may move from the
/// old workspace version to the new one and nothing else.
fn lockfile_differs_only_in_workspace_versions(
    base_lockfile: &str,
    head_lockfile: &str,
    old_version: &str,
    new_version: &str,
) -> bool {
    let base_packages: Vec<&str> = base_lockfile.split(LOCKFILE_PACKAGE_SEPARATOR).collect();
    let head_packages: Vec<&str> = head_lockfile.split(LOCKFILE_PACKAGE_SEPARATOR).collect();
    if base_packages.len() != head_packages.len() {
        return false;
    }
    let old_version_line = format!("\nversion = \"{old_version}\"");
    let new_version_line = format!("\nversion = \"{new_version}\"");
    for (base_package, head_package) in base_packages.iter().zip(head_packages.iter()) {
        if base_package == head_package {
            continue;
        }
        let is_workspace_package = !base_package.contains("\nsource = ");
        let bumped_base_package = base_package.replacen(&old_version_line, &new_version_line, 1);
        if !is_workspace_package || bumped_base_package != *head_package {
            return false;
        }
    }
    true
}
