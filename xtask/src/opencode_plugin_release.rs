//! `cargo xtask opencode-plugin-release-check TAG` — the gate the npm publish
//! workflow runs before it publishes the OpenCode plugin.
//!
//! A pushed tag cannot be moved without confusing everyone who saw it, and an
//! npm version can never be published twice, so the check refuses a release
//! whose three records disagree: the tag's version, the `version` in the
//! plugin's `package.json`, and a heading for that version in its
//! `CHANGELOG.md`. The plugin's tags carry their own prefix so they never look
//! like an app release tag (`v0.2.118`), which the release tooling and the
//! website's "latest release" links are built around.

use std::path::Path;

/// Every plugin release tag starts with this, followed by the npm version.
pub const OPENCODE_PLUGIN_TAG_PREFIX: &str = "opencode-plugin-v";
const PACKAGE_MANIFEST_RELATIVE_PATH: &str = "integrations/opencode/package.json";
const CHANGELOG_RELATIVE_PATH: &str = "integrations/opencode/CHANGELOG.md";

/// The version a plugin release tag names, or `None` for any other tag.
#[must_use]
pub fn version_from_tag(tag: &str) -> Option<&str> {
    let version = tag.strip_prefix(OPENCODE_PLUGIN_TAG_PREFIX)?;
    if is_plain_release_version(version) {
        Some(version)
    } else {
        None
    }
}

/// `MAJOR.MINOR.PATCH` with digits only: npm also accepts pre-release and build
/// suffixes, but the plugin publishes plain versions to the `latest` tag, so a
/// suffix is more likely a typo than a plan.
fn is_plain_release_version(version: &str) -> bool {
    let version_parts: Vec<&str> = version.split('.').collect();
    if version_parts.len() != 3 {
        return false;
    }
    for version_part in version_parts {
        if version_part.is_empty() {
            return false;
        }
        for version_byte in version_part.bytes() {
            if !version_byte.is_ascii_digit() {
                return false;
            }
        }
    }
    true
}

/// True when `changelog_text` has a `## VERSION` heading, alone or followed by
/// a space and a date.
#[must_use]
pub fn changelog_has_version_heading(changelog_text: &str, version: &str) -> bool {
    let heading = format!("## {version}");
    for changelog_line in changelog_text.lines() {
        if changelog_line == heading || changelog_line.starts_with(&format!("{heading} ")) {
            return true;
        }
    }
    false
}

/// Every reason `tag` cannot be published, empty when it can.
#[must_use]
pub fn release_problems(
    tag: &str,
    package_manifest_text: &str,
    changelog_text: &str,
) -> Vec<String> {
    let Some(tag_version) = version_from_tag(tag) else {
        return vec![format!(
            "tag {tag:?} is not {OPENCODE_PLUGIN_TAG_PREFIX}MAJOR.MINOR.PATCH"
        )];
    };
    let mut problems = Vec::new();
    match package_manifest_version(package_manifest_text) {
        Some(manifest_version) if manifest_version == tag_version => {}
        Some(manifest_version) => problems.push(format!(
            "tag names {tag_version} but {PACKAGE_MANIFEST_RELATIVE_PATH} says {manifest_version}"
        )),
        None => problems.push(format!(
            "{PACKAGE_MANIFEST_RELATIVE_PATH} has no string \"version\""
        )),
    }
    if !changelog_has_version_heading(changelog_text, tag_version) {
        problems.push(format!(
            "{CHANGELOG_RELATIVE_PATH} has no \"## {tag_version}\" heading"
        ));
    }
    problems
}

fn package_manifest_version(package_manifest_text: &str) -> Option<String> {
    let package_manifest: serde_json::Value = serde_json::from_str(package_manifest_text).ok()?;
    package_manifest.get("version")?.as_str().map(str::to_owned)
}

fn read_repository_text(repository_root: &Path, relative_path: &str) -> Result<String, String> {
    std::fs::read_to_string(repository_root.join(relative_path))
        .map_err(|failure| format!("{relative_path} could not be read: {failure}"))
}

/// What the command was asked to check.
#[derive(Debug, PartialEq, Eq)]
pub struct ReleaseCheckRequest {
    pub tag: String,
    /// Also refuse a tagged commit that is not on `origin/main`. The publish
    /// workflow always asks for this: without it, a tag pushed on an unmerged
    /// branch would publish code no pull request or CI run ever checked.
    pub requires_commit_on_main: bool,
}

const REQUIRE_ON_MAIN_FLAG: &str = "--require-on-main";
const USAGE: &str = "usage: cargo xtask opencode-plugin-release-check [--require-on-main] TAG";

/// Reads `[--require-on-main] TAG`; anything else is a usage error.
pub fn parse_release_check_arguments(arguments: &[String]) -> Result<ReleaseCheckRequest, String> {
    let mut requires_commit_on_main = false;
    let mut tags = Vec::new();
    for argument in arguments {
        if argument == REQUIRE_ON_MAIN_FLAG {
            requires_commit_on_main = true;
        } else if argument.starts_with('-') {
            return Err(format!("unknown option {argument}; {USAGE}"));
        } else {
            tags.push(argument.clone());
        }
    }
    let [tag] = tags.as_slice() else {
        return Err(USAGE.to_owned());
    };
    Ok(ReleaseCheckRequest {
        tag: tag.clone(),
        requires_commit_on_main,
    })
}

/// True when the checked-out commit is `origin/main` or one of its ancestors.
fn is_checked_out_commit_on_main(repository_root: &Path) -> Result<bool, String> {
    let merge_base_status = std::process::Command::new("git")
        .current_dir(repository_root)
        .args(["merge-base", "--is-ancestor", "HEAD", "origin/main"])
        .status()
        .map_err(|failure| format!("git could not run: {failure}"))?;
    match merge_base_status.code() {
        Some(0) => Ok(true),
        Some(1) => Ok(false),
        _ => Err(
            "git could not compare HEAD with origin/main; the check needs the full history"
                .to_owned(),
        ),
    }
}

pub fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    let request = parse_release_check_arguments(arguments)?;
    let tag = &request.tag;
    let mut problems = release_problems(
        tag,
        &read_repository_text(repository_root, PACKAGE_MANIFEST_RELATIVE_PATH)?,
        &read_repository_text(repository_root, CHANGELOG_RELATIVE_PATH)?,
    );
    if request.requires_commit_on_main && !is_checked_out_commit_on_main(repository_root)? {
        problems.push("the tagged commit is not on origin/main; tag a merged commit".to_owned());
    }
    if problems.is_empty() {
        eprintln!("[opencode-plugin-release] status=success tag={tag}");
        return Ok(());
    }
    for problem in &problems {
        eprintln!("[opencode-plugin-release] status=failed reason={problem}");
    }
    Err(format!(
        "{tag} cannot be published: {}",
        problems.join("; ")
    ))
}
