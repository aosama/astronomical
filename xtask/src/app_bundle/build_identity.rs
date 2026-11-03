//! The build identity stamped into the bundle and the daemon binary: version
//! and repository from Cargo metadata, commit, build number, and dirty state
//! from git, and the UTC build date.

use std::path::Path;
use std::process::Command;

/// Identity facts shared by `Info.plist` and the daemon's `--version`, which
/// the bundle check compares so a stale daemon can never hide in a new app.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BuildIdentity {
    pub(crate) application_version: String,
    pub(crate) repository_url: String,
    pub(crate) build_commit: String,
    pub(crate) build_number: String,
    pub(crate) build_date_utc: String,
    pub(crate) is_dirty: bool,
}

impl BuildIdentity {
    pub(crate) fn gather(repository_root: &Path) -> Result<Self, String> {
        let cargo_metadata_text = command_stdout(
            repository_root,
            "cargo",
            &["metadata", "--no-deps", "--format-version", "1"],
        )?;
        let (application_version, repository_url) = daemon_package_identity(&cargo_metadata_text)?;
        let porcelain_status = command_stdout(
            repository_root,
            "git",
            &["status", "--porcelain", "--untracked-files=normal"],
        )?;
        Ok(Self {
            application_version,
            repository_url,
            build_commit: command_stdout(
                repository_root,
                "git",
                &["rev-parse", "--short=12", "HEAD"],
            )?,
            build_number: command_stdout(repository_root, "git", &["rev-list", "--count", "HEAD"])?,
            // One UTC date for every stamped field, so the visible and the
            // machine-readable build identities cannot disagree at midnight.
            build_date_utc: command_stdout(repository_root, "date", &["-u", "+%Y%m%d"])?,
            is_dirty: !porcelain_status.is_empty(),
        })
    }

    /// The environment the daemon's build script reads to embed the same
    /// identity into `astronomicald --version`.
    pub(crate) fn cargo_environment(&self) -> [(&'static str, String); 3] {
        [
            ("ASTRONOMICAL_BUILD_COMMIT", self.build_commit.clone()),
            ("ASTRONOMICAL_BUILD_NUMBER", self.build_number.clone()),
            ("ASTRONOMICAL_BUILD_DIRTY", self.is_dirty.to_string()),
        ]
    }
}

/// Version and repository URL of the `astronomical-daemon` package, which
/// owns the product version.
pub(crate) fn daemon_package_identity(
    cargo_metadata_text: &str,
) -> Result<(String, String), String> {
    let cargo_metadata: serde_json::Value = match serde_json::from_str(cargo_metadata_text) {
        Ok(parsed_metadata) => parsed_metadata,
        Err(parse_error) => return Err(format!("cargo metadata is not JSON: {parse_error}")),
    };
    let Some(workspace_packages) = cargo_metadata["packages"].as_array() else {
        return Err("cargo metadata has no packages array".to_owned());
    };
    for workspace_package in workspace_packages {
        if workspace_package["name"] != "astronomical-daemon" {
            continue;
        }
        let (Some(package_version), Some(package_repository)) = (
            workspace_package["version"].as_str(),
            workspace_package["repository"].as_str(),
        ) else {
            return Err("astronomical-daemon lacks a version or repository".to_owned());
        };
        return Ok((package_version.to_owned(), package_repository.to_owned()));
    }
    Err("cargo metadata does not list astronomical-daemon".to_owned())
}

/// The Sparkle appcast URL GitHub Pages serves for the repository.
pub(crate) fn github_pages_feed_url(repository_url: &str) -> Result<String, String> {
    let Some(repository_slug) = repository_url.strip_prefix("https://github.com/") else {
        return Err(
            "the repository must be an HTTPS GitHub URL to derive the update feed".to_owned(),
        );
    };
    let repository_slug = repository_slug
        .strip_suffix(".git")
        .unwrap_or(repository_slug);
    match repository_slug.split_once('/') {
        Some((repository_owner, repository_name))
            if !repository_owner.is_empty()
                && !repository_name.is_empty()
                && !repository_name.contains('/') =>
        {
            Ok(format!(
                "https://{repository_owner}.github.io/{repository_name}/appcast.xml"
            ))
        }
        _ => Err("the repository URL does not name an owner and a repository".to_owned()),
    }
}

fn command_stdout(
    working_directory: &Path,
    program: &str,
    arguments: &[&str],
) -> Result<String, String> {
    let command_output = match Command::new(program)
        .args(arguments)
        .current_dir(working_directory)
        .output()
    {
        Ok(command_output) => command_output,
        Err(spawn_error) => return Err(format!("could not run {program}: {spawn_error}")),
    };
    if !command_output.status.success() {
        return Err(format!(
            "{program} {} failed: {}",
            arguments.join(" "),
            String::from_utf8_lossy(&command_output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&command_output.stdout)
        .trim()
        .to_owned())
}
