//! `cargo xtask skill-site [--check]`: publishes the Astronomical skill to the
//! website so installed apps can pick up a newer skill without an app release
//! (issue #2397).
//!
//! The skill's source is `skill/astronomical`, and `skill/release.json` states
//! its version and the oldest app version it suits. This command copies the
//! files to `site/skill/astronomical` and writes `site/skill/manifest.json`,
//! which adds each file's SHA-256 so an app can tell a complete download from
//! a partial one. `--check` changes nothing and fails when the site's copy
//! differs, so a skill edit cannot merge without its published copy.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use sha2::{Digest, Sha256};

const SKILL_SOURCE_DIRECTORY: &str = "skill/astronomical";
const SKILL_RELEASE_FILE: &str = "skill/release.json";
const SITE_SKILL_DIRECTORY: &str = "site/skill";
const SITE_SKILL_FILES_FOLDER: &str = "astronomical";
const SITE_MANIFEST_FILE_NAME: &str = "manifest.json";
const SKILL_INDEX_FILE_NAME: &str = "SKILL.md";
const REFERENCES_FOLDER_NAME: &str = "references";
const USAGE: &str = "usage: cargo xtask skill-site [--check]";

pub fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    let is_check_only = match arguments {
        [] => false,
        [flag] if flag == "--check" => true,
        _ => return Err(USAGE.to_owned()),
    };
    let published_files = published_files(repository_root)?;
    let site_skill_directory = repository_root.join(SITE_SKILL_DIRECTORY);
    let stale_paths = stale_site_paths(&site_skill_directory, &published_files)?;
    let differing_paths = differing_site_paths(&site_skill_directory, &published_files);
    if is_check_only {
        return check_outcome(&differing_paths, &stale_paths);
    }
    for (relative_path, file_bytes) in &published_files {
        let site_path = site_skill_directory.join(relative_path);
        if let Some(parent_directory) = site_path.parent() {
            fs::create_dir_all(parent_directory).map_err(|create_error| {
                format!(
                    "cannot create {}: {create_error}",
                    parent_directory.display()
                )
            })?;
        }
        fs::write(&site_path, file_bytes).map_err(|write_error| {
            format!("cannot write {}: {write_error}", site_path.display())
        })?;
    }
    for stale_path in &stale_paths {
        fs::remove_file(site_skill_directory.join(stale_path)).map_err(|remove_error| {
            format!("cannot remove site/skill/{stale_path}: {remove_error}")
        })?;
    }
    println!(
        "[skill-site] status=written files={} changed={} removed={}",
        published_files.len(),
        differing_paths.len(),
        stale_paths.len()
    );
    Ok(())
}

fn check_outcome(differing_paths: &[String], stale_paths: &[String]) -> Result<(), String> {
    if differing_paths.is_empty() && stale_paths.is_empty() {
        println!("[skill-site] status=current");
        return Ok(());
    }
    let mut problems = Vec::new();
    for differing_path in differing_paths {
        problems.push(format!("[skill-site] differs: site/skill/{differing_path}"));
    }
    for stale_path in stale_paths {
        problems.push(format!(
            "[skill-site] no longer in the skill: site/skill/{stale_path}"
        ));
    }
    problems.push("the published skill is out of date; run `cargo xtask skill-site`".to_owned());
    Err(problems.join("\n"))
}

/// Every file the site must hold, keyed by its path under `site/skill`: the
/// skill's files and the manifest that describes them.
fn published_files(repository_root: &Path) -> Result<BTreeMap<String, Vec<u8>>, String> {
    let skill_files = skill_source_files(&repository_root.join(SKILL_SOURCE_DIRECTORY))?;
    let release_text = fs::read_to_string(repository_root.join(SKILL_RELEASE_FILE))
        .map_err(|read_error| format!("cannot read {SKILL_RELEASE_FILE}: {read_error}"))?;
    let manifest_text = manifest_text(&release_text, &skill_files)?;
    let mut published_files = BTreeMap::new();
    for (relative_path, file_bytes) in skill_files {
        published_files.insert(
            format!("{SITE_SKILL_FILES_FOLDER}/{relative_path}"),
            file_bytes,
        );
    }
    published_files.insert(
        SITE_MANIFEST_FILE_NAME.to_owned(),
        manifest_text.into_bytes(),
    );
    Ok(published_files)
}

/// The skill's index and its reference files, keyed by path inside the skill.
fn skill_source_files(skill_source_directory: &Path) -> Result<BTreeMap<String, Vec<u8>>, String> {
    let mut relative_paths = vec![SKILL_INDEX_FILE_NAME.to_owned()];
    let references_directory = skill_source_directory.join(REFERENCES_FOLDER_NAME);
    let directory_entries = fs::read_dir(&references_directory).map_err(|read_error| {
        format!(
            "cannot list {}: {read_error}",
            references_directory.display()
        )
    })?;
    for directory_entry in directory_entries {
        let file_name = directory_entry
            .map_err(|entry_error| format!("cannot list the skill's references: {entry_error}"))?
            .file_name();
        relative_paths.push(format!(
            "{REFERENCES_FOLDER_NAME}/{}",
            file_name.to_string_lossy()
        ));
    }
    let mut skill_files = BTreeMap::new();
    for relative_path in relative_paths {
        let file_bytes =
            fs::read(skill_source_directory.join(&relative_path)).map_err(|read_error| {
                format!("cannot read skill file {relative_path}: {read_error}")
            })?;
        skill_files.insert(relative_path, file_bytes);
    }
    Ok(skill_files)
}

/// The manifest: the release's version and oldest suitable app version, plus
/// each file's SHA-256, as stable pretty JSON so a rerun changes no bytes.
fn manifest_text(
    release_text: &str,
    skill_files: &BTreeMap<String, Vec<u8>>,
) -> Result<String, String> {
    let release: serde_json::Value = serde_json::from_str(release_text)
        .map_err(|parse_error| format!("{SKILL_RELEASE_FILE} is not JSON: {parse_error}"))?;
    let Some(skill_version) = release["skill_version"].as_u64() else {
        return Err(format!(
            "{SKILL_RELEASE_FILE} needs a whole-number skill_version"
        ));
    };
    let Some(minimum_app_version) = release["minimum_app_version"].as_str() else {
        return Err(format!("{SKILL_RELEASE_FILE} needs a minimum_app_version"));
    };
    let mut file_digests = serde_json::Map::new();
    for (relative_path, file_bytes) in skill_files {
        let digest = lowercase_hex::encode_lowercase_hex(&Sha256::digest(file_bytes));
        file_digests.insert(relative_path.clone(), serde_json::Value::String(digest));
    }
    let manifest = serde_json::json!({
        "skill_version": skill_version,
        "minimum_app_version": minimum_app_version,
        "files": file_digests,
    });
    let mut manifest_text = serde_json::to_string_pretty(&manifest)
        .map_err(|serialize_error| format!("cannot write the skill manifest: {serialize_error}"))?;
    manifest_text.push('\n');
    Ok(manifest_text)
}

fn differing_site_paths(
    site_skill_directory: &Path,
    published_files: &BTreeMap<String, Vec<u8>>,
) -> Vec<String> {
    let mut differing_paths = Vec::new();
    for (relative_path, expected_bytes) in published_files {
        let site_bytes = fs::read(site_skill_directory.join(relative_path)).ok();
        if site_bytes.as_deref() != Some(expected_bytes.as_slice()) {
            differing_paths.push(relative_path.clone());
        }
    }
    differing_paths
}

/// Files under `site/skill` that the skill no longer has.
fn stale_site_paths(
    site_skill_directory: &Path,
    published_files: &BTreeMap<String, Vec<u8>>,
) -> Result<Vec<String>, String> {
    let mut site_paths = Vec::new();
    collect_relative_file_paths(site_skill_directory, PathBuf::new(), &mut site_paths)?;
    let mut stale_paths = Vec::new();
    for site_path in site_paths {
        if !published_files.contains_key(&site_path) {
            stale_paths.push(site_path);
        }
    }
    Ok(stale_paths)
}

fn collect_relative_file_paths(
    root_directory: &Path,
    relative_directory: PathBuf,
    relative_file_paths: &mut Vec<String>,
) -> Result<(), String> {
    let Ok(directory_entries) = fs::read_dir(root_directory.join(&relative_directory)) else {
        return Ok(());
    };
    for directory_entry in directory_entries {
        let directory_entry = directory_entry
            .map_err(|entry_error| format!("cannot list site/skill: {entry_error}"))?;
        let relative_path = relative_directory.join(directory_entry.file_name());
        if directory_entry.path().is_dir() {
            collect_relative_file_paths(root_directory, relative_path, relative_file_paths)?;
        } else {
            relative_file_paths.push(relative_path.to_string_lossy().into_owned());
        }
    }
    Ok(())
}
