//! The prepared release's ledger: tag, commit, version, and the name, SHA-256,
//! and size of the DMG, the release notes, and the appcast. Publishing
//! re-hashes every artifact against it, so the bytes that were reviewed are
//! provably the bytes that are uploaded.

use std::path::Path;

use astronomical_native_identity::digest_bytes;

pub(crate) const MANIFEST_FILE_NAME: &str = "release-manifest.json";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ArtifactRecord {
    pub(crate) artifact_name: String,
    pub(crate) sha256: String,
    pub(crate) size_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReleaseManifest {
    pub(crate) release_tag: String,
    pub(crate) release_commit: String,
    pub(crate) release_version: String,
    pub(crate) disk_image: ArtifactRecord,
    pub(crate) release_notes: ArtifactRecord,
    pub(crate) appcast: ArtifactRecord,
}

pub(crate) fn record_artifact(artifact_path: &Path) -> Result<ArtifactRecord, String> {
    let artifact_bytes = match std::fs::read(artifact_path) {
        Ok(artifact_bytes) => artifact_bytes,
        Err(read_error) => {
            return Err(format!(
                "could not read {}: {read_error}",
                artifact_path.display()
            ));
        }
    };
    let artifact_name = match artifact_path.file_name() {
        Some(artifact_name) => artifact_name.to_string_lossy().into_owned(),
        None => return Err(format!("{} has no file name", artifact_path.display())),
    };
    Ok(ArtifactRecord {
        artifact_name,
        sha256: digest_bytes(&artifact_bytes),
        size_bytes: artifact_bytes.len() as u64,
    })
}

/// The artifact on disk must hash and measure exactly as recorded.
pub(crate) fn verify_artifact(
    artifact_path: &Path,
    expected_record: &ArtifactRecord,
    artifact_label: &str,
) -> Result<(), String> {
    let actual_record = match record_artifact(artifact_path) {
        Ok(actual_record) => actual_record,
        Err(_) => return Err("prepared release artifacts are incomplete".to_owned()),
    };
    if actual_record.size_bytes == 0 {
        return Err("prepared release artifacts are incomplete".to_owned());
    }
    if actual_record.sha256 != expected_record.sha256 {
        return Err(format!(
            "prepared {artifact_label} digest does not match its manifest"
        ));
    }
    if actual_record.size_bytes != expected_record.size_bytes {
        return Err(format!(
            "prepared {artifact_label} size does not match its manifest"
        ));
    }
    Ok(())
}

impl ReleaseManifest {
    pub(crate) fn to_json_text(&self) -> String {
        serde_json::json!({
            "tag": self.release_tag,
            "commit": self.release_commit,
            "version": self.release_version,
            "dmg": artifact_json(&self.disk_image),
            "notes": artifact_json(&self.release_notes),
            "appcast": artifact_json(&self.appcast),
        })
        .to_string()
    }

    pub(crate) fn from_json_text(manifest_text: &str) -> Result<Self, String> {
        let manifest_document: serde_json::Value = match serde_json::from_str(manifest_text) {
            Ok(manifest_document) => manifest_document,
            Err(_) => return Err("prepared release manifest is unavailable".to_owned()),
        };
        let (Some(release_tag), Some(release_commit), Some(release_version)) = (
            manifest_document["tag"].as_str(),
            manifest_document["commit"].as_str(),
            manifest_document["version"].as_str(),
        ) else {
            return Err("prepared release manifest is incomplete".to_owned());
        };
        Ok(Self {
            release_tag: release_tag.to_owned(),
            release_commit: release_commit.to_owned(),
            release_version: release_version.to_owned(),
            disk_image: artifact_from_json(&manifest_document["dmg"])?,
            release_notes: artifact_from_json(&manifest_document["notes"])?,
            appcast: artifact_from_json(&manifest_document["appcast"])?,
        })
    }
}

fn artifact_json(artifact_record: &ArtifactRecord) -> serde_json::Value {
    serde_json::json!({
        "name": artifact_record.artifact_name,
        "sha256": artifact_record.sha256,
        "size": artifact_record.size_bytes,
    })
}

fn artifact_from_json(artifact_document: &serde_json::Value) -> Result<ArtifactRecord, String> {
    let (Some(artifact_name), Some(sha256), Some(size_bytes)) = (
        artifact_document["name"].as_str(),
        artifact_document["sha256"].as_str(),
        artifact_document["size"].as_u64(),
    ) else {
        return Err("prepared release manifest is incomplete".to_owned());
    };
    Ok(ArtifactRecord {
        artifact_name: artifact_name.to_owned(),
        sha256: sha256.to_owned(),
        size_bytes,
    })
}
