//! Pinned-manifest policy for the native dependency bootstrap: one line per
//! archive, four pipe-separated fields, exactly five pinned archives. The
//! validations mirror what the CMake manifest can express and what the cache
//! layer must never have to trust: no line-break or empty fields, a path-safe
//! archive file name, HTTPS-only URLs, and a lowercase 64-hex SHA-256.

pub(crate) const EXPECTED_ARCHIVE_COUNT: usize = 5;

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ManifestEntry {
    pub(crate) archive_file_name: String,
    pub(crate) archive_url: String,
    pub(crate) archive_sha256: String,
    pub(crate) dependency_description: String,
}

pub(crate) fn parse_manifest_text(manifest_text: &str) -> Result<Vec<ManifestEntry>, String> {
    let mut manifest_entries = Vec::new();
    for manifest_line in manifest_text.lines() {
        manifest_entries.push(parse_manifest_line(manifest_line)?);
    }
    if manifest_entries.len() != EXPECTED_ARCHIVE_COUNT {
        return Err("native dependency manifest must contain exactly five archives".to_owned());
    }
    Ok(manifest_entries)
}

pub(crate) fn parse_manifest_line(manifest_line: &str) -> Result<ManifestEntry, String> {
    let manifest_fields: Vec<&str> = manifest_line.split('|').collect();
    if manifest_fields.len() != 4 {
        return Err(
            "native dependency manifest entry must have four |-separated fields".to_owned(),
        );
    }
    let manifest_entry = ManifestEntry {
        archive_file_name: manifest_fields[0].to_owned(),
        archive_url: manifest_fields[1].to_owned(),
        archive_sha256: manifest_fields[2].to_owned(),
        dependency_description: manifest_fields[3].to_owned(),
    };
    validate_manifest_field("archive file name", &manifest_entry.archive_file_name)?;
    validate_manifest_field("archive URL", &manifest_entry.archive_url)?;
    validate_manifest_field("archive SHA-256", &manifest_entry.archive_sha256)?;
    validate_manifest_field(
        "dependency description",
        &manifest_entry.dependency_description,
    )?;
    validate_archive_file_name(&manifest_entry.archive_file_name)?;
    validate_archive_url(&manifest_entry.archive_url)?;
    validate_archive_sha256(&manifest_entry.archive_sha256)?;
    Ok(manifest_entry)
}

fn validate_manifest_field(field_name: &str, field_text: &str) -> Result<(), String> {
    let is_delimited_field =
        !field_text.is_empty() && !field_text.contains('\n') && !field_text.contains('\r');
    if is_delimited_field {
        Ok(())
    } else {
        Err(format!(
            "native dependency manifest has an invalid {field_name}"
        ))
    }
}

fn validate_archive_file_name(file_name: &str) -> Result<(), String> {
    let is_safe_file_name = !file_name.is_empty()
        && file_name.bytes().all(|byte| {
            matches!(
                byte,
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'.' | b'_' | b'-'
            )
        });
    if is_safe_file_name {
        Ok(())
    } else {
        Err("native dependency manifest has an unsafe archive file name".to_owned())
    }
}

fn validate_archive_url(archive_url: &str) -> Result<(), String> {
    if archive_url.starts_with("https://") {
        Ok(())
    } else {
        Err("native dependency manifest requires an HTTPS archive URL".to_owned())
    }
}

fn validate_archive_sha256(archive_sha256: &str) -> Result<(), String> {
    let is_lowercase_hex = archive_sha256
        .bytes()
        .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'));
    if !is_lowercase_hex {
        return Err("native dependency manifest has a non-lowercase SHA-256".to_owned());
    }
    if archive_sha256.len() != 64 {
        return Err("native dependency manifest has an invalid SHA-256 length".to_owned());
    }
    Ok(())
}
