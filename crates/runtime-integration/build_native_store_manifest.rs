//! Integrity manifest and filesystem operations for native store entries.
//!
//! Publication rejects symlinks and hashes every reusable product so a partial
//! cache restore or interrupted write cannot become executable build input.

use std::{
    error::Error,
    fs::{self, File},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
};

use sha2::{Digest, Sha256};

use super::{NativeBuildProfile, store_schema_version};

pub(super) fn validate_entry(
    entry_directory: &Path,
    native_identity: &str,
    native_build_profile: NativeBuildProfile,
) -> Result<(), Box<dyn Error>> {
    require_regular_directory(entry_directory, "native entry")?;
    let completion_path = entry_directory.join("complete");
    require_regular_file(&completion_path, "native completion marker")?;
    if fs::read_to_string(&completion_path)? != format!("{native_identity}\n") {
        return Err("native completion marker does not match its identity".into());
    }
    let manifest_path = entry_directory.join("manifest.sha256");
    require_regular_file(&manifest_path, "native entry manifest")?;
    let manifest_text = fs::read_to_string(&manifest_path)?;
    let expected_prefix = format!(
        "schema={}\nidentity={native_identity}\nprofile={}\n",
        store_schema_version(),
        native_build_profile.identity_name()
    );
    let payload_manifest = manifest_text
        .strip_prefix(&expected_prefix)
        .ok_or("native manifest compatibility header is invalid")?;
    if payload_manifest != payload_manifest_text(entry_directory)? {
        return Err("native entry payload does not match its SHA-256 manifest".into());
    }
    require_profile_products(entry_directory, native_build_profile)
}

pub(super) fn write_manifest(
    payload_directory: &Path,
    native_identity: &str,
    native_build_profile: NativeBuildProfile,
) -> Result<(), Box<dyn Error>> {
    let manifest_text = format!(
        "schema={}\nidentity={native_identity}\nprofile={}\n{}",
        store_schema_version(),
        native_build_profile.identity_name(),
        payload_manifest_text(payload_directory)?
    );
    write_synced_file(
        &payload_directory.join("manifest.sha256"),
        manifest_text.as_bytes(),
    )
}

pub(super) fn copy_directory_without_symlinks(
    source_directory: &Path,
    destination_directory: &Path,
) -> Result<(), Box<dyn Error>> {
    require_regular_directory(source_directory, "native source header directory")?;
    fs::create_dir_all(destination_directory)?;
    for directory_entry_result in fs::read_dir(source_directory)? {
        let directory_entry = directory_entry_result?;
        let source_path = directory_entry.path();
        let destination_path = destination_directory.join(directory_entry.file_name());
        let source_metadata = fs::symlink_metadata(&source_path)?;
        if source_metadata.file_type().is_symlink() {
            return Err(format!(
                "native source contains a symlink: {}",
                source_path.display()
            )
            .into());
        }
        if source_metadata.is_dir() {
            copy_directory_without_symlinks(&source_path, &destination_path)?;
        } else if source_metadata.is_file() {
            copy_required_file(&source_path, &destination_path)?;
        } else {
            return Err(format!(
                "native source contains an unsupported entry: {}",
                source_path.display()
            )
            .into());
        }
    }
    Ok(())
}

pub(super) fn copy_required_file(
    source_path: &Path,
    destination_path: &Path,
) -> Result<(), Box<dyn Error>> {
    require_regular_file(source_path, "native build product")?;
    let destination_parent = destination_path
        .parent()
        .ok_or("native build product destination has no parent")?;
    fs::create_dir_all(destination_parent)?;
    fs::copy(source_path, destination_path)?;
    File::open(destination_path)?.sync_all()?;
    Ok(())
}

pub(super) fn write_synced_file(file_path: &Path, file_bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    let mut output_file = File::create(file_path)?;
    output_file.write_all(file_bytes)?;
    output_file.sync_all()?;
    Ok(())
}

pub(super) fn sync_directory(directory_path: &Path) -> Result<(), Box<dyn Error>> {
    File::open(directory_path)?.sync_all()?;
    Ok(())
}

pub(super) fn sync_directory_tree(directory_path: &Path) -> Result<(), Box<dyn Error>> {
    require_regular_directory(directory_path, "native publication directory")?;
    for directory_entry_result in fs::read_dir(directory_path)? {
        let directory_entry = directory_entry_result?;
        let entry_path = directory_entry.path();
        let entry_metadata = fs::symlink_metadata(&entry_path)?;
        if entry_metadata.file_type().is_symlink() {
            return Err(format!(
                "native publication contains a symlink: {}",
                entry_path.display()
            )
            .into());
        }
        if entry_metadata.is_dir() {
            sync_directory_tree(&entry_path)?;
        } else if !entry_metadata.is_file() {
            return Err(format!(
                "native publication contains an unsupported entry: {}",
                entry_path.display()
            )
            .into());
        }
    }
    sync_directory(directory_path)
}

fn require_profile_products(
    entry_directory: &Path,
    native_build_profile: NativeBuildProfile,
) -> Result<(), Box<dyn Error>> {
    for (relative_path, description) in [
        ("include/mlx/c/mlx.h", "published MLX C umbrella header"),
        ("lib/libmlx.a", "published MLX static library"),
        ("lib/libmlxc.a", "published MLX C static library"),
        ("share/mlx/mlx.metallib", "published MLX metallib"),
    ] {
        require_regular_file(&entry_directory.join(relative_path), description)?;
    }
    if native_build_profile.should_build_memory_contract_probe() {
        require_regular_file(
            &entry_directory.join("bin/mlx_memory_contract_probe"),
            "published MLX memory contract probe",
        )?;
    }
    if native_build_profile.should_build_experimental_aligned_expert_packs() {
        require_regular_file(
            &entry_directory.join("lib/libastronomical_metal_expert_loader.a"),
            "published experimental native library",
        )?;
    }
    Ok(())
}

fn payload_manifest_text(payload_directory: &Path) -> Result<String, Box<dyn Error>> {
    let mut payload_paths = Vec::new();
    collect_payload_paths(payload_directory, payload_directory, &mut payload_paths)?;
    payload_paths.sort();
    let mut manifest_text = String::new();
    for relative_path in payload_paths {
        if relative_path == Path::new("manifest.sha256") || relative_path == Path::new("complete") {
            continue;
        }
        let absolute_path = payload_directory.join(&relative_path);
        let file_size_bytes = absolute_path.metadata()?.len();
        let relative_path_text = relative_path
            .to_str()
            .ok_or("native payload path is not valid UTF-8")?;
        if relative_path_text.contains(['\n', '\r', '\t']) {
            return Err("native payload path contains a manifest delimiter".into());
        }
        manifest_text.push_str(&format!(
            "{}\t{}\t{}\n",
            sha256_file_hex(&absolute_path)?,
            file_size_bytes,
            relative_path_text
        ));
    }
    Ok(manifest_text)
}

fn collect_payload_paths(
    payload_root: &Path,
    current_directory: &Path,
    payload_paths: &mut Vec<PathBuf>,
) -> Result<(), Box<dyn Error>> {
    for directory_entry_result in fs::read_dir(current_directory)? {
        let directory_entry = directory_entry_result?;
        let entry_path = directory_entry.path();
        let entry_metadata = fs::symlink_metadata(&entry_path)?;
        if entry_metadata.file_type().is_symlink() {
            return Err(format!(
                "native payload contains a symlink: {}",
                entry_path.display()
            )
            .into());
        }
        if entry_metadata.is_dir() {
            collect_payload_paths(payload_root, &entry_path, payload_paths)?;
        } else if entry_metadata.is_file() {
            let relative_path = entry_path.strip_prefix(payload_root)?.to_owned();
            validate_relative_payload_path(&relative_path)?;
            payload_paths.push(relative_path);
        } else {
            return Err(format!(
                "native payload contains an unsupported entry: {}",
                entry_path.display()
            )
            .into());
        }
    }
    Ok(())
}

fn validate_relative_payload_path(relative_path: &Path) -> Result<(), Box<dyn Error>> {
    if relative_path.is_absolute()
        || relative_path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!(
            "native payload path escapes its entry: {}",
            relative_path.display()
        )
        .into());
    }
    Ok(())
}

fn require_regular_file(file_path: &Path, description: &str) -> Result<(), Box<dyn Error>> {
    let file_metadata = fs::symlink_metadata(file_path)
        .map_err(|source| format!("missing {description} at {}: {source}", file_path.display()))?;
    if !file_metadata.is_file() || file_metadata.file_type().is_symlink() {
        return Err(format!(
            "{description} is not a regular file: {}",
            file_path.display()
        )
        .into());
    }
    Ok(())
}

fn require_regular_directory(
    directory_path: &Path,
    description: &str,
) -> Result<(), Box<dyn Error>> {
    let directory_metadata = fs::symlink_metadata(directory_path).map_err(|source| {
        format!(
            "missing {description} at {}: {source}",
            directory_path.display()
        )
    })?;
    if !directory_metadata.is_dir() || directory_metadata.file_type().is_symlink() {
        return Err(format!(
            "{description} is not a regular directory: {}",
            directory_path.display()
        )
        .into());
    }
    Ok(())
}

fn sha256_file_hex(file_path: &Path) -> Result<String, Box<dyn Error>> {
    let mut source_file = File::open(file_path)?;
    let mut digest = Sha256::new();
    let mut digest_buffer = [0_u8; 64 * 1024];
    loop {
        let bytes_read = source_file.read(&mut digest_buffer)?;
        if bytes_read == 0 {
            break;
        }
        digest.update(&digest_buffer[..bytes_read]);
    }
    Ok(digest
        .finalize()
        .iter()
        .map(|digest_byte| format!("{digest_byte:02x}"))
        .collect())
}
