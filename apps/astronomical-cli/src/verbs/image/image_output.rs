//! Where a generated image lands: choosing the path, claiming it before any
//! work starts, and writing the PNG only once it is known to be one.
//!
//! Generation can take minutes of GPU time, so the output file is claimed
//! first. A path that already exists, sits in a missing folder, or cannot be
//! written then fails in a moment, not after the image is already drawn and
//! has nowhere to go. Claiming uses `create_new`, so an existing file is
//! never overwritten, by this process or by a race with another one.

use std::fs::{File, OpenOptions};
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use super::ImageError;

/// The eight bytes every PNG begins with.
const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

/// An output path this process created and now owns, still empty.
pub(super) struct ReservedOutput {
    path: PathBuf,
    file: File,
}

/// Picks the path (the requested one, or a timestamped name in
/// `default_directory`) and claims it.
pub(super) fn reserve_output_file(
    requested_path: Option<&Path>,
    default_directory: &Path,
) -> Result<ReservedOutput, ImageError> {
    let output_path = match requested_path {
        Some(requested_path) => requested_path.to_path_buf(),
        None => default_directory.join(default_file_name()),
    };
    if !has_png_extension(&output_path) {
        return Err(ImageError::OutputMustBePng { path: output_path });
    }
    match OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output_path)
    {
        Ok(file) => Ok(ReservedOutput {
            path: output_path,
            file,
        }),
        Err(open_error) if open_error.kind() == ErrorKind::AlreadyExists => {
            Err(ImageError::OutputExists { path: output_path })
        }
        Err(open_error) => Err(ImageError::OutputUnwritable {
            path: output_path,
            cause: open_error.to_string(),
        }),
    }
}

/// The image is always a PNG, so a different extension would mislabel it.
fn has_png_extension(output_path: &Path) -> bool {
    match output_path.extension().and_then(std::ffi::OsStr::to_str) {
        Some(extension) => extension.eq_ignore_ascii_case("png"),
        None => false,
    }
}

/// `astronomical-image-<unix seconds>.png`: sorts by creation time and does
/// not collide across runs. Two runs in the same second collide on purpose:
/// the claim then refuses rather than overwrites.
fn default_file_name() -> String {
    let unix_seconds = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration_since_epoch) => duration_since_epoch.as_secs(),
        Err(_clock_before_epoch) => 0,
    };
    format!("astronomical-image-{unix_seconds}.png")
}

/// Writes the PNG into the claimed file. A reply that is not a PNG is a
/// daemon fault, so the empty claim is removed instead of leaving a corrupt
/// image behind.
pub(super) fn write_png(
    mut reserved_output: ReservedOutput,
    png_bytes: &[u8],
) -> Result<PathBuf, ImageError> {
    if !png_bytes.starts_with(&PNG_SIGNATURE) {
        discard_reserved_output(reserved_output);
        return Err(ImageError::ReturnedImageInvalid);
    }
    let write_result = match reserved_output.file.write_all(png_bytes) {
        Ok(()) => reserved_output.file.flush(),
        Err(write_error) => Err(write_error),
    };
    match write_result {
        Ok(()) => Ok(absolute_path(&reserved_output.path)),
        Err(write_error) => {
            let output_path = reserved_output.path.clone();
            discard_reserved_output(reserved_output);
            Err(ImageError::OutputUnwritable {
                path: output_path,
                cause: write_error.to_string(),
            })
        }
    }
}

/// Removes a claimed file that will never be filled.
pub(super) fn discard_reserved_output(reserved_output: ReservedOutput) {
    let ReservedOutput { path, file } = reserved_output;
    drop(file);
    // Best effort: the claim was ours and empty, so a failed removal leaves
    // at worst an empty file the user can see and delete.
    let _removal_result = std::fs::remove_file(path);
}

/// The path to print: absolute when the OS can say, as given when not.
fn absolute_path(output_path: &Path) -> PathBuf {
    match std::fs::canonicalize(output_path) {
        Ok(canonical_path) => canonical_path,
        Err(_canonicalize_error) => output_path.to_path_buf(),
    }
}
