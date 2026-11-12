//! Where a generated image lands: choosing the path, claiming it before any
//! work starts (`reserved_output_file`), and writing the PNG only once it is
//! known to be one.

use std::path::{Path, PathBuf};

use super::ImageError;
use crate::verbs::reserved_output_file::{
    OutputReservationError, ReservedOutputFile, timestamped_file_name,
};

/// The eight bytes every PNG begins with.
const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

/// An output path this process created and now owns, still empty.
pub(super) type ReservedOutput = ReservedOutputFile;

/// Picks the path (the requested one, or a timestamped name in
/// `default_directory`) and claims it.
pub(super) fn reserve_output_file(
    requested_path: Option<&Path>,
    default_directory: &Path,
) -> Result<ReservedOutput, ImageError> {
    let output_path = match requested_path {
        Some(requested_path) => requested_path.to_path_buf(),
        None => default_directory.join(timestamped_file_name("image", "png")),
    };
    if !has_png_extension(&output_path) {
        return Err(ImageError::OutputMustBePng { path: output_path });
    }
    ReservedOutputFile::claim(&output_path).map_err(|reservation_error| match reservation_error {
        OutputReservationError::AlreadyExists => ImageError::OutputExists { path: output_path },
        OutputReservationError::Unwritable { cause } => ImageError::OutputUnwritable {
            path: output_path,
            cause,
        },
    })
}

/// The image is always a PNG, so a different extension would mislabel it.
fn has_png_extension(output_path: &Path) -> bool {
    match output_path.extension().and_then(std::ffi::OsStr::to_str) {
        Some(extension) => extension.eq_ignore_ascii_case("png"),
        None => false,
    }
}

/// Writes the PNG into the claimed file. A reply that is not a PNG is a
/// daemon fault, so the empty claim is removed instead of leaving a corrupt
/// image behind.
pub(super) fn write_png(
    reserved_output: ReservedOutput,
    png_bytes: &[u8],
) -> Result<PathBuf, ImageError> {
    if !png_bytes.starts_with(&PNG_SIGNATURE) {
        reserved_output.discard();
        return Err(ImageError::ReturnedImageInvalid);
    }
    let output_path = reserved_output.path().to_path_buf();
    reserved_output
        .fill(png_bytes)
        .map_err(|cause| ImageError::OutputUnwritable {
            path: output_path,
            cause,
        })
}

/// Removes a claimed file that will never be filled.
pub(super) fn discard_reserved_output(reserved_output: ReservedOutput) {
    reserved_output.discard();
}
