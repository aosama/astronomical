//! Owns the one-time removal of the retired complete CMake tree from `OUT_DIR`.
//! Keeping this operation independent makes its narrow destructive boundary
//! behaviorally testable without running the native build.

use std::{error::Error, fs, path::Path, time::Instant};

const LEGACY_CARGO_NATIVE_BUILD_DIRECTORY_NAME: &str = "mlx-c-runtime-build";

pub(crate) fn remove_legacy_cargo_native_build_directory(
    output_directory: &Path,
) -> Result<(), Box<dyn Error>> {
    let legacy_native_build_directory =
        output_directory.join(LEGACY_CARGO_NATIVE_BUILD_DIRECTORY_NAME);
    let legacy_directory_metadata = match fs::symlink_metadata(&legacy_native_build_directory) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    if legacy_directory_metadata.file_type().is_symlink()
        || !legacy_directory_metadata.file_type().is_dir()
    {
        return Err(format!(
            "refusing to remove unexpected legacy native output: {}",
            legacy_native_build_directory.display()
        )
        .into());
    }

    // Cargo intentionally preserves OUT_DIR between build-script runs, so the
    // retired complete CMake tree needs an explicit one-time ownership handoff.
    let cleanup_started_at = Instant::now();
    eprintln!(
        "[native-build-store] operation=remove-legacy-cargo-native-output status=start path={}",
        legacy_native_build_directory.display()
    );
    fs::remove_dir_all(&legacy_native_build_directory)?;
    eprintln!(
        "[native-build-store] operation=remove-legacy-cargo-native-output status=success elapsed_ms={}",
        cleanup_started_at.elapsed().as_millis()
    );
    Ok(())
}
