use std::{
    fs,
    os::unix::fs::symlink,
    path::{Path, PathBuf},
};

use astronomical_runtime_integration::{MlxRuntimeError, resolve_mlx_metallib_path};

fn write_regular_file(file_path: &Path, file_contents: &[u8]) {
    if let Some(parent_directory) = file_path.parent() {
        fs::create_dir_all(parent_directory)
            .expect("the test should create parent directories for fixture files");
    }
    fs::write(file_path, file_contents).expect("the test should write a regular fixture file");
}

fn packaged_worker_layout(sandbox_directory: &Path) -> (PathBuf, PathBuf) {
    let app_bundle_root = sandbox_directory.join("Astronomical.app");
    let worker_executable_path =
        app_bundle_root.join("Contents/MacOS/astronomical-inference-worker");
    let bundled_metallib_path = app_bundle_root.join("Contents/Resources/share/mlx/mlx.metallib");
    write_regular_file(&worker_executable_path, b"worker");
    (worker_executable_path, bundled_metallib_path)
}

#[test]
fn should_select_the_bundled_metallib_for_a_macos_app_executable() {
    let sandbox_directory =
        tempfile::tempdir().expect("the test should create a metallib resolution sandbox");
    let (worker_executable_path, bundled_metallib_path) =
        packaged_worker_layout(sandbox_directory.path());
    write_regular_file(&bundled_metallib_path, b"bundled");
    let unpackaged_native_store_metallib_path =
        sandbox_directory.path().join("native-store/mlx.metallib");
    write_regular_file(&unpackaged_native_store_metallib_path, b"unpackaged");

    let resolved_metallib_path = resolve_mlx_metallib_path(
        None,
        &worker_executable_path,
        Some(&unpackaged_native_store_metallib_path),
    )
    .expect("a packaged worker should load the bundled metallib");

    assert_eq!(
        fs::read(&resolved_metallib_path).expect("the bundled metallib should be readable"),
        b"bundled"
    );
}

#[test]
fn should_prefer_an_environment_override_over_the_bundled_metallib() {
    let sandbox_directory =
        tempfile::tempdir().expect("the test should create a metallib resolution sandbox");
    let (worker_executable_path, bundled_metallib_path) =
        packaged_worker_layout(sandbox_directory.path());
    write_regular_file(&bundled_metallib_path, b"bundled");
    let environment_override_path = sandbox_directory.path().join("override/mlx.metallib");
    write_regular_file(&environment_override_path, b"override");

    let resolved_metallib_path = resolve_mlx_metallib_path(
        Some(environment_override_path.clone()),
        &worker_executable_path,
        None,
    )
    .expect("an explicit metallib override should win");

    assert_eq!(resolved_metallib_path, environment_override_path);
}

#[test]
fn should_use_the_unpackaged_native_store_when_the_executable_is_not_in_an_app_bundle() {
    let sandbox_directory =
        tempfile::tempdir().expect("the test should create a metallib resolution sandbox");
    let worker_executable_path = sandbox_directory
        .path()
        .join("target/release/astronomical-inference-worker");
    write_regular_file(&worker_executable_path, b"worker");
    let unpackaged_native_store_metallib_path =
        sandbox_directory.path().join("native-store/mlx.metallib");
    write_regular_file(&unpackaged_native_store_metallib_path, b"unpackaged");

    let resolved_metallib_path = resolve_mlx_metallib_path(
        None,
        &worker_executable_path,
        Some(&unpackaged_native_store_metallib_path),
    )
    .expect("an unpackaged cargo binary should use the native-store metallib");

    assert_eq!(
        resolved_metallib_path,
        unpackaged_native_store_metallib_path
    );
}

#[test]
fn should_skip_a_symlinked_bundled_metallib_and_use_the_unpackaged_store() {
    let sandbox_directory =
        tempfile::tempdir().expect("the test should create a metallib resolution sandbox");
    let (worker_executable_path, bundled_metallib_path) =
        packaged_worker_layout(sandbox_directory.path());
    let symlink_target_path = sandbox_directory.path().join("elsewhere/mlx.metallib");
    write_regular_file(&symlink_target_path, b"linked");
    fs::create_dir_all(
        bundled_metallib_path
            .parent()
            .expect("the bundled metallib path should have a parent"),
    )
    .expect("the test should create the bundled metallib directory");
    symlink(&symlink_target_path, &bundled_metallib_path)
        .expect("the test should create a bundled metallib symlink");
    let unpackaged_native_store_metallib_path =
        sandbox_directory.path().join("native-store/mlx.metallib");
    write_regular_file(&unpackaged_native_store_metallib_path, b"unpackaged");

    let resolved_metallib_path = resolve_mlx_metallib_path(
        None,
        &worker_executable_path,
        Some(&unpackaged_native_store_metallib_path),
    )
    .expect("a bundled symlink should not be treated as the packaged metallib");

    assert_eq!(
        resolved_metallib_path,
        unpackaged_native_store_metallib_path
    );
}

#[test]
fn should_fail_when_no_metallib_is_available_for_the_worker() {
    let sandbox_directory =
        tempfile::tempdir().expect("the test should create a metallib resolution sandbox");
    let worker_executable_path = sandbox_directory
        .path()
        .join("target/release/astronomical-inference-worker");
    write_regular_file(&worker_executable_path, b"worker");

    let resolution_error = resolve_mlx_metallib_path(None, &worker_executable_path, None)
        .expect_err("a missing metallib should fail before MLX starts");

    assert!(matches!(
        resolution_error,
        MlxRuntimeError::InvalidMetallibPath { .. }
    ));
    let error_description = resolution_error.to_string();
    assert!(
        error_description.contains("Reinstall Astronomical"),
        "missing metallib error should tell the user to reinstall the app: {error_description}"
    );
    assert!(
        error_description.contains("rebuild native artifacts"),
        "missing metallib error should tell cargo-run users to rebuild native artifacts: {error_description}"
    );
}

#[test]
fn should_reject_a_relative_environment_override_path() {
    let sandbox_directory =
        tempfile::tempdir().expect("the test should create a metallib resolution sandbox");
    let worker_executable_path = sandbox_directory
        .path()
        .join("target/release/astronomical-inference-worker");
    write_regular_file(&worker_executable_path, b"worker");

    let resolution_error = resolve_mlx_metallib_path(
        Some(PathBuf::from("mlx.metallib")),
        &worker_executable_path,
        None,
    )
    .expect_err("a relative override must not be used");

    assert!(matches!(
        resolution_error,
        MlxRuntimeError::InvalidMetallibPath { description }
            if description.contains("path must be absolute")
    ));
}
