use super::cuda_jit_header_directories;

fn existing_folder_text(folder: &tempfile::TempDir) -> String {
    folder.path().display().to_string()
}

// --- positive ---

#[test]
fn should_accept_two_existing_header_folders() {
    let bundled_headers_folder = tempfile::tempdir().unwrap();
    let toolkit_include_folder = tempfile::tempdir().unwrap();
    let bundled_headers_text = existing_folder_text(&bundled_headers_folder);
    let toolkit_include_text = existing_folder_text(&toolkit_include_folder);

    let header_directories =
        cuda_jit_header_directories(Some(&bundled_headers_text), Some(&toolkit_include_text))
            .unwrap()
            .expect("a CUDA build names both folders");

    assert_eq!(
        header_directories
            .bundled_headers_directory
            .to_str()
            .unwrap(),
        bundled_headers_text
    );
    assert_eq!(
        header_directories
            .cuda_toolkit_include_directory
            .to_str()
            .unwrap(),
        toolkit_include_text
    );
}

// --- edge / boundary ---

#[test]
fn should_change_nothing_on_a_build_without_the_cuda_backend() {
    assert_eq!(cuda_jit_header_directories(None, None), Ok(None));
}

// --- negative ---

#[test]
fn should_refuse_a_header_folder_that_no_longer_exists() {
    let toolkit_include_folder = tempfile::tempdir().unwrap();
    let removed_folder = tempfile::tempdir().unwrap();
    let removed_folder_text = existing_folder_text(&removed_folder);
    drop(removed_folder);

    let refusal = cuda_jit_header_directories(
        Some(&removed_folder_text),
        Some(&existing_folder_text(&toolkit_include_folder)),
    )
    .expect_err("a removed store entry must be reported at start-up");

    assert!(
        refusal.contains("MLX's run-time compiler headers"),
        "{refusal}"
    );
    assert!(refusal.contains(&removed_folder_text), "{refusal}");
}

#[test]
fn should_refuse_a_header_path_that_is_a_file() {
    let bundled_headers_folder = tempfile::tempdir().unwrap();
    let toolkit_include_file = tempfile::NamedTempFile::new().unwrap();
    let toolkit_include_file_text = toolkit_include_file.path().display().to_string();

    let refusal = cuda_jit_header_directories(
        Some(&existing_folder_text(&bundled_headers_folder)),
        Some(&toolkit_include_file_text),
    )
    .expect_err("a file is not a header folder");

    assert!(refusal.contains("the CUDA toolkit headers"), "{refusal}");
}

#[test]
fn should_refuse_a_build_that_recorded_only_one_folder() {
    let bundled_headers_folder = tempfile::tempdir().unwrap();
    let bundled_headers_text = existing_folder_text(&bundled_headers_folder);

    assert!(cuda_jit_header_directories(Some(&bundled_headers_text), None).is_err());
    assert!(cuda_jit_header_directories(None, Some(&bundled_headers_text)).is_err());
}
