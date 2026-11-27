use std::fs;
use std::path::Path;

use crate::kernel_files::{kernel_folder_files, registered_kernels, repository_root_above};

fn write_file(file_path: &Path, file_text: &str) {
    fs::create_dir_all(file_path.parent().expect("a parent")).expect("the folder is created");
    fs::write(file_path, file_text).expect("the file is written");
}

#[test]
fn should_find_only_files_inside_folders_named_metal() {
    let crate_folder = tempfile::tempdir().expect("a temporary folder");
    write_file(
        &crate_folder
            .path()
            .join("src/a/metal/astro_primitive_one.metal"),
        "// @input X float\n// @output Y float\n",
    );
    write_file(
        &crate_folder
            .path()
            .join("src/a/not_metal/astro_primitive_two.metal"),
        "",
    );
    write_file(
        &crate_folder
            .path()
            .join("src/metal/nested/astro_primitive_three.h"),
        "",
    );
    let found_files = kernel_folder_files(&crate_folder.path().join("src"));
    assert_eq!(found_files.len(), 2, "{found_files:?}");
    assert!(found_files[0].ends_with("src/a/metal/astro_primitive_one.metal"));
}

#[test]
fn should_find_nothing_under_a_folder_that_does_not_exist() {
    assert!(kernel_folder_files(Path::new("/no/such/folder/anywhere")).is_empty());
}

#[test]
fn should_resolve_headers_beside_the_kernel_before_the_shared_folder() {
    let repository = tempfile::tempdir().expect("a temporary folder");
    let kernel_folder = repository.path().join("src/metal");
    let shared_folder = repository.path().join("shared");
    write_file(&kernel_folder.join("astro_primitive_local.h"), "");
    write_file(&shared_folder.join("astro_primitive_local.h"), "");
    write_file(&shared_folder.join("astro_primitive_matrix_units.h"), "");
    write_file(&kernel_folder.join("astro_primitive_pair_body.metal"), "");
    write_file(
        &kernel_folder.join("astro_primitive_pair.metal"),
        "// @header astro_primitive_matrix_units.h astro_primitive_local.h\n// @body astro_primitive_pair_body.metal\n// @input X float\n// @output Y float\n",
    );
    let kernels = registered_kernels(
        &kernel_folder_files(repository.path()),
        "astro_primitive_",
        &shared_folder,
    )
    .expect("the kernel resolves");
    assert_eq!(
        kernels.len(),
        1,
        "the body and headers are not registered kernels"
    );
    assert_eq!(kernels[0].name, "astro_primitive_pair");
    assert_eq!(
        kernels[0].header_files[0],
        shared_folder.join("astro_primitive_matrix_units.h")
    );
    assert_eq!(
        kernels[0].header_files[1],
        kernel_folder.join("astro_primitive_local.h")
    );
    assert_eq!(
        kernels[0].source_files[1],
        kernel_folder.join("astro_primitive_pair_body.metal")
    );
}

#[test]
fn should_refuse_a_kernel_naming_a_missing_body() {
    let repository = tempfile::tempdir().expect("a temporary folder");
    write_file(
        &repository.path().join("metal/astro_primitive_lonely.metal"),
        "// @body astro_primitive_missing_body.metal\n// @input X float\n// @output Y float\n",
    );
    let refusal = registered_kernels(
        &kernel_folder_files(repository.path()),
        "astro_primitive_",
        repository.path(),
    )
    .expect_err("a missing body is refused");
    assert!(refusal.contains("astro_primitive_missing_body.metal"));
}

#[test]
fn should_refuse_a_misnamed_file() {
    let repository = tempfile::tempdir().expect("a temporary folder");
    write_file(
        &repository.path().join("metal/sorted_expert_matmul.metal"),
        "// @input X float\n// @output Y float\n",
    );
    let refusal = registered_kernels(
        &kernel_folder_files(repository.path()),
        "astro_primitive_",
        repository.path(),
    )
    .expect_err("a name without the prefix is refused");
    assert!(refusal.contains("astro_primitive_"));
}

#[test]
fn should_find_the_workspace_root_above_a_crate() {
    let repository = tempfile::tempdir().expect("a temporary folder");
    write_file(
        &repository.path().join("Cargo.toml"),
        "[workspace]\nmembers = []\n",
    );
    write_file(
        &repository.path().join("crates/a/Cargo.toml"),
        "[package]\n",
    );
    assert_eq!(
        repository_root_above(&repository.path().join("crates/a")).as_deref(),
        Some(repository.path())
    );
    assert_eq!(repository_root_above(Path::new("/")), None);
}

#[test]
fn should_place_prologues_before_the_kernel_file_and_bodies_after_it() {
    let repository = tempfile::tempdir().expect("a temporary folder");
    let kernel_folder = repository.path().join("metal");
    write_file(&kernel_folder.join("astro_primitive_start_body.metal"), "");
    write_file(&kernel_folder.join("astro_primitive_end_body.metal"), "");
    write_file(
        &kernel_folder.join("astro_primitive_sum.metal"),
        "// @prologue astro_primitive_start_body.metal\n// @body astro_primitive_end_body.metal\n// @input X float\n// @output Y float\n",
    );
    let kernels = registered_kernels(
        &kernel_folder_files(repository.path()),
        "astro_primitive_",
        repository.path(),
    )
    .expect("the kernel resolves");
    assert_eq!(
        kernels[0].source_files,
        [
            kernel_folder.join("astro_primitive_start_body.metal"),
            kernel_folder.join("astro_primitive_sum.metal"),
            kernel_folder.join("astro_primitive_end_body.metal"),
        ]
    );
}
