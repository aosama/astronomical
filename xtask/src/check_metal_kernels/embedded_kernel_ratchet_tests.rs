use std::fs;
use std::path::Path;

use super::embedded_kernel_ratchet::embedded_kernel_sites;

fn write_file(file_path: &Path, file_text: &str) {
    fs::create_dir_all(file_path.parent().expect("a parent")).expect("the folder is created");
    fs::write(file_path, file_text).expect("the file is written");
}

const EMBEDDED_DECLARATION: &str =
    "let kernel = MlxMetalKernel::new(\"copy\", &[\"X\"], &[\"Y\"], SOURCE)?;\n";

#[test]
fn should_name_every_line_that_declares_a_kernel_from_text() {
    let repository = tempfile::tempdir().expect("a temporary folder");
    write_file(
        &repository
            .path()
            .join("crates/families/text/qwen35/src/rotation.rs"),
        &format!("fn a() {{}}\n{EMBEDDED_DECLARATION}"),
    );
    write_file(
        &repository
            .path()
            .join("verification/acceptance/tests/other.rs"),
        "MlxMetalKernel::new_with_header(\"a\", &[], &[], \"\", \"\");\n",
    );
    assert_eq!(
        embedded_kernel_sites(repository.path()),
        [
            "crates/families/text/qwen35/src/rotation.rs:2",
            "verification/acceptance/tests/other.rs:1"
        ]
    );
}

#[test]
fn should_allow_the_one_declarer_and_the_mlx_mechanism_contracts() {
    let repository = tempfile::tempdir().expect("a temporary folder");
    write_file(
        &repository
            .path()
            .join("crates/engine/model-primitives/src/metal_kernel_declaration.rs"),
        EMBEDDED_DECLARATION,
    );
    write_file(
        &repository.path().join(
            "verification/acceptance/tests/gpu_contracts/runtime_mlx/direct_mlx/metal_kernel.rs",
        ),
        EMBEDDED_DECLARATION,
    );
    assert!(embedded_kernel_sites(repository.path()).is_empty());
}

#[test]
fn should_ignore_comments_build_output_and_other_files() {
    let repository = tempfile::tempdir().expect("a temporary folder");
    write_file(
        &repository.path().join("crates/a/src/lib.rs"),
        "// MlxMetalKernel::new( in a comment\n",
    );
    write_file(
        &repository.path().join("crates/a/target/debug/generated.rs"),
        EMBEDDED_DECLARATION,
    );
    write_file(
        &repository.path().join("crates/a/README.md"),
        EMBEDDED_DECLARATION,
    );
    assert!(embedded_kernel_sites(repository.path()).is_empty());
}
