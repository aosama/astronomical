use std::fs;
use std::path::Path;

use super::kernel_inventory::kernel_inventory;

const COVERED_BLOCK: &str = "// @input X float\n// @output Y float\n// @contract direct_mlx::copy\n// @probe none, a test kernel\n";

fn write_file(file_path: &Path, file_text: &str) {
    fs::create_dir_all(file_path.parent().expect("a parent")).expect("the folder is created");
    fs::write(file_path, file_text).expect("the file is written");
}

fn repository_with_crates() -> tempfile::TempDir {
    let repository = tempfile::tempdir().expect("a temporary folder");
    for crate_folder in [
        "crates/engine/model-primitives",
        "crates/families/text/qwen35",
        "crates/engine/model-serving",
    ] {
        write_file(
            &repository.path().join(crate_folder).join("Cargo.toml"),
            "[package]\n",
        );
    }
    repository
}

#[test]
fn should_read_a_well_named_kernel_of_each_scope() {
    let repository = repository_with_crates();
    let root = repository.path();
    write_file(
        &root
            .join("crates/engine/model-primitives/src/metal_common/astro_primitive_matrix_units.h"),
        "",
    );
    write_file(
        &root.join("crates/engine/model-primitives/src/top_k/metal/astro_primitive_top_k.metal"),
        &format!("// @header astro_primitive_matrix_units.h\n{COVERED_BLOCK}"),
    );
    write_file(
        &root.join("crates/families/text/qwen35/src/metal/astro_family_qwen35_gated_delta.metal"),
        COVERED_BLOCK,
    );
    let inventory = kernel_inventory(root);
    assert!(
        inventory.violations.is_empty(),
        "{:?}",
        inventory.violations
    );
    assert_eq!(inventory.registered_kernels.len(), 2);
}

#[test]
fn should_refuse_a_family_kernel_named_as_a_primitive() {
    let repository = repository_with_crates();
    write_file(
        &repository
            .path()
            .join("crates/families/text/qwen35/src/metal/astro_primitive_gated_delta.metal"),
        COVERED_BLOCK,
    );
    let inventory = kernel_inventory(repository.path());
    assert_eq!(inventory.violations.len(), 1, "{:?}", inventory.violations);
    assert!(inventory.violations[0].contains("astro_family_qwen35_"));
}

#[test]
fn should_refuse_metal_files_in_a_crate_that_may_own_none() {
    let repository = repository_with_crates();
    write_file(
        &repository
            .path()
            .join("crates/engine/model-serving/src/metal/astro_primitive_stray.metal"),
        COVERED_BLOCK,
    );
    let inventory = kernel_inventory(repository.path());
    assert!(inventory.violations[0].contains("model-primitives, a family crate or a format crate"));
}

#[test]
fn should_refuse_a_kernel_without_contract_and_probe_lines() {
    let repository = repository_with_crates();
    write_file(
        &repository
            .path()
            .join("crates/engine/model-primitives/src/metal/astro_primitive_bare.metal"),
        "// @input X float\n// @output Y float\n",
    );
    let inventory = kernel_inventory(repository.path());
    assert_eq!(inventory.violations.len(), 2, "{:?}", inventory.violations);
    assert!(inventory.violations[0].contains("@contract"));
    assert!(inventory.violations[1].contains("@probe"));
}

#[test]
fn should_refuse_a_helper_no_kernel_uses() {
    let repository = repository_with_crates();
    write_file(
        &repository
            .path()
            .join("crates/engine/model-primitives/src/metal_common/astro_primitive_orphan.h"),
        "",
    );
    let inventory = kernel_inventory(repository.path());
    assert_eq!(inventory.violations.len(), 1, "{:?}", inventory.violations);
    assert!(inventory.violations[0].contains("astro_primitive_orphan.h"));
}

#[test]
fn should_find_nothing_wrong_in_a_repository_without_kernels() {
    let repository = repository_with_crates();
    let inventory = kernel_inventory(repository.path());
    assert!(inventory.violations.is_empty());
    assert!(inventory.registered_kernels.is_empty());
}
