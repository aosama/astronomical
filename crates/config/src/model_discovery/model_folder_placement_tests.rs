//! Where a newly written model folder is found by discovery, and where not.

use std::fs;
use std::path::{Path, PathBuf};

use super::*;

/// A canonical, existing models folder to place new folders in.
fn model_root() -> (tempfile::TempDir, PathBuf) {
    let root_directory = tempfile::tempdir().unwrap();
    let canonical_root = fs::canonicalize(root_directory.path()).unwrap();
    (root_directory, canonical_root)
}

fn make_model_folder(folder: &Path) {
    fs::create_dir_all(folder).unwrap();
    fs::write(folder.join("config.json"), "{}").unwrap();
}

#[test]
fn should_find_a_folder_directly_inside_the_models_folder() {
    let (_root_directory, root) = model_root();

    assert!(is_discoverable_model_folder(
        &root,
        &root.join("laya-moods")
    ));
}

#[test]
fn should_find_a_folder_up_to_four_levels_down_even_before_its_parents_exist() {
    let (_root_directory, root) = model_root();

    assert!(is_discoverable_model_folder(
        &root,
        &root.join("calibrated/romeo/juliet/laya-moods")
    ));
}

#[test]
fn should_not_find_a_folder_five_levels_down() {
    let (_root_directory, root) = model_root();

    assert!(!is_discoverable_model_folder(
        &root,
        &root.join("one/two/three/four/laya-moods")
    ));
}

#[test]
fn should_not_find_a_folder_outside_the_models_folder() {
    let (_root_directory, root) = model_root();
    let (_other_directory, elsewhere) = model_root();

    assert!(!is_discoverable_model_folder(
        &root,
        &elsewhere.join("laya-moods")
    ));
    assert!(!is_discoverable_model_folder(&root, &root));
}

#[test]
fn should_not_find_a_folder_inside_another_model_folder() {
    let (_root_directory, root) = model_root();
    make_model_folder(&root.join("laya-MLX-8bit"));

    assert!(!is_discoverable_model_folder(
        &root,
        &root.join("laya-MLX-8bit/laya-moods")
    ));
}

#[test]
fn should_not_find_a_folder_under_a_hidden_folder_or_a_hugging_face_cache_entry() {
    let (_root_directory, root) = model_root();

    assert!(!is_discoverable_model_folder(
        &root,
        &root.join(".drafts/laya-moods")
    ));
    assert!(!is_discoverable_model_folder(
        &root,
        &root.join(".laya-moods")
    ));
    assert!(!is_discoverable_model_folder(
        &root,
        &root.join("models--cavi-ai--laya/laya-moods")
    ));
}

#[test]
fn should_not_find_anything_below_a_models_folder_that_is_itself_a_model() {
    let (_root_directory, root) = model_root();
    make_model_folder(&root);

    assert!(!is_discoverable_model_folder(
        &root,
        &root.join("laya-moods")
    ));
}

#[test]
fn should_not_find_a_folder_named_through_a_parent_step() {
    let (_root_directory, root) = model_root();

    assert!(!is_discoverable_model_folder(
        &root,
        &root.join("calibrated/../laya-moods")
    ));
}
