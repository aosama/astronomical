use std::path::PathBuf;

use super::{
    ModelFolderFingerprint, ModelFolderWatchStep, model_folder_fingerprint, model_folder_watch_step,
};

fn fresh_library(test_name: &str) -> PathBuf {
    let library_root = std::env::temp_dir().join(format!(
        "astronomical-model-folder-watch-{test_name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&library_root);
    std::fs::create_dir_all(library_root.join("example-org/example-qwen"))
        .expect("the test library is writable");
    library_root
}

#[test]
fn should_keep_the_same_fingerprint_when_nothing_changed() {
    let library_root = fresh_library("unchanged");
    let roots = vec![library_root.clone()];
    assert_eq!(
        model_folder_fingerprint(&roots),
        model_folder_fingerprint(&roots)
    );
    let _ = std::fs::remove_dir_all(&library_root);
}

#[test]
fn should_change_the_fingerprint_when_a_model_folder_is_copied_in() {
    let library_root = fresh_library("copied-in");
    let roots = vec![library_root.clone()];
    let before_copy = model_folder_fingerprint(&roots);
    std::fs::create_dir_all(library_root.join("example-org/example-ornith"))
        .expect("the test library is writable");
    assert_ne!(before_copy, model_folder_fingerprint(&roots));
    let _ = std::fs::remove_dir_all(&library_root);
}

#[test]
fn should_change_the_fingerprint_when_a_drafter_folder_appears_in_a_model() {
    let library_root = fresh_library("drafter-added");
    let roots = vec![library_root.clone()];
    let before_drafter = model_folder_fingerprint(&roots);
    std::fs::create_dir_all(library_root.join("example-org/example-qwen/drafter"))
        .expect("the test library is writable");
    assert_ne!(before_drafter, model_folder_fingerprint(&roots));
    let _ = std::fs::remove_dir_all(&library_root);
}

#[test]
fn should_give_an_empty_fingerprint_for_a_missing_library() {
    let missing_root = std::env::temp_dir().join("astronomical-model-folder-watch-missing-root");
    assert_eq!(
        model_folder_fingerprint(&[missing_root]),
        ModelFolderFingerprint::default()
    );
}

#[test]
fn should_ignore_a_hidden_folder_such_as_a_download_in_progress() {
    let library_root = fresh_library("hidden");
    let roots = vec![library_root.clone()];
    std::fs::create_dir_all(library_root.join(".incomplete"))
        .expect("the test library is writable");
    let before_download = model_folder_fingerprint(&roots);
    std::fs::create_dir_all(library_root.join(".incomplete/example-org/partial-model"))
        .expect("the test library is writable");
    assert_eq!(before_download, model_folder_fingerprint(&roots));
    let _ = std::fs::remove_dir_all(&library_root);
}

#[test]
fn should_change_the_fingerprint_while_a_weight_file_is_still_growing() {
    let library_root = fresh_library("growing");
    let roots = vec![library_root.clone()];
    let weights_path = library_root.join("example-org/example-qwen/model.safetensors");
    std::fs::write(&weights_path, [0_u8; 16]).expect("the test library is writable");
    let partly_copied = model_folder_fingerprint(&roots);
    std::fs::write(&weights_path, [0_u8; 32]).expect("the test library is writable");
    assert_ne!(partly_copied, model_folder_fingerprint(&roots));
    let _ = std::fs::remove_dir_all(&library_root);
}

#[test]
fn should_ignore_download_progress_saved_in_the_library_root() {
    let library_root = fresh_library("download-progress");
    let roots = vec![library_root.clone()];
    let before_progress = model_folder_fingerprint(&roots);
    std::fs::write(library_root.join(".download-job.json"), "{}")
        .expect("the test library is writable");
    assert_eq!(before_progress, model_folder_fingerprint(&roots));
    let _ = std::fs::remove_dir_all(&library_root);
}

fn fingerprint_of(library_root: &std::path::Path, model_name: &str) -> ModelFolderFingerprint {
    std::fs::create_dir_all(library_root.join("example-org").join(model_name))
        .expect("the test library is writable");
    model_folder_fingerprint(&[library_root.to_path_buf()])
}

#[test]
fn should_wait_a_tick_before_acting_on_a_change() {
    let library_root = fresh_library("settling");
    let acted = model_folder_fingerprint(std::slice::from_ref(&library_root));
    let changed = fingerprint_of(&library_root, "example-ornith");
    assert_eq!(
        model_folder_watch_step(&acted, &acted, &acted),
        ModelFolderWatchStep::Idle
    );
    assert_eq!(
        model_folder_watch_step(&acted, &acted, &changed),
        ModelFolderWatchStep::WaitForSettling
    );
    assert_eq!(
        model_folder_watch_step(&acted, &changed, &changed),
        ModelFolderWatchStep::Rescan
    );
    let _ = std::fs::remove_dir_all(&library_root);
}

#[test]
fn should_keep_waiting_while_a_copy_keeps_changing() {
    let library_root = fresh_library("still-copying");
    let acted = model_folder_fingerprint(std::slice::from_ref(&library_root));
    let first_tick = fingerprint_of(&library_root, "example-ornith");
    let second_tick = fingerprint_of(&library_root, "example-ornith-vision");
    assert_eq!(
        model_folder_watch_step(&acted, &first_tick, &second_tick),
        ModelFolderWatchStep::WaitForSettling
    );
    let _ = std::fs::remove_dir_all(&library_root);
}
