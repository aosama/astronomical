use std::fs;

use super::ProgramSetCache;

#[test]
fn should_name_each_layers_program_inside_the_set() {
    let model_cache_directory = tempfile::tempdir().expect("a scratch directory");

    let cache = ProgramSetCache::open(model_cache_directory.path(), "mlp", "v1-960-of-2048")
        .expect("the set opens");

    assert_eq!(
        cache.program_path(7),
        model_cache_directory
            .path()
            .join("mlp-v1-960-of-2048")
            .join("layer-7.mlmodelc")
    );
    assert!(
        model_cache_directory
            .path()
            .join("mlp-v1-960-of-2048")
            .is_dir()
    );
}

#[test]
fn should_remove_outdated_sets_of_the_same_family_only() {
    let model_cache_directory = tempfile::tempdir().expect("a scratch directory");
    let outdated_set = model_cache_directory.path().join("mlp-v1-448-of-1024");
    let other_family_set = model_cache_directory.path().join("output-gate-v1-2048");
    fs::create_dir_all(outdated_set.join("layer-0.mlmodelc")).expect("the outdated set exists");
    fs::create_dir_all(&other_family_set).expect("the other family's set exists");

    ProgramSetCache::open(model_cache_directory.path(), "mlp", "v1-960-of-2048")
        .expect("the set opens");

    assert!(!outdated_set.exists());
    assert!(other_family_set.is_dir());
}

#[test]
fn should_keep_the_current_set_and_its_programs_when_reopened() {
    let model_cache_directory = tempfile::tempdir().expect("a scratch directory");
    let first_open = ProgramSetCache::open(model_cache_directory.path(), "mlp", "v1-960-of-2048")
        .expect("the set opens");
    fs::create_dir_all(first_open.program_path(0)).expect("a kept program exists");

    let reopened = ProgramSetCache::open(model_cache_directory.path(), "mlp", "v1-960-of-2048")
        .expect("the set reopens");

    assert!(reopened.program_path(0).is_dir());
}

#[test]
fn should_refuse_a_cache_directory_that_cannot_be_created() {
    let model_cache_directory = tempfile::tempdir().expect("a scratch directory");
    let blocking_file = model_cache_directory.path().join("not-a-directory");
    fs::write(&blocking_file, b"a file where a folder is needed").expect("the file writes");

    assert!(ProgramSetCache::open(&blocking_file, "mlp", "v1-960-of-2048").is_none());
}
