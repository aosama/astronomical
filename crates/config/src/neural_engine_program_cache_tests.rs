use std::fs;
use std::path::{Path, PathBuf};

use super::*;

fn write_set(root: &Path, model: &str, revision: &str, set: &str, programs: usize) -> PathBuf {
    let set_directory = root.join(model).join(revision).join(set);
    for program_index in 0..programs {
        let program = set_directory.join(format!("layer-{program_index}.mlmodelc"));
        fs::create_dir_all(&program).unwrap();
        fs::write(program.join("weight.bin"), [0_u8; 100]).unwrap();
    }
    fs::create_dir_all(&set_directory).unwrap();
    set_directory
}

#[test]
fn should_place_a_models_programs_by_id_and_revision() {
    assert_eq!(
        neural_engine_program_cache_directory_for_model(
            Path::new("/fictional/programs"),
            "fictional-model",
            "revision-1",
        ),
        PathBuf::from("/fictional/programs/fictional-model/revision-1")
    );
}

#[test]
fn should_list_every_set_with_its_size_and_completeness() {
    let root = tempfile::tempdir().unwrap();
    let finished = write_set(root.path(), "model-b", "rev", "mlp-v1-960-of-2048", 2);
    fs::write(
        finished.join(NEURAL_ENGINE_PROGRAM_MANIFEST_FILE_NAME),
        "{}",
    )
    .unwrap();
    write_set(root.path(), "model-a", "rev", "output-gate-v1-2048", 1);

    let sets = list_neural_engine_program_sets(root.path());

    assert_eq!(sets.len(), 2);
    assert_eq!(sets[0].model_id, "model-a", "sorted by model");
    assert_eq!(sets[0].program_count, 1);
    assert!(!sets[0].is_complete);
    assert_eq!(sets[1].set_name, "mlp-v1-960-of-2048");
    assert_eq!(sets[1].program_count, 2);
    assert_eq!(sets[1].byte_count, 202, "the programs and the manifest");
    assert!(sets[1].is_complete);
}

#[test]
fn should_list_nothing_for_a_directory_that_does_not_exist_yet() {
    let root = tempfile::tempdir().unwrap();
    assert!(list_neural_engine_program_sets(&root.path().join("absent")).is_empty());
}

#[test]
fn should_ignore_stray_files_at_every_level() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join(".DS_Store"), "x").unwrap();
    let set_directory = write_set(root.path(), "model", "rev", "set", 0);
    fs::write(set_directory.join("notes.txt"), "").unwrap();
    let sets = list_neural_engine_program_sets(root.path());
    assert_eq!(sets.len(), 1);
    assert_eq!(sets[0].program_count, 0);
}

#[test]
fn should_count_a_missing_directory_as_empty() {
    let root = tempfile::tempdir().unwrap();
    assert_eq!(directory_byte_count(&root.path().join("absent")), 0);
}
