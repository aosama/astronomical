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
fn should_place_a_models_programs_by_id_revision_and_weights() {
    assert_eq!(
        neural_engine_program_cache_directory_for_model(
            Path::new("/fictional/programs"),
            "fictional-model",
            "revision-1",
            "abcd",
        ),
        PathBuf::from("/fictional/programs/fictional-model/revision-1-abcd")
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

fn model_folder_with_shard(shard_bytes: &[u8]) -> tempfile::TempDir {
    let model_folder = tempfile::tempdir().expect("a scratch model folder");
    std::fs::write(model_folder.path().join("model.safetensors"), shard_bytes)
        .expect("the shard writes");
    model_folder
}

#[test]
fn should_fingerprint_the_same_weights_alike() {
    let model_folder = model_folder_with_shard(b"weights");

    assert_eq!(
        super::model_weights_fingerprint(model_folder.path()),
        super::model_weights_fingerprint(model_folder.path())
    );
}

#[test]
fn should_change_the_fingerprint_when_the_weights_change() {
    let model_folder = model_folder_with_shard(b"weights");
    let before = super::model_weights_fingerprint(model_folder.path());

    std::fs::write(
        model_folder.path().join("model.safetensors"),
        b"other weights",
    )
    .expect("the shard rewrites");

    assert_ne!(
        before,
        super::model_weights_fingerprint(model_folder.path())
    );
}

#[test]
fn should_change_the_fingerprint_when_a_shard_is_added() {
    let model_folder = model_folder_with_shard(b"weights");
    let before = super::model_weights_fingerprint(model_folder.path());

    std::fs::write(model_folder.path().join("second.safetensors"), b"more")
        .expect("the shard writes");

    assert_ne!(
        before,
        super::model_weights_fingerprint(model_folder.path())
    );
}

#[test]
fn should_ignore_files_that_are_not_weights() {
    let model_folder = model_folder_with_shard(b"weights");
    let before = super::model_weights_fingerprint(model_folder.path());

    std::fs::write(model_folder.path().join("config.json"), b"{}").expect("the config writes");

    assert_eq!(
        before,
        super::model_weights_fingerprint(model_folder.path())
    );
}

#[test]
fn should_have_no_fingerprint_without_weights() {
    let model_folder = tempfile::tempdir().expect("a scratch model folder");

    assert_eq!(super::model_weights_fingerprint(model_folder.path()), None);
}

#[test]
fn should_have_no_fingerprint_for_a_missing_folder() {
    let model_folder = tempfile::tempdir().expect("a scratch model folder");

    assert_eq!(
        super::model_weights_fingerprint(&model_folder.path().join("gone")),
        None
    );
}

fn write_program_set(weights_folder: &std::path::Path) {
    let layer = weights_folder
        .join("mlp-v1-960-of-2048")
        .join("layer-0.mlmodelc");
    std::fs::create_dir_all(layer).expect("the set writes");
}

#[test]
fn should_remove_the_model_s_programs_for_other_weights_only() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let model = directory.path().join("model-a");
    let kept = model.join("rev-new");
    write_program_set(&kept);
    write_program_set(&model.join("rev-old"));
    write_program_set(&directory.path().join("model-b").join("rev-old"));

    remove_other_weights_of_model(directory.path(), "model-a", &kept);

    assert!(kept.exists());
    assert!(!model.join("rev-old").exists());
    assert!(directory.path().join("model-b").join("rev-old").exists());
}

#[test]
fn should_leave_a_folder_that_holds_no_program_sets() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let model = directory.path().join("model-a");
    let kept = model.join("rev-new");
    let unrelated = model.join("notes").join("plain-folder");
    std::fs::create_dir_all(&unrelated).expect("the folder writes");
    std::fs::create_dir_all(&kept).expect("the folder writes");

    remove_other_weights_of_model(directory.path(), "model-a", &kept);

    assert!(unrelated.exists());
}

#[test]
fn should_ignore_a_model_with_no_programs_yet() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    remove_other_weights_of_model(directory.path(), "model-a", &directory.path().join("x"));
    assert!(!directory.path().join("model-a").exists());
}

#[test]
fn should_refuse_an_id_that_is_not_one_folder_name() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let program_directory = directory.path().join("programs");
    let outside_model = directory.path().join("outside").join("rev-old");
    write_program_set(&outside_model);
    let inside_model = program_directory.join("model-a").join("rev-old");
    write_program_set(&inside_model);
    let kept = program_directory.join("kept");

    for surprising_id in ["", "..", "../outside", "model-a/..", "/"] {
        remove_other_weights_of_model(&program_directory, surprising_id, &kept);
    }

    assert!(outside_model.exists());
    assert!(inside_model.exists());
}

#[cfg(unix)]
#[test]
fn should_leave_a_linked_folder_s_contents_alone() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let linked_target = directory.path().join("elsewhere");
    write_program_set(&linked_target);
    let model = directory.path().join("programs").join("model-a");
    let kept = model.join("rev-new");
    std::fs::create_dir_all(&kept).expect("the folder writes");
    std::os::unix::fs::symlink(&linked_target, model.join("rev-linked")).expect("the link writes");

    remove_other_weights_of_model(&directory.path().join("programs"), "model-a", &kept);

    assert!(linked_target.join("mlp-v1-960-of-2048").exists());
}

#[test]
fn should_fingerprint_a_shard_as_the_first_eight_digest_bytes_in_lowercase_hex() {
    let model_folder = model_folder_with_shard(b"weights");
    let shard_file = std::fs::File::options()
        .write(true)
        .open(model_folder.path().join("model.safetensors"))
        .expect("the shard opens");
    shard_file
        .set_modified(std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000))
        .expect("the shard modification time sets");

    // SHA-256 of "model.safetensors:7:1700000000", first eight bytes: the
    // fingerprint names cache folders on disk, so its text must not drift.
    assert_eq!(
        super::model_weights_fingerprint(model_folder.path()).as_deref(),
        Some("077b99e4651cfdbc")
    );
}
