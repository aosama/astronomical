use std::fs;

use astronomical_neural_engine::program_shape::ProgramShape;

use super::{CoreMlProgramBuilder, ProgramSetCache};
use crate::qwen3_5_resident::neural_engine_program_placement_record::{
    PlacementCheck, placement_check_for, record_placement_verified,
};
use crate::qwen3_5_resident::neural_engine_program_preparation::{
    NeuralEngineProgramFamily, ProgramBuildJob, ProgramBuildOutcome, ProgramBuilder,
    ProgramIdentity,
};

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
fn should_record_a_finished_set_with_its_identity_and_size() {
    let root = tempfile::tempdir().expect("a scratch directory");
    let revision_directory = root.path().join("fictional-model/revision-1");
    let cache =
        ProgramSetCache::open(&revision_directory, "mlp", "v1-960-of-2048").expect("the set opens");
    fs::create_dir_all(cache.program_path(0)).expect("a program folder");
    fs::write(cache.program_path(0).join("weight.bin"), [0_u8; 10]).expect("a weight file");

    cache.record_finished_set(1);

    let manifest_bytes = fs::read(revision_directory.join("mlp-v1-960-of-2048/programs.json"))
        .expect("the manifest exists");
    let manifest: astronomical_config::NeuralEngineProgramSetManifest =
        serde_json::from_slice(&manifest_bytes).expect("the manifest parses");
    assert_eq!(manifest.model_id, "fictional-model");
    assert_eq!(manifest.model_revision, "revision-1");
    assert_eq!(manifest.set_name, "mlp-v1-960-of-2048");
    assert_eq!(manifest.program_count, 1);
    assert!(manifest.byte_count >= 10);
    assert!(manifest.built_with.starts_with("Astronomical "));
    assert!(
        !revision_directory
            .join("mlp-v1-960-of-2048/.programs.json.incomplete")
            .exists()
    );
}

#[test]
fn should_rewrite_the_record_only_when_the_set_changed() {
    let model_cache_directory = tempfile::tempdir().expect("a scratch directory");
    let cache = ProgramSetCache::open(model_cache_directory.path(), "mlp", "v1-960-of-2048")
        .expect("the set opens");
    let manifest_path = model_cache_directory
        .path()
        .join("mlp-v1-960-of-2048/programs.json");
    let recorded = r#"{"model_id":"kept","model_revision":"r","set_name":"s","program_count":2,"byte_count":1,"macos_version":null,"built_with":"b"}"#;
    fs::write(&manifest_path, recorded).expect("a manifest exists");

    cache.record_finished_set(2);
    assert_eq!(fs::read_to_string(&manifest_path).unwrap(), recorded);

    cache.record_finished_set(3);
    assert!(
        fs::read_to_string(&manifest_path)
            .unwrap()
            .contains("\"program_count\": 3")
    );
}

#[test]
fn should_replace_a_damaged_record() {
    let model_cache_directory = tempfile::tempdir().expect("a scratch directory");
    let cache = ProgramSetCache::open(model_cache_directory.path(), "mlp", "v1-960-of-2048")
        .expect("the set opens");
    let manifest_path = model_cache_directory
        .path()
        .join("mlp-v1-960-of-2048/programs.json");
    fs::write(&manifest_path, "not json").expect("a damaged manifest exists");

    cache.record_finished_set(1);

    assert!(
        fs::read_to_string(&manifest_path)
            .unwrap()
            .contains("program_count")
    );
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

const PLACEMENT_ENVIRONMENT: &str = "macOS 27.0.1 (26A434) on Mac99,1";

fn kept_build_job_without_a_package(kept_program_path: &std::path::Path) -> ProgramBuildJob {
    ProgramBuildJob {
        identity: ProgramIdentity {
            family: NeuralEngineProgramFamily::Mlp,
            decoder_layer_index: 7,
        },
        shape: ProgramShape::new(64, 64, 64).expect("a valid shape"),
        kept_program_path: Some(kept_program_path.to_path_buf()),
        package: None,
    }
}

#[test]
fn should_ask_for_a_package_when_the_kept_program_is_missing() {
    let set_directory = tempfile::tempdir().expect("a scratch directory");
    let kept_program_path = set_directory.path().join("layer-7.mlmodelc");

    let build_outcome =
        CoreMlProgramBuilder.build(&kept_build_job_without_a_package(&kept_program_path));

    assert!(matches!(
        build_outcome,
        ProgramBuildOutcome::NeedsUncachedPackage
    ));
}

#[test]
fn should_forget_a_record_left_by_a_program_that_is_gone_before_compiling_again() {
    let set_directory = tempfile::tempdir().expect("a scratch directory");
    let kept_program_path = set_directory.path().join("layer-7.mlmodelc");
    record_placement_verified(&kept_program_path, PLACEMENT_ENVIRONMENT);

    let _build_outcome =
        CoreMlProgramBuilder.build(&kept_build_job_without_a_package(&kept_program_path));

    assert_eq!(
        placement_check_for(&kept_program_path, Some(PLACEMENT_ENVIRONMENT)),
        PlacementCheck::Required
    );
}

#[test]
fn should_remove_a_kept_program_that_cannot_load_together_with_its_record() {
    let set_directory = tempfile::tempdir().expect("a scratch directory");
    let kept_program_path = set_directory.path().join("layer-7.mlmodelc");
    fs::create_dir_all(&kept_program_path).expect("an empty program folder");
    record_placement_verified(&kept_program_path, PLACEMENT_ENVIRONMENT);

    let build_outcome =
        CoreMlProgramBuilder.build(&kept_build_job_without_a_package(&kept_program_path));

    assert!(matches!(
        build_outcome,
        ProgramBuildOutcome::NeedsUncachedPackage
    ));
    assert!(!kept_program_path.exists());
    assert_eq!(
        placement_check_for(&kept_program_path, Some(PLACEMENT_ENVIRONMENT)),
        PlacementCheck::Required
    );
}
