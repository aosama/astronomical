use std::path::{Path, PathBuf};

use super::neural_engine_program_cache_directory_for_model;

#[test]
fn should_place_programs_beside_the_prompt_cache_per_model_revision() {
    let directory = neural_engine_program_cache_directory_for_model(
        Path::new("/fictional/instance/cache"),
        "fictional-model",
        "revision-1",
    );

    assert_eq!(
        directory,
        Some(PathBuf::from(
            "/fictional/instance/neural-engine-programs/fictional-model/revision-1"
        ))
    );
}

#[test]
fn should_never_place_programs_inside_the_prompt_cache() {
    let prompt_cache_root = Path::new("/fictional/instance/cache");

    let directory =
        neural_engine_program_cache_directory_for_model(prompt_cache_root, "model", "revision")
            .expect("the root has a parent");

    assert!(!directory.starts_with(prompt_cache_root));
}

#[test]
fn should_leave_programs_uncached_when_the_prompt_cache_root_has_no_parent() {
    assert_eq!(
        neural_engine_program_cache_directory_for_model(Path::new("/"), "model", "revision"),
        None
    );
    assert_eq!(
        neural_engine_program_cache_directory_for_model(
            Path::new("prompt-cache"),
            "model",
            "revision"
        ),
        None
    );
}
