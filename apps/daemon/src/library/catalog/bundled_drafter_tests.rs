use ipc_protocol::DraftModelGeneration;

use super::{bundled_drafter_generation, drafter_generation_of_configuration};

#[test]
fn should_recognize_a_first_generation_dflash_drafter() {
    assert_eq!(
        drafter_generation_of_configuration(
            r#"{"architectures":["DFlashDraftModel"],"dflash_config":{"target_layer_ids":[1,6]}}"#
        ),
        Some(DraftModelGeneration::Dflash)
    );
}

#[test]
fn should_recognize_a_dflash2_drafter() {
    assert_eq!(
        drafter_generation_of_configuration(
            r#"{"architectures":["DFlash2DraftModel"],"dflash_config":{}}"#
        ),
        Some(DraftModelGeneration::Dflash2)
    );
}

#[test]
fn should_not_count_a_model_without_a_dflash_configuration() {
    assert_eq!(
        drafter_generation_of_configuration(r#"{"architectures":["DFlashDraftModel"]}"#),
        None
    );
    assert_eq!(
        drafter_generation_of_configuration(
            r#"{"architectures":["Qwen3_5MoeForConditionalGeneration"],"dflash_config":{}}"#
        ),
        None
    );
    assert_eq!(drafter_generation_of_configuration("not json"), None);
}

#[test]
fn should_find_no_drafter_in_a_model_folder_without_one() {
    let model_directory = std::env::temp_dir().join(format!(
        "astronomical-bundled-drafter-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&model_directory).expect("the test folder is writable");
    assert_eq!(bundled_drafter_generation(&model_directory), None);
    std::fs::create_dir_all(model_directory.join("drafter")).expect("the test folder is writable");
    std::fs::write(
        model_directory.join("drafter/config.json"),
        r#"{"architectures":["DFlashDraftModel"],"dflash_config":{}}"#,
    )
    .expect("the test folder is writable");
    assert_eq!(
        bundled_drafter_generation(&model_directory),
        Some(DraftModelGeneration::Dflash)
    );
    let _ = std::fs::remove_dir_all(&model_directory);
}

#[test]
fn should_find_a_community_pack_drafter_in_its_dflash_folder() {
    let model_directory = std::env::temp_dir().join(format!(
        "astronomical-community-drafter-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(model_directory.join("dflash")).expect("the test folder is writable");
    std::fs::write(
        model_directory.join("dflash/config.json"),
        r#"{"architectures":["DFlash2DraftModel"],"dflash_config":{}}"#,
    )
    .expect("the test folder is writable");
    assert_eq!(
        bundled_drafter_generation(&model_directory),
        Some(DraftModelGeneration::Dflash2)
    );
    let _ = std::fs::remove_dir_all(&model_directory);
}
