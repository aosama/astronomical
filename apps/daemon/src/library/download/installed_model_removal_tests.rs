use std::fs;
use std::path::Path;

use super::{InstalledModelRemovalError, installed_model_folder_in};

const MODEL_ID: &str = "astronomical-test/example-qwen";
const OWN_PROVENANCE: &str =
    r#"{"schema_version":1,"provider_model_id":"astronomical-test/example-qwen","revision":"r"}"#;

fn write_provenance(model_folder: &Path, provenance_text: &str) {
    fs::create_dir_all(model_folder).expect("the model folder should be created");
    fs::write(
        model_folder.join(".astronomical-library-provenance.json"),
        provenance_text,
    )
    .expect("the provenance file should be written");
}

fn models_directory_fixture() -> (tempfile::TempDir, std::path::PathBuf) {
    let test_directory = tempfile::tempdir().expect("temporary directory should be available");
    let models_directory = test_directory.path().join("models");
    fs::create_dir_all(&models_directory).expect("the models folder should be created");
    (test_directory, models_directory)
}

#[test]
fn should_return_the_folder_whose_provenance_names_the_model() {
    let (_test_directory, models_directory) = models_directory_fixture();
    write_provenance(&models_directory.join(MODEL_ID), OWN_PROVENANCE);

    assert_eq!(
        installed_model_folder_in(&models_directory, MODEL_ID).ok(),
        Some(models_directory.join(MODEL_ID))
    );
}

#[test]
fn should_report_a_model_with_no_folder_as_not_on_this_mac() {
    let (_test_directory, models_directory) = models_directory_fixture();

    assert!(matches!(
        installed_model_folder_in(&models_directory, MODEL_ID),
        Err(InstalledModelRemovalError::NotOnThisMac)
    ));
}

#[test]
fn should_refuse_a_folder_without_matching_provenance() {
    let (_test_directory, models_directory) = models_directory_fixture();
    let model_folder = models_directory.join(MODEL_ID);
    for provenance_text in [
        r#"{"schema_version":1,"provider_model_id":"astronomical-test/other-qwen","revision":"r"}"#,
        "not json",
    ] {
        write_provenance(&model_folder, provenance_text);
        assert!(matches!(
            installed_model_folder_in(&models_directory, MODEL_ID),
            Err(InstalledModelRemovalError::NotInstalledByDownloadModels)
        ));
    }
    fs::remove_file(model_folder.join(".astronomical-library-provenance.json"))
        .expect("the provenance file should be removed");
    assert!(matches!(
        installed_model_folder_in(&models_directory, MODEL_ID),
        Err(InstalledModelRemovalError::NotInstalledByDownloadModels)
    ));
}

#[test]
fn should_refuse_a_symlinked_model_folder() {
    let (test_directory, models_directory) = models_directory_fixture();
    let outside_folder = test_directory.path().join("outside");
    write_provenance(&outside_folder, OWN_PROVENANCE);
    fs::create_dir_all(models_directory.join("astronomical-test"))
        .expect("the publisher folder should be created");
    std::os::unix::fs::symlink(&outside_folder, models_directory.join(MODEL_ID))
        .expect("the symlink should be created");

    assert!(matches!(
        installed_model_folder_in(&models_directory, MODEL_ID),
        Err(InstalledModelRemovalError::UnsafeFolder)
    ));
}

/// The case the last-component check alone missed: the publisher folder is a
/// link, so `<models>/<publisher>/<model>` resolves outside the models folder.
#[test]
fn should_refuse_a_model_folder_under_a_symlinked_publisher_folder() {
    let (test_directory, models_directory) = models_directory_fixture();
    let outside_publisher = test_directory.path().join("elsewhere");
    write_provenance(&outside_publisher.join("example-qwen"), OWN_PROVENANCE);
    std::os::unix::fs::symlink(
        &outside_publisher,
        models_directory.join("astronomical-test"),
    )
    .expect("the publisher symlink should be created");

    assert!(matches!(
        installed_model_folder_in(&models_directory, MODEL_ID),
        Err(InstalledModelRemovalError::UnsafeFolder)
    ));
    assert!(outside_publisher.join("example-qwen").is_dir());
}

#[test]
fn should_accept_a_models_folder_that_is_itself_reached_through_a_link() {
    let (test_directory, models_directory) = models_directory_fixture();
    write_provenance(&models_directory.join(MODEL_ID), OWN_PROVENANCE);
    let linked_models_directory = test_directory.path().join("linked-models");
    std::os::unix::fs::symlink(&models_directory, &linked_models_directory)
        .expect("the models symlink should be created");

    assert_eq!(
        installed_model_folder_in(&linked_models_directory, MODEL_ID).ok(),
        Some(linked_models_directory.join(MODEL_ID))
    );
}

#[test]
fn should_refuse_a_plain_file_where_the_model_folder_should_be() {
    let (_test_directory, models_directory) = models_directory_fixture();
    fs::create_dir_all(models_directory.join("astronomical-test"))
        .expect("the publisher folder should be created");
    fs::write(models_directory.join(MODEL_ID), "not a folder")
        .expect("the stand-in file should be written");

    assert!(matches!(
        installed_model_folder_in(&models_directory, MODEL_ID),
        Err(InstalledModelRemovalError::UnsafeFolder)
    ));
}

#[test]
fn should_refuse_a_folder_whose_provenance_file_is_a_link() {
    let (test_directory, models_directory) = models_directory_fixture();
    let model_folder = models_directory.join(MODEL_ID);
    fs::create_dir_all(&model_folder).expect("the model folder should be created");
    let outside_provenance = test_directory.path().join("outside-provenance.json");
    fs::write(&outside_provenance, OWN_PROVENANCE).expect("the outside record should be written");
    std::os::unix::fs::symlink(
        &outside_provenance,
        model_folder.join(".astronomical-library-provenance.json"),
    )
    .expect("the provenance symlink should be created");

    assert!(matches!(
        installed_model_folder_in(&models_directory, MODEL_ID),
        Err(InstalledModelRemovalError::NotInstalledByDownloadModels)
    ));
}
