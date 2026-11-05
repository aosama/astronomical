use std::fs;

use super::{package_manifest_json, write_ml_package};
use crate::ml_package_layout::MlProgramPackageContents;

#[test]
fn should_name_the_model_item_as_the_root() {
    let manifest = package_manifest_json();

    assert!(manifest.contains(r#""rootModelIdentifier": "A1B2C3D4-0161-4000-8000-000000000001""#));
    assert!(manifest.contains(r#""path": "com.apple.CoreML/model.mlmodel""#));
    assert!(manifest.contains(r#""path": "com.apple.CoreML/weights""#));
}

#[test]
fn should_write_the_manifest_document_and_weights_where_core_ml_looks() {
    let parent_directory = tempfile::tempdir().expect("a scratch directory");
    let contents = MlProgramPackageContents {
        model_document: vec![0x08, 0x09],
        weight_blob_file: vec![0; 64],
    };

    let package_path =
        write_ml_package(parent_directory.path(), &contents).expect("package writes");

    assert!(package_path.ends_with("projection.mlpackage"));
    assert_eq!(
        fs::read(package_path.join("Data/com.apple.CoreML/model.mlmodel")).expect("model readable"),
        vec![0x08, 0x09]
    );
    assert_eq!(
        fs::read(package_path.join("Data/com.apple.CoreML/weights/weight.bin"))
            .expect("weights readable")
            .len(),
        64
    );
    assert!(package_path.join("Manifest.json").is_file());
}

#[test]
fn should_fail_when_the_parent_is_a_file() {
    let parent_directory = tempfile::tempdir().expect("a scratch directory");
    let blocking_file = parent_directory.path().join("not-a-directory");
    fs::write(&blocking_file, b"x").expect("file writes");
    let contents = MlProgramPackageContents {
        model_document: Vec::new(),
        weight_blob_file: Vec::new(),
    };

    assert!(write_ml_package(&blocking_file, &contents).is_err());
}
