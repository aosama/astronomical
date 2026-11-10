//! Unit contracts for reading Kev's head and settings from `head.pt`.

use super::{HEAD_TENSOR_NAMES, KevHead};
use crate::convert_kev::test_support::{TestHeadSettings, kev_head_archive_bytes};
use crate::model_files::stored_zip::StoredZipArchive;

fn head_settings() -> TestHeadSettings<'static> {
    TestHeadSettings {
        pointer_dimension: 4,
        hidden_size: 6,
        temperature: 2.3510958125672174,
        base_revision: "0123456789abcdef0123456789abcdef01234567",
        option_isolation: false,
    }
}

fn read_head(head_settings: &TestHeadSettings<'_>) -> Result<KevHead, String> {
    let head_archive = StoredZipArchive::from_bytes(kev_head_archive_bytes(head_settings))
        .expect("the test archive should open");
    KevHead::from_archive(&head_archive)
}

#[test]
fn should_read_the_four_head_tensors_and_the_settings() {
    let kev_head = read_head(&head_settings()).expect("a Kev head should read");

    let tensor_names: Vec<&str> = kev_head
        .head_tensors
        .iter()
        .map(|head_tensor| head_tensor.tensor_name)
        .collect();
    assert_eq!(tensor_names, HEAD_TENSOR_NAMES);
    assert_eq!(kev_head.head_tensors[0].shape, vec![4, 6]);
    assert_eq!(kev_head.head_tensors[1].shape, vec![4]);
    assert_eq!(kev_head.head_tensors[0].values[..3], [0.5, 1.5, 2.5]);
    assert_eq!(kev_head.head_tensors[3].values, vec![0.5, 1.5, 2.5, 3.5]);
    assert_eq!(kev_head.temperature, 2.3510958125672174);
    assert_eq!(kev_head.pointer_dimension, 4);
    assert_eq!(kev_head.base_model, "example-org/example-base");
    assert_eq!(
        kev_head.base_revision.as_deref(),
        Some("0123456789abcdef0123456789abcdef01234567")
    );
}

#[test]
fn should_refuse_a_checkpoint_trained_with_isolated_options() {
    let read_outcome = read_head(&TestHeadSettings {
        option_isolation: true,
        ..head_settings()
    });

    assert!(matches!(read_outcome, Err(reason) if reason.contains("option_isolation")));
}

#[test]
fn should_refuse_a_temperature_that_is_not_positive() {
    let read_outcome = read_head(&TestHeadSettings {
        temperature: 0.0,
        ..head_settings()
    });

    assert!(matches!(read_outcome, Err(reason) if reason.contains("temperature")));
}

#[test]
fn should_refuse_an_archive_that_is_not_a_torch_save_archive() {
    let head_archive =
        StoredZipArchive::from_bytes(crate::model_files::test_support::stored_zip_bytes(&[
            crate::model_files::test_support::TestEntry {
                entry_name: "notes.txt",
                entry_bytes: b"not a checkpoint",
                compression_method: 0,
            },
        ]))
        .expect("the archive should open");

    assert!(
        matches!(KevHead::from_archive(&head_archive), Err(reason) if reason.contains("data.pkl"))
    );
}
