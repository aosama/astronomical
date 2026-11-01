use std::fs;

use super::*;

#[test]
fn should_detect_optiq_from_its_metadata() {
    assert_eq!(
        detect_quantization_producer(true, false),
        QuantizationProducer::OptiQ
    );
}

#[test]
fn should_detect_oq_from_its_imatrix_report() {
    assert_eq!(
        detect_quantization_producer(false, true),
        QuantizationProducer::OmlxOq
    );
}

#[test]
fn should_leave_a_bare_artifact_unspecified() {
    assert_eq!(
        detect_quantization_producer(false, false),
        QuantizationProducer::Unspecified
    );
}

#[test]
fn should_prefer_optiq_when_both_sidecars_are_present() {
    assert_eq!(
        detect_quantization_producer(true, true),
        QuantizationProducer::OptiQ
    );
}

#[test]
fn should_detect_the_producer_from_files_in_a_directory() {
    let model_directory = tempfile::tempdir().expect("temporary model directory");
    assert_eq!(
        detect_quantization_producer_in_directory(model_directory.path()),
        QuantizationProducer::Unspecified
    );

    fs::write(
        model_directory.path().join(OQ_IMATRIX_REPORT_FILE_NAME),
        b"{}",
    )
    .expect("report written");
    assert_eq!(
        detect_quantization_producer_in_directory(model_directory.path()),
        QuantizationProducer::OmlxOq
    );

    fs::create_dir(model_directory.path().join("optiq")).expect("optiq folder created");
    fs::write(model_directory.path().join("optiq/metadata.json"), b"{}").expect("metadata written");
    assert_eq!(
        detect_quantization_producer_in_directory(model_directory.path()),
        QuantizationProducer::OptiQ
    );
}
