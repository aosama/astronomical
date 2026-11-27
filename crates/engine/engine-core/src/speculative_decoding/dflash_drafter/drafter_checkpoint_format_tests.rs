use std::io::Write;

use super::drafter_checkpoint_format::{
    AffinePacking, DrafterCheckpointFormat, DrafterCheckpointFormatError,
    drafter_checkpoint_format_from_header, drafter_load_peak_bytes, read_drafter_checkpoint_format,
};
use super::drafter_configuration::DFlashDrafterConfiguration;
use super::drafter_configuration_tests::qwen38_27b_drafter_document;

fn qwen38_27b_drafter_configuration() -> DFlashDrafterConfiguration {
    DFlashDrafterConfiguration::from_config_document(&qwen38_27b_drafter_document())
        .expect("the published 27B drafter configuration reads")
}

/// The two selector codebooks of the 27B's drafter in bf16:
/// 2 x 248,320 x 256 x 2 bytes.
const QWEN38_27B_DEQUANTIZED_CODEBOOK_BYTES: u64 = 254_279_680;

#[test]
fn should_ask_for_three_eighths_of_a_full_precision_drafter_checkpoint() {
    let dense_drafter_checkpoint_bytes = 3_843_000_000;

    assert_eq!(
        drafter_load_peak_bytes(
            dense_drafter_checkpoint_bytes,
            DrafterCheckpointFormat::FullPrecision,
            &qwen38_27b_drafter_configuration()
        ),
        1_441_125_000
    );
}

#[test]
fn should_round_a_partial_eighth_up_rather_than_ask_for_too_little() {
    let configuration = qwen38_27b_drafter_configuration();
    let full_precision = DrafterCheckpointFormat::FullPrecision;

    assert_eq!(
        drafter_load_peak_bytes(9, full_precision, &configuration),
        6,
        "two eighths rounded up, times three"
    );
    assert_eq!(
        drafter_load_peak_bytes(1, full_precision, &configuration),
        3
    );
    assert_eq!(
        drafter_load_peak_bytes(0, full_precision, &configuration),
        0
    );
}

#[test]
fn should_ask_for_a_whole_packed_checkpoint_and_its_dequantized_codebooks() {
    let packed_drafter_checkpoint_bytes = 1_146_451_617;

    assert_eq!(
        drafter_load_peak_bytes(
            packed_drafter_checkpoint_bytes,
            DrafterCheckpointFormat::Prequantized,
            &qwen38_27b_drafter_configuration()
        ),
        packed_drafter_checkpoint_bytes + QWEN38_27B_DEQUANTIZED_CODEBOOK_BYTES
    );
}

#[test]
fn should_read_a_header_with_scales_as_a_packed_checkpoint() {
    let packed_header = br#"{"__metadata__":{"bits":"4"},
        "fc.weight":{"dtype":"U32","shape":[5120,3200],"data_offsets":[0,1]},
        "fc.scales":{"dtype":"BF16","shape":[5120,400],"data_offsets":[1,2]}}"#;

    assert_eq!(
        drafter_checkpoint_format_from_header(packed_header).expect("the header reads"),
        DrafterCheckpointFormat::Prequantized
    );
}

#[test]
fn should_read_a_header_without_scales_as_a_full_precision_checkpoint() {
    let full_precision_header =
        br#"{"fc.weight":{"dtype":"BF16","shape":[5120,25600],"data_offsets":[0,1]}}"#;

    assert_eq!(
        drafter_checkpoint_format_from_header(full_precision_header).expect("the header reads"),
        DrafterCheckpointFormat::FullPrecision
    );
}

#[test]
fn should_refuse_a_header_that_is_not_json() {
    assert!(matches!(
        drafter_checkpoint_format_from_header(b"not a header"),
        Err(DrafterCheckpointFormatError::MalformedHeader { .. })
    ));
}

#[test]
fn should_read_a_checkpoint_file_by_its_header_alone() {
    let header = br#"{"fc.scales":{"dtype":"BF16","shape":[1,1],"data_offsets":[0,2]}}"#;
    let mut checkpoint = tempfile::NamedTempFile::new().expect("a scratch checkpoint opens");
    checkpoint
        .write_all(&(header.len() as u64).to_le_bytes())
        .and_then(|()| checkpoint.write_all(header))
        .and_then(|()| checkpoint.write_all(&[0, 0]))
        .expect("the scratch checkpoint writes");

    assert_eq!(
        read_drafter_checkpoint_format(checkpoint.path()).expect("the checkpoint reads"),
        DrafterCheckpointFormat::Prequantized
    );
}

#[test]
fn should_refuse_a_header_length_past_the_bound_without_reading_it() {
    let mut checkpoint = tempfile::NamedTempFile::new().expect("a scratch checkpoint opens");
    checkpoint
        .write_all(&u64::MAX.to_le_bytes())
        .expect("the scratch checkpoint writes");

    assert!(matches!(
        read_drafter_checkpoint_format(checkpoint.path()),
        Err(DrafterCheckpointFormatError::MalformedHeader { .. })
    ));
}

#[test]
fn should_refuse_a_checkpoint_shorter_than_its_length_field() {
    let mut checkpoint = tempfile::NamedTempFile::new().expect("a scratch checkpoint opens");
    checkpoint
        .write_all(&[1, 2, 3])
        .expect("the scratch checkpoint writes");

    assert!(matches!(
        read_drafter_checkpoint_format(checkpoint.path()),
        Err(DrafterCheckpointFormatError::Unreadable { .. })
    ));
}

#[test]
fn should_read_the_packing_of_a_4_bit_linear_and_an_8_bit_codebook() {
    assert_eq!(
        AffinePacking::read(&[5120, 25600], &[5120, 3200], &[5120, 400]),
        Some(AffinePacking {
            bits: 4,
            group_size: 64
        }),
        "the tap fusion linear of the published 4-bit Bonsai drafter"
    );
    assert_eq!(
        AffinePacking::read(&[248_320, 256], &[248_320, 64], &[248_320, 4]),
        Some(AffinePacking {
            bits: 8,
            group_size: 64
        }),
        "its selector codebook"
    );
}

#[test]
fn should_refuse_packed_shapes_no_whole_packing_explains() {
    assert_eq!(
        AffinePacking::read(&[5120, 25600], &[5121, 3200], &[5120, 400]),
        None,
        "row counts disagree"
    );
    assert_eq!(
        AffinePacking::read(&[5120, 100], &[5120, 3], &[5120, 4]),
        None,
        "96 bits do not split over 100 inputs"
    );
    assert_eq!(
        AffinePacking::read(&[5120, 0], &[5120, 0], &[5120, 0]),
        None,
        "an empty input"
    );
    assert_eq!(
        AffinePacking::read(&[5120], &[5120, 3200], &[5120, 400]),
        None,
        "a logical shape that is not a matrix"
    );
}
