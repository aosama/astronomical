use super::validate_payload_response;
use crate::library::DownloadPayloadTransferError;

const FILE_BYTES: u64 = 1_000;

#[test]
fn should_accept_exactly_the_requested_range() {
    let outcome = validate_payload_response(
        206,
        Some("bytes 100-299/1000"),
        Some(200),
        (100, 300),
        FILE_BYTES,
    );

    assert!(outcome.is_ok());
}

#[test]
fn should_accept_the_rest_of_the_file_from_a_server_that_ignores_the_upper_bound() {
    let outcome = validate_payload_response(
        206,
        Some("bytes 100-999/1000"),
        Some(900),
        (100, 300),
        FILE_BYTES,
    );

    assert!(outcome.is_ok());
}

#[test]
fn should_accept_the_whole_file_with_200_only_for_a_request_from_byte_zero() {
    assert!(validate_payload_response(200, None, Some(1_000), (0, 300), FILE_BYTES).is_ok());
    assert!(matches!(
        validate_payload_response(200, None, Some(900), (100, 300), FILE_BYTES),
        Err(DownloadPayloadTransferError::InvalidRangeResponse)
    ));
}

#[test]
fn should_accept_a_response_that_declares_no_length() {
    assert!(
        validate_payload_response(
            206,
            Some("bytes 100-299/1000"),
            None,
            (100, 300),
            FILE_BYTES
        )
        .is_ok()
    );
}

#[test]
fn should_refuse_a_range_that_starts_elsewhere_or_names_another_file_size() {
    for wrong_content_range in [
        Some("bytes 101-299/1000"),
        Some("bytes 100-299/2000"),
        Some("bytes 100-298/1000"),
        None,
    ] {
        assert!(matches!(
            validate_payload_response(206, wrong_content_range, None, (100, 300), FILE_BYTES),
            Err(DownloadPayloadTransferError::InvalidRangeResponse)
        ));
    }
}

#[test]
fn should_refuse_a_declared_length_that_differs_from_the_range_it_answers() {
    assert!(matches!(
        validate_payload_response(
            206,
            Some("bytes 100-299/1000"),
            Some(900),
            (100, 300),
            FILE_BYTES
        ),
        Err(DownloadPayloadTransferError::InvalidPayloadLength)
    ));
    assert!(matches!(
        validate_payload_response(200, None, Some(999), (0, 300), FILE_BYTES),
        Err(DownloadPayloadTransferError::InvalidPayloadLength)
    ));
}

#[test]
fn should_accept_the_last_piece_whose_exact_range_is_also_the_rest_of_the_file() {
    assert!(
        validate_payload_response(
            206,
            Some("bytes 900-999/1000"),
            Some(100),
            (900, 1_000),
            FILE_BYTES
        )
        .is_ok()
    );
}
