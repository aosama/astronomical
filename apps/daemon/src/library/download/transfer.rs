//! Ranged payload transfer, segment assembly, and digest verification.

pub(crate) mod download_file_digest;
pub(crate) mod download_payload_response;
pub(crate) mod download_payload_transfer;
pub(crate) mod download_payload_transfer_segment;
pub(crate) mod download_payload_transfer_window;
pub(crate) mod download_payload_verification;
pub(crate) mod download_staged_file;

pub use download_file_digest::DownloadFileDigest;
pub use download_payload_transfer::{
    DownloadPayloadTransfer, DownloadPayloadTransferError, DownloadPayloadTransferOutcome,
};
