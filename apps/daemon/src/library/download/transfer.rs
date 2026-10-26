//! Ranged payload transfer, segment assembly, and digest verification.

pub(crate) mod file_digest;
pub(crate) mod payload_response;
pub(crate) mod payload_transfer;
pub(crate) mod payload_transfer_segment;
pub(crate) mod payload_transfer_window;
pub(crate) mod payload_verification;
pub(crate) mod staged_file;

pub use file_digest::DownloadFileDigest;
pub use payload_transfer::{
    DownloadPayloadTransfer, DownloadPayloadTransferError, DownloadPayloadTransferOutcome,
};
