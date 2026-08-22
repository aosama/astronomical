//! Verifies staged payloads against immutable provider digests before publication.

use std::{
    fs::{File, OpenOptions},
    io::{BufReader, Read},
    os::unix::fs::OpenOptionsExt,
    path::Path,
};

use sha1::Sha1;
use sha2::{Digest, Sha256};

use super::{
    DownloadFileDigest, DownloadJob, DownloadPayloadTransferError,
    download_job_store_filesystem::safely_join_existing_path,
};

pub(super) fn verify_download_job(
    models_directory: &Path,
    download_job: &DownloadJob,
) -> Result<(), DownloadPayloadTransferError> {
    let staging_directory = download_job.staging_directory(models_directory);
    verify_download_job_at_directory(&staging_directory, download_job)
}

pub(super) fn verify_download_job_at_directory(
    download_directory: &Path,
    download_job: &DownloadJob,
) -> Result<(), DownloadPayloadTransferError> {
    for download_file in download_job.files() {
        let staged_file_path = safely_join_existing_path(
            download_directory,
            Path::new(download_file.relative_path()),
        )?;
        let staged_file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&staged_file_path)
            .map_err(DownloadPayloadTransferError::WritePayload)?;
        let staged_file_bytes = staged_file
            .metadata()
            .map_err(DownloadPayloadTransferError::WritePayload)?
            .len();
        if staged_file_bytes != download_file.expected_bytes() {
            return Err(checksum_mismatch(download_file.relative_path()));
        }
        let actual_digest = hash_file(
            staged_file,
            download_file.expected_bytes(),
            download_file.expected_digest(),
        )?;
        if actual_digest != download_file.expected_digest().hex() {
            return Err(checksum_mismatch(download_file.relative_path()));
        }
    }
    Ok(())
}

fn checksum_mismatch(relative_path: &str) -> DownloadPayloadTransferError {
    DownloadPayloadTransferError::ChecksumMismatch {
        relative_path: relative_path.to_owned(),
    }
}

fn hash_file(
    staged_file: File,
    expected_bytes: u64,
    expected_digest: &DownloadFileDigest,
) -> Result<String, DownloadPayloadTransferError> {
    let mut reader = BufReader::with_capacity(1_000_000, staged_file);
    let mut read_bytes = vec![0_u8; 1_000_000];
    match expected_digest {
        DownloadFileDigest::Sha256(_) => {
            let mut digest = Sha256::new();
            copy_into_digest(&mut reader, &mut read_bytes, &mut digest)?;
            Ok(lowercase_hex(digest.finalize().as_ref()))
        }
        DownloadFileDigest::GitBlobSha1(_) => {
            let mut digest = Sha1::new();
            digest.update(format!("blob {expected_bytes}\0").as_bytes());
            copy_into_digest(&mut reader, &mut read_bytes, &mut digest)?;
            Ok(lowercase_hex(digest.finalize().as_ref()))
        }
    }
}

fn copy_into_digest<DigestOwner: Digest>(
    reader: &mut BufReader<File>,
    read_bytes: &mut [u8],
    digest: &mut DigestOwner,
) -> Result<(), DownloadPayloadTransferError> {
    loop {
        let read_byte_count = reader
            .read(read_bytes)
            .map_err(DownloadPayloadTransferError::WritePayload)?;
        if read_byte_count == 0 {
            return Ok(());
        }
        digest.update(&read_bytes[..read_byte_count]);
    }
}

fn lowercase_hex(digest_bytes: &[u8]) -> String {
    const HEX_CHARACTERS: &[u8; 16] = b"0123456789abcdef";
    let mut hexadecimal_digest = String::with_capacity(digest_bytes.len() * 2);
    for digest_byte in digest_bytes {
        hexadecimal_digest.push(HEX_CHARACTERS[(digest_byte >> 4) as usize] as char);
        hexadecimal_digest.push(HEX_CHARACTERS[(digest_byte & 0x0f) as usize] as char);
    }
    hexadecimal_digest
}
