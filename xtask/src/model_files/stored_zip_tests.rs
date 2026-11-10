//! Unit contracts for the stored-zip reader, on archives built here.

use super::StoredZipArchive;
use crate::model_files::test_support::{TestEntry, stored_zip_bytes};

fn stored(entry_name: &'static str, entry_bytes: &'static [u8]) -> TestEntry<'static> {
    TestEntry {
        entry_name,
        entry_bytes,
        compression_method: 0,
    }
}

#[test]
fn should_read_every_stored_entry_by_name() {
    let archive = StoredZipArchive::from_bytes(stored_zip_bytes(&[
        stored("head/data.pkl", b"pickle bytes"),
        stored("head/data/0", &[1, 2, 3, 4]),
        stored("head/byteorder", b"little"),
    ]))
    .expect("a stored archive should open");

    assert_eq!(
        archive.entry_names(),
        ["head/data.pkl", "head/data/0", "head/byteorder"]
    );
    assert_eq!(archive.entry_bytes("head/data/0"), Ok(&[1_u8, 2, 3, 4][..]));
    assert_eq!(archive.entry_bytes("head/byteorder"), Ok(&b"little"[..]));
}

#[test]
fn should_read_an_empty_entry() {
    let archive = StoredZipArchive::from_bytes(stored_zip_bytes(&[stored("empty", b"")]))
        .expect("an archive with an empty entry should open");

    assert_eq!(archive.entry_bytes("empty"), Ok(&[][..]));
}

#[test]
fn should_refuse_a_compressed_entry() {
    let archive_outcome = StoredZipArchive::from_bytes(stored_zip_bytes(&[TestEntry {
        entry_name: "deflated",
        entry_bytes: b"compressed",
        compression_method: 8,
    }]));

    assert!(matches!(archive_outcome, Err(reason) if reason.contains("compressed")));
}

#[test]
fn should_refuse_a_missing_entry_and_a_file_that_is_not_an_archive() {
    let archive = StoredZipArchive::from_bytes(stored_zip_bytes(&[stored("a", b"x")]))
        .expect("the archive should open");

    assert!(archive.entry_bytes("b").is_err());
    assert!(StoredZipArchive::from_bytes(b"not a zip archive at all, just text".to_vec()).is_err());
    assert!(StoredZipArchive::from_bytes(Vec::new()).is_err());
}

#[test]
fn should_refuse_an_archive_whose_records_point_past_its_end() {
    let mut archive_bytes = stored_zip_bytes(&[stored("a", b"data")]);
    let directory_offset_field = archive_bytes.len() - 6;
    archive_bytes[directory_offset_field..directory_offset_field + 4]
        .copy_from_slice(&10_000_u32.to_le_bytes());

    assert!(StoredZipArchive::from_bytes(archive_bytes).is_err());
}
