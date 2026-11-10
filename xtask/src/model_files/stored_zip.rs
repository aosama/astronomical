//! Reading the entries of a zip archive that stores its files uncompressed.
//!
//! `torch.save` writes a zip archive whose entries are stored, not
//! compressed, so tensors can be mapped straight from the file. That is the
//! only kind of archive a converter needs to open, so this reader supports
//! exactly that and refuses everything else (compression, ZIP64, archives
//! split across files) instead of guessing.

use std::fs;
use std::path::Path;

const END_OF_CENTRAL_DIRECTORY_SIGNATURE: u32 = 0x0605_4B50;
const CENTRAL_DIRECTORY_ENTRY_SIGNATURE: u32 = 0x0201_4B50;
const LOCAL_FILE_HEADER_SIGNATURE: u32 = 0x0403_4B50;
const END_OF_CENTRAL_DIRECTORY_BYTES: usize = 22;
const CENTRAL_DIRECTORY_ENTRY_BYTES: usize = 46;
const LOCAL_FILE_HEADER_BYTES: usize = 30;
/// The end record may be followed by a comment of up to 65,535 bytes.
const MAXIMUM_END_SEARCH_BYTES: usize = END_OF_CENTRAL_DIRECTORY_BYTES + 65_535;
const STORED_COMPRESSION_METHOD: u16 = 0;
/// A size or offset of all ones means the real value is in a ZIP64 record.
const ZIP64_MARKER: u32 = 0xFFFF_FFFF;
/// PyTorch archives of a decision head are a few megabytes; a larger file
/// is not one, and reading it whole would be a mistake.
const MAXIMUM_ARCHIVE_BYTES: u64 = 512 * 1024 * 1024;

/// One archive held in memory, with its entries located.
pub(crate) struct StoredZipArchive {
    archive_bytes: Vec<u8>,
    entries: Vec<StoredZipEntry>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct StoredZipEntry {
    entry_name: String,
    data_start: usize,
    byte_count: usize,
}

impl StoredZipArchive {
    pub(crate) fn open(archive_path: &Path) -> Result<Self, String> {
        let archive_length = fs::metadata(archive_path)
            .map_err(|metadata_error| format!("could not read the archive: {metadata_error}"))?
            .len();
        if archive_length > MAXIMUM_ARCHIVE_BYTES {
            return Err(format!(
                "the archive is {archive_length} bytes, above the {MAXIMUM_ARCHIVE_BYTES}-byte limit"
            ));
        }
        let archive_bytes = fs::read(archive_path)
            .map_err(|read_error| format!("could not read the archive: {read_error}"))?;
        Self::from_bytes(archive_bytes)
    }

    pub(crate) fn from_bytes(archive_bytes: Vec<u8>) -> Result<Self, String> {
        let end_record_start = find_end_of_central_directory(&archive_bytes)?;
        let entry_count = read_u16(&archive_bytes, end_record_start + 10)? as usize;
        let directory_start = read_u32(&archive_bytes, end_record_start + 16)?;
        if directory_start == ZIP64_MARKER {
            return Err("ZIP64 archives are not supported".to_owned());
        }
        let mut entries = Vec::with_capacity(entry_count);
        let mut entry_start = directory_start as usize;
        for _ in 0..entry_count {
            let (entry, next_entry_start) =
                read_central_directory_entry(&archive_bytes, entry_start)?;
            entries.push(entry);
            entry_start = next_entry_start;
        }
        Ok(Self {
            archive_bytes,
            entries,
        })
    }

    /// The name of every entry, in archive order.
    pub(crate) fn entry_names(&self) -> Vec<&str> {
        self.entries
            .iter()
            .map(|entry| entry.entry_name.as_str())
            .collect()
    }

    /// The bytes of the entry named `entry_name`.
    pub(crate) fn entry_bytes(&self, entry_name: &str) -> Result<&[u8], String> {
        let Some(entry) = self
            .entries
            .iter()
            .find(|entry| entry.entry_name == entry_name)
        else {
            return Err(format!("the archive has no entry named {entry_name}"));
        };
        Ok(&self.archive_bytes[entry.data_start..entry.data_start + entry.byte_count])
    }
}

fn find_end_of_central_directory(archive_bytes: &[u8]) -> Result<usize, String> {
    if archive_bytes.len() < END_OF_CENTRAL_DIRECTORY_BYTES {
        return Err("the file is too short to be a zip archive".to_owned());
    }
    let search_floor = archive_bytes.len().saturating_sub(MAXIMUM_END_SEARCH_BYTES);
    let mut candidate_start = archive_bytes.len() - END_OF_CENTRAL_DIRECTORY_BYTES;
    loop {
        if read_u32(archive_bytes, candidate_start)? == END_OF_CENTRAL_DIRECTORY_SIGNATURE {
            return Ok(candidate_start);
        }
        if candidate_start == search_floor {
            return Err("the file has no zip end-of-directory record".to_owned());
        }
        candidate_start -= 1;
    }
}

/// Reads one central-directory entry and returns it with where the next starts.
fn read_central_directory_entry(
    archive_bytes: &[u8],
    entry_start: usize,
) -> Result<(StoredZipEntry, usize), String> {
    if read_u32(archive_bytes, entry_start)? != CENTRAL_DIRECTORY_ENTRY_SIGNATURE {
        return Err("a zip directory entry is malformed".to_owned());
    }
    let compression_method = read_u16(archive_bytes, entry_start + 10)?;
    let compressed_size = read_u32(archive_bytes, entry_start + 20)?;
    let uncompressed_size = read_u32(archive_bytes, entry_start + 24)?;
    let name_length = read_u16(archive_bytes, entry_start + 28)? as usize;
    let extra_length = read_u16(archive_bytes, entry_start + 30)? as usize;
    let comment_length = read_u16(archive_bytes, entry_start + 32)? as usize;
    let local_header_start = read_u32(archive_bytes, entry_start + 42)?;
    let name_start = entry_start + CENTRAL_DIRECTORY_ENTRY_BYTES;
    let entry_name =
        String::from_utf8(byte_range(archive_bytes, name_start, name_length)?.to_vec())
            .map_err(|_| "a zip entry name is not UTF-8".to_owned())?;
    if compression_method != STORED_COMPRESSION_METHOD || compressed_size != uncompressed_size {
        return Err(format!(
            "zip entry {entry_name} is compressed; only stored entries are read"
        ));
    }
    if [compressed_size, local_header_start].contains(&ZIP64_MARKER) {
        return Err("ZIP64 archives are not supported".to_owned());
    }
    let data_start = local_data_start(archive_bytes, local_header_start as usize)?;
    byte_range(archive_bytes, data_start, compressed_size as usize)?;
    let next_entry_start = name_start + name_length + extra_length + comment_length;
    Ok((
        StoredZipEntry {
            entry_name,
            data_start,
            byte_count: compressed_size as usize,
        },
        next_entry_start,
    ))
}

/// Where an entry's data begins: after its local header, whose name and
/// extra field may differ in length from the central directory's copy.
fn local_data_start(archive_bytes: &[u8], local_header_start: usize) -> Result<usize, String> {
    if read_u32(archive_bytes, local_header_start)? != LOCAL_FILE_HEADER_SIGNATURE {
        return Err("a zip local file header is malformed".to_owned());
    }
    let name_length = read_u16(archive_bytes, local_header_start + 26)? as usize;
    let extra_length = read_u16(archive_bytes, local_header_start + 28)? as usize;
    Ok(local_header_start + LOCAL_FILE_HEADER_BYTES + name_length + extra_length)
}

fn byte_range(
    archive_bytes: &[u8],
    range_start: usize,
    range_length: usize,
) -> Result<&[u8], String> {
    archive_bytes
        .get(range_start..range_start.saturating_add(range_length))
        .ok_or_else(|| "a zip record points past the end of the file".to_owned())
}

fn read_u16(archive_bytes: &[u8], field_start: usize) -> Result<u16, String> {
    let field_bytes = byte_range(archive_bytes, field_start, 2)?;
    Ok(u16::from_le_bytes([field_bytes[0], field_bytes[1]]))
}

fn read_u32(archive_bytes: &[u8], field_start: usize) -> Result<u32, String> {
    let field_bytes = byte_range(archive_bytes, field_start, 4)?;
    Ok(u32::from_le_bytes([
        field_bytes[0],
        field_bytes[1],
        field_bytes[2],
        field_bytes[3],
    ]))
}

#[cfg(test)]
#[path = "stored_zip_tests.rs"]
mod tests;
