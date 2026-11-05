//! The MIL weight-blob file (`weights/weight.bin`) an ML Program reads its
//! constant tensors from.
//!
//! Layout (Apple's `MILBlob` storage format, version 2): a 64-byte header
//! holding the blob count, then for every blob a 64-byte metadata record
//! followed by its data, each starting on a 64-byte boundary. A program refers
//! to a blob by the file offset of its metadata record. Large weights go here
//! rather than inline in the program because Core ML memory-maps this file,
//! and because inline immediate values have no sub-byte (4-bit) encoding.

const STORAGE_ALIGNMENT_BYTES: usize = 64;
const STORAGE_FORMAT_VERSION: u32 = 2;
const BLOB_METADATA_SENTINEL: u32 = 0xDEAD_BEEF;
const HEADER_BYTES: usize = 64;
const METADATA_BYTES: usize = 64;

/// `MILBlob::Blob::BlobDataType` codes for the element types used here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlobElementType {
    Float16,
    UnsignedInt8,
    UnsignedInt4,
}

impl BlobElementType {
    fn storage_code(self) -> u32 {
        match self {
            Self::Float16 => 1,
            Self::UnsignedInt8 => 3,
            Self::UnsignedInt4 => 11,
        }
    }
}

/// Accumulates blobs and produces the finished file bytes.
pub struct WeightBlobFile {
    file_bytes: Vec<u8>,
    blob_count: u32,
}

impl Default for WeightBlobFile {
    fn default() -> Self {
        Self::new()
    }
}

impl WeightBlobFile {
    #[must_use]
    pub fn new() -> Self {
        Self {
            file_bytes: vec![0; HEADER_BYTES],
            blob_count: 0,
        }
    }

    /// Appends fp16 values (binary16 bit patterns) and returns the offset the
    /// program's `BlobFileValue` must carry.
    pub fn append_fp16(&mut self, fp16_values: &[u16]) -> u64 {
        let mut little_endian_halves = Vec::with_capacity(fp16_values.len() * 2);
        for half_bits in fp16_values {
            little_endian_halves.extend_from_slice(&half_bits.to_le_bytes());
        }
        self.append_blob(BlobElementType::Float16, &little_endian_halves, 0)
    }

    /// Appends 8-bit codes, one per byte.
    pub fn append_uint8(&mut self, codes: &[u8]) -> u64 {
        self.append_blob(BlobElementType::UnsignedInt8, codes, 0)
    }

    /// Appends 4-bit codes (each `0..=15`), packed two per byte.
    pub fn append_uint4(&mut self, codes: &[u8]) -> u64 {
        let unused_bits = if codes.len().is_multiple_of(2) { 0 } else { 4 };
        self.append_blob(
            BlobElementType::UnsignedInt4,
            &pack_uint4_low_nibble_first(codes),
            unused_bits,
        )
    }

    #[must_use]
    pub fn into_bytes(mut self) -> Vec<u8> {
        self.file_bytes[0..4].copy_from_slice(&self.blob_count.to_le_bytes());
        self.file_bytes[4..8].copy_from_slice(&STORAGE_FORMAT_VERSION.to_le_bytes());
        self.file_bytes
    }

    fn append_blob(
        &mut self,
        element_type: BlobElementType,
        blob_bytes: &[u8],
        unused_bits: u64,
    ) -> u64 {
        let metadata_offset = self.file_bytes.len();
        let data_offset = metadata_offset + METADATA_BYTES;
        let mut metadata = Vec::with_capacity(METADATA_BYTES);
        metadata.extend_from_slice(&BLOB_METADATA_SENTINEL.to_le_bytes());
        metadata.extend_from_slice(&element_type.storage_code().to_le_bytes());
        metadata.extend_from_slice(&(blob_bytes.len() as u64).to_le_bytes());
        metadata.extend_from_slice(&(data_offset as u64).to_le_bytes());
        metadata.extend_from_slice(&unused_bits.to_le_bytes());
        metadata.resize(METADATA_BYTES, 0);
        self.file_bytes.extend_from_slice(&metadata);
        self.file_bytes.extend_from_slice(blob_bytes);
        let aligned_length = self
            .file_bytes
            .len()
            .next_multiple_of(STORAGE_ALIGNMENT_BYTES);
        self.file_bytes.resize(aligned_length, 0);
        self.blob_count += 1;
        metadata_offset as u64
    }
}

/// Element `n` occupies bits `4·(n mod 2)` of byte `n / 2`, the order
/// `MILBlob::PackSubByteVec` uses.
#[must_use]
pub fn pack_uint4_low_nibble_first(codes: &[u8]) -> Vec<u8> {
    let mut packed_bytes = Vec::with_capacity(codes.len().div_ceil(2));
    for code_pair in codes.chunks(2) {
        let low_nibble = code_pair[0] & 0x0f;
        let high_nibble = code_pair.get(1).copied().unwrap_or(0) & 0x0f;
        packed_bytes.push(low_nibble | (high_nibble << 4));
    }
    packed_bytes
}

#[cfg(test)]
#[path = "weight_blob_tests.rs"]
mod tests;
