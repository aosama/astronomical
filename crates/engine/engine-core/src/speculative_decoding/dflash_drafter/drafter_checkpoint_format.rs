//! How a drafter checkpoint stores its weights, read before any of them is
//! loaded: bf16 as its publisher trained it, or already in MLX's affine
//! packing, as community drafters for Apple Silicon ship (a 4-bit drafter is
//! about a third of the download). The memory a load asks for depends on it.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use super::drafter_configuration::DFlashDrafterConfiguration;

/// A safetensors header names every tensor once; a few hundred tensors take
/// tens of kilobytes. A larger header is not a drafter's.
const MAXIMUM_HEADER_BYTES: u64 = 16 * 1024 * 1024;
/// The checkpoint's codebooks are dequantized to bf16 at load.
const BFLOAT16_BYTES: u64 = 2;
/// The predecessor and successor codebooks of the DFlash2 selector.
const SELECTOR_CODEBOOK_COUNT: u64 = 2;

/// Why a drafter checkpoint's header could not be read.
#[derive(Debug, thiserror::Error)]
pub enum DrafterCheckpointFormatError {
    #[error("the drafter checkpoint cannot be read: {source}")]
    Unreadable { source: std::io::Error },
    #[error("the drafter checkpoint's header is not a safetensors header: {description}")]
    MalformedHeader { description: String },
}

/// How the checkpoint holds its linears.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrafterCheckpointFormat {
    /// bf16, quantized to 4 bits at load.
    FullPrecision,
    /// Already in MLX's affine packing: served as stored.
    Prequantized,
}

/// Reads the format of the safetensors checkpoint at `weights_path` from its
/// header alone: a checkpoint that stores any `.scales` tensor is packed.
pub fn read_drafter_checkpoint_format(
    weights_path: &Path,
) -> Result<DrafterCheckpointFormat, DrafterCheckpointFormatError> {
    let mut checkpoint = File::open(weights_path)
        .map_err(|source| DrafterCheckpointFormatError::Unreadable { source })?;
    let mut header_length_bytes = [0_u8; 8];
    checkpoint
        .read_exact(&mut header_length_bytes)
        .map_err(|source| DrafterCheckpointFormatError::Unreadable { source })?;
    let header_byte_count = u64::from_le_bytes(header_length_bytes);
    if header_byte_count > MAXIMUM_HEADER_BYTES {
        return Err(DrafterCheckpointFormatError::MalformedHeader {
            description: format!("a {header_byte_count}-byte header is too large"),
        });
    }
    let mut header_bytes = Vec::with_capacity(header_byte_count as usize);
    checkpoint
        .take(header_byte_count)
        .read_to_end(&mut header_bytes)
        .map_err(|source| DrafterCheckpointFormatError::Unreadable { source })?;
    drafter_checkpoint_format_from_header(&header_bytes)
}

/// The format a safetensors header's tensor names declare.
pub fn drafter_checkpoint_format_from_header(
    header_bytes: &[u8],
) -> Result<DrafterCheckpointFormat, DrafterCheckpointFormatError> {
    let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(header_bytes)
        .map_err(|parse_error| {
        DrafterCheckpointFormatError::MalformedHeader {
            description: parse_error.to_string(),
        }
    })?;
    let stores_packed_modules = header
        .keys()
        .any(|tensor_name| tensor_name.ends_with(".scales"));
    Ok(if stores_packed_modules {
        DrafterCheckpointFormat::Prequantized
    } else {
        DrafterCheckpointFormat::FullPrecision
    })
}

/// The most a full-precision drafter's load holds at once, as a share of its
/// checkpoint file, when the checkpoint is let go right after binding so
/// each linear's full-precision copy is freed once it is requantized: the
/// 4-bit linears (9/32 of their bf16 bytes with group-64 scales and biases),
/// the tensors that stay bf16, and one linear still in bf16. Measured at 0.32
/// of the checkpoint on both shipped drafters (`dflash_drafter_load_memory`);
/// 3/8 leaves room for a drafter whose largest linear is a larger share.
const LOAD_PEAK_EIGHTHS_OF_CHECKPOINT: u64 = 3;

/// What loading a drafter whose checkpoint file is `checkpoint_bytes` holds at
/// its peak, so a load check asks for that room and no more. A packed
/// checkpoint is served as stored, so all of it stays, plus its selector
/// codebooks once dequantized to bf16.
#[must_use]
pub fn drafter_load_peak_bytes(
    checkpoint_bytes: u64,
    checkpoint_format: DrafterCheckpointFormat,
    configuration: &DFlashDrafterConfiguration,
) -> u64 {
    checkpoint_format.load_peak_bytes(checkpoint_bytes, configuration)
}

impl DrafterCheckpointFormat {
    /// The peak of loading a checkpoint of `checkpoint_bytes` in this format.
    fn load_peak_bytes(
        self,
        checkpoint_bytes: u64,
        configuration: &DFlashDrafterConfiguration,
    ) -> u64 {
        match self {
            Self::FullPrecision => checkpoint_bytes.div_ceil(8) * LOAD_PEAK_EIGHTHS_OF_CHECKPOINT,
            Self::Prequantized => {
                checkpoint_bytes.saturating_add(dequantized_codebook_bytes(configuration))
            }
        }
    }
}

/// The bf16 bytes of the selector's codebooks, `[vocabulary, rank]` each;
/// zero for a drafter without a selector.
fn dequantized_codebook_bytes(configuration: &DFlashDrafterConfiguration) -> u64 {
    configuration
        .candidate_selector
        .map_or(0, |selector_shape| {
            let codebook_values = u64::try_from(configuration.vocabulary_size).unwrap_or(0)
                * u64::try_from(selector_shape.rank).unwrap_or(0);
            SELECTOR_CODEBOOK_COUNT * codebook_values * BFLOAT16_BYTES
        })
}

/// The bits and group size of one affine-packed module, read from its
/// shapes: `[output, input x bits / 32]` 32-bit words of codes and
/// `[output, input / group]` scales.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AffinePacking {
    pub bits: i32,
    pub group_size: i32,
}

impl AffinePacking {
    /// The packing that turns `logical_shape` into these packed shapes, or
    /// `None` when no whole number of bits and groups does.
    #[must_use]
    pub fn read(
        logical_shape: &[i32],
        packed_weight_shape: &[i32],
        scales_shape: &[i32],
    ) -> Option<Self> {
        let [output_width, input_width] = *logical_shape else {
            return None;
        };
        let [packed_rows, packed_words] = *packed_weight_shape else {
            return None;
        };
        let [scale_rows, group_count] = *scales_shape else {
            return None;
        };
        let shapes_agree = packed_rows == output_width && scale_rows == output_width;
        let packs_whole = input_width > 0
            && packed_words > 0
            && group_count > 0
            && (packed_words * 32) % input_width == 0
            && input_width % group_count == 0;
        if !shapes_agree || !packs_whole {
            return None;
        }
        Some(Self {
            bits: packed_words * 32 / input_width,
            group_size: input_width / group_count,
        })
    }
}
