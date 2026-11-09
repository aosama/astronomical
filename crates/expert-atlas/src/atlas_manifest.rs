//! The manifest that makes an Expert Atlas directory a model of its own.
//!
//! It replaces the safetensors shard index: it names the resident weight
//! bundle (every tensor that is not a routed expert), one record file per
//! expert layer, and, for every layer, exactly where each of the nine expert
//! tensors sits inside a record. Serving builds its expert plans from this
//! document alone, so a layer's layout is spelled out per layer rather than
//! assumed shared, which keeps mixed-precision checkpoints exact.

use std::path::{Component, Path};

use serde::{Deserialize, Serialize};

/// The file that marks a directory as an Atlas model.
pub const ATLAS_MANIFEST_FILE_NAME: &str = "manifest.json";
/// Every tensor that is not a routed expert, as one standard safetensors file.
pub const RESIDENT_BUNDLE_FILE_NAME: &str = "resident.safetensors";
/// The directory holding one record file per expert layer.
pub const EXPERT_LAYER_DIRECTORY_NAME: &str = "experts";
/// The `format` value of every Atlas manifest.
pub const ATLAS_FORMAT_NAME: &str = "astronomical-expert-atlas";
/// On-disk version. Version 3 was the retired per-expert pack layout; the
/// Atlas takes the next number so a reader never mistakes one for the other.
pub const ATLAS_FORMAT_VERSION: u32 = 4;
/// Record starts and lengths are multiples of the Apple Silicon page size, so
/// a record read is a direct (uncached) read and a record can be wrapped as a
/// Metal buffer without copying.
pub const ATLAS_RECORD_ALIGNMENT_BYTES: u64 = 16_384;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AtlasManifest {
    pub format: String,
    pub format_version: u32,
    pub source: AtlasSource,
    pub resident_files: Vec<AtlasDeclaredFile>,
    pub expert_files: Vec<AtlasDeclaredFile>,
    pub expert_layers: Vec<AtlasExpertLayer>,
}

/// Where the Atlas came from, so it can be rebuilt or compared.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AtlasSource {
    pub model_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
}

/// A file the Atlas consists of, with the size and digest it was written with.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AtlasDeclaredFile {
    pub file_name: String,
    pub expected_file_byte_count: u64,
    pub sha256: String,
}

/// One expert layer's record file: `expert_count` records, each
/// `record_stride_bytes` apart, starting at offset 0.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AtlasExpertLayer {
    pub layer_index: usize,
    pub file_name: String,
    pub expert_count: usize,
    pub record_byte_count: u64,
    pub record_stride_bytes: u64,
    pub slots: Vec<AtlasRecordSlot>,
}

/// One expert tensor inside every record of its layer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AtlasRecordSlot {
    /// The tensor's stored name in the source checkpoint.
    pub tensor_name: String,
    /// Safetensors dtype name, for example `U32` or `BF16`.
    pub dtype: String,
    /// The full expert-batched shape, `[expert_count, ...]`.
    pub shape: Vec<usize>,
    pub offset_in_record: u64,
    pub bytes_per_expert: u64,
}

impl AtlasExpertLayer {
    /// Where expert `expert_id`'s copy of `slot` starts in the layer file.
    #[must_use]
    pub fn slot_file_offset(&self, expert_id: usize, slot: &AtlasRecordSlot) -> u64 {
        expert_id as u64 * self.record_stride_bytes + slot.offset_in_record
    }

    #[must_use]
    pub fn file_byte_count(&self) -> u64 {
        self.expert_count as u64 * self.record_stride_bytes
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AtlasManifestError {
    #[error("the Atlas manifest at {path} could not be read: {source}")]
    Unreadable {
        path: String,
        source: std::io::Error,
    },
    #[error("the Atlas manifest at {path} is not valid: {source}")]
    Invalid {
        path: String,
        source: serde_json::Error,
    },
    #[error(
        "the manifest at {path} is not an Atlas manifest (format {format:?}, version {format_version})"
    )]
    NotAnAtlas {
        path: String,
        format: String,
        format_version: u32,
    },
    #[error("the Atlas manifest declares expert layer {layer_index} inconsistently: {reason}")]
    InconsistentLayer { layer_index: usize, reason: String },
    #[error("the Atlas manifest declares files inconsistently: {reason}")]
    InconsistentFiles { reason: String },
}

impl AtlasManifest {
    /// Reads and checks the manifest in `model_directory`: the format name
    /// and version, that every layer's slots tile its records, and that
    /// every file it names is a declared file inside the model directory.
    pub fn read_from_directory(model_directory: &Path) -> Result<Self, AtlasManifestError> {
        let manifest_path = model_directory.join(ATLAS_MANIFEST_FILE_NAME);
        let path = manifest_path.display().to_string();
        let manifest_bytes =
            std::fs::read(&manifest_path).map_err(|source| AtlasManifestError::Unreadable {
                path: path.clone(),
                source,
            })?;
        let manifest: Self = serde_json::from_slice(&manifest_bytes).map_err(|source| {
            AtlasManifestError::Invalid {
                path: path.clone(),
                source,
            }
        })?;
        if manifest.format != ATLAS_FORMAT_NAME || manifest.format_version != ATLAS_FORMAT_VERSION {
            return Err(AtlasManifestError::NotAnAtlas {
                path,
                format: manifest.format,
                format_version: manifest.format_version,
            });
        }
        for expert_layer in &manifest.expert_layers {
            check_layer_geometry(expert_layer)?;
        }
        check_declared_files(&manifest)?;
        Ok(manifest)
    }

    /// The layer whose experts live at `layer_index`, if the model has one.
    #[must_use]
    pub fn expert_layer(&self, layer_index: usize) -> Option<&AtlasExpertLayer> {
        self.expert_layers
            .iter()
            .find(|expert_layer| expert_layer.layer_index == layer_index)
    }
}

/// Whether `model_directory` holds an Atlas manifest, without validating it.
/// Discovery uses this to route a directory to the Atlas path.
#[must_use]
pub fn declares_atlas_manifest(model_directory: &Path) -> bool {
    #[derive(Deserialize)]
    struct ManifestFormatProbe {
        #[serde(default)]
        format: String,
        format_version: u32,
    }
    std::fs::read(model_directory.join(ATLAS_MANIFEST_FILE_NAME))
        .ok()
        .and_then(|manifest_bytes| {
            serde_json::from_slice::<ManifestFormatProbe>(&manifest_bytes).ok()
        })
        .is_some_and(|format_probe| {
            format_probe.format == ATLAS_FORMAT_NAME
                && format_probe.format_version == ATLAS_FORMAT_VERSION
        })
}

/// Slots must sit back to back from offset 0, fill the record exactly, and
/// the stride must be the record rounded up to the alignment; each slot's
/// shape must lead with the layer's expert count, and its dtype and shape
/// must give exactly its declared bytes per expert.
fn check_layer_geometry(expert_layer: &AtlasExpertLayer) -> Result<(), AtlasManifestError> {
    let inconsistent = |reason: String| AtlasManifestError::InconsistentLayer {
        layer_index: expert_layer.layer_index,
        reason,
    };
    let mut expected_offset_in_record = 0;
    for slot in &expert_layer.slots {
        if slot.offset_in_record != expected_offset_in_record {
            return Err(inconsistent(format!(
                "{} starts at {}, expected {expected_offset_in_record}",
                slot.tensor_name, slot.offset_in_record
            )));
        }
        if slot.shape.first() != Some(&expert_layer.expert_count) {
            return Err(inconsistent(format!(
                "{} does not lead with {} experts",
                slot.tensor_name, expert_layer.expert_count
            )));
        }
        if expert_slice_byte_count(slot) != Some(slot.bytes_per_expert) {
            return Err(inconsistent(format!(
                "{} is {} {:?}, which is not {} bytes per expert",
                slot.tensor_name, slot.dtype, slot.shape, slot.bytes_per_expert
            )));
        }
        expected_offset_in_record = expected_offset_in_record
            .checked_add(slot.bytes_per_expert)
            .ok_or_else(|| inconsistent("record size overflows".to_owned()))?;
    }
    if expected_offset_in_record != expert_layer.record_byte_count {
        return Err(inconsistent(format!(
            "slots fill {expected_offset_in_record} bytes, record declares {}",
            expert_layer.record_byte_count
        )));
    }
    let aligned_stride = expert_layer
        .record_byte_count
        .div_ceil(ATLAS_RECORD_ALIGNMENT_BYTES)
        .checked_mul(ATLAS_RECORD_ALIGNMENT_BYTES);
    if Some(expert_layer.record_stride_bytes) != aligned_stride {
        return Err(inconsistent(format!(
            "stride {} is not the record rounded to {ATLAS_RECORD_ALIGNMENT_BYTES} bytes",
            expert_layer.record_stride_bytes
        )));
    }
    Ok(())
}

/// One expert's bytes of `slot` as its dtype and shape imply, or `None` for
/// an unknown dtype or a size past `u64`. Serving reads exactly
/// `bytes_per_expert` and hands MLX the shape, so the two must agree or the
/// expert math would run on misread bytes without any error.
#[must_use]
pub fn expert_slice_byte_count(slot: &AtlasRecordSlot) -> Option<u64> {
    let element_byte_count: u64 = match slot.dtype.as_str() {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => 1,
        "U16" | "I16" | "F16" | "BF16" => 2,
        "U32" | "I32" | "F32" => 4,
        "U64" | "I64" | "F64" => 8,
        _ => return None,
    };
    slot.shape
        .get(1..)?
        .iter()
        .try_fold(element_byte_count, |byte_count, &dimension| {
            byte_count.checked_mul(u64::try_from(dimension).ok()?)
        })
}

/// Every layer file must be a declared expert file whose declared size is
/// exactly its records, and every declared name must stay inside the model
/// directory. Serving joins these names onto the model directory, and an
/// Atlas can arrive by download, so `..` or an absolute path must never be
/// followed; the declared sizes are what validation checks against disk.
fn check_declared_files(manifest: &AtlasManifest) -> Result<(), AtlasManifestError> {
    let inconsistent = |reason: String| AtlasManifestError::InconsistentFiles { reason };
    for declared_file in manifest.resident_files.iter().chain(&manifest.expert_files) {
        if !is_plain_relative_file_name(&declared_file.file_name) {
            return Err(inconsistent(format!(
                "{:?} is not a plain path inside the model directory",
                declared_file.file_name
            )));
        }
    }
    for expert_layer in &manifest.expert_layers {
        let declared_file = manifest
            .expert_files
            .iter()
            .find(|declared_file| declared_file.file_name == expert_layer.file_name)
            .ok_or_else(|| {
                inconsistent(format!(
                    "layer {} reads {:?}, which is not a declared expert file",
                    expert_layer.layer_index, expert_layer.file_name
                ))
            })?;
        let layer_file_byte_count =
            (expert_layer.expert_count as u64).checked_mul(expert_layer.record_stride_bytes);
        if layer_file_byte_count != Some(declared_file.expected_file_byte_count) {
            return Err(inconsistent(format!(
                "layer {} declares {} bytes for {} records of {} bytes",
                expert_layer.layer_index,
                declared_file.expected_file_byte_count,
                expert_layer.expert_count,
                expert_layer.record_stride_bytes
            )));
        }
    }
    Ok(())
}

fn is_plain_relative_file_name(file_name: &str) -> bool {
    let file_path = Path::new(file_name);
    file_path.components().next().is_some()
        && file_path
            .components()
            .all(|path_component| matches!(path_component, Component::Normal(_)))
}

#[cfg(test)]
#[path = "atlas_manifest_tests.rs"]
mod atlas_manifest_tests;
