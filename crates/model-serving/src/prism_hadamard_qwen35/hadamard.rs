//! Strict Hadamard transform contract for folded ternary artifacts.
//!
//! Prism-style folded packs store projection weights in a Hadamard-rotated
//! basis and require the runtime to apply the matching activation-side
//! transform. The upstream contract is explicit that a runtime either applies
//! the declared transform or refuses to load the artifact, so every unsupported
//! declaration here is a typed load refusal rather than a silent fallback.

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;
use thiserror::Error;

#[cfg(test)]
mod tests;

/// The only accepted Hadamard manifest schema version.
const SUPPORTED_CONTRACT_VERSION: u32 = 1;
/// The only accepted orthonormal transform. The rotation is folded into the
/// stored weights at packaging time, so the runtime transform must match it exactly.
const SUPPORTED_TRANSFORM: &str = "normalized-sylvester-walsh-hadamard";
/// The only accepted application axis: the activation input dimension of each
/// rotated projection.
const SUPPORTED_AXIS: &str = "input-last-dimension";
/// The only accepted sign representation. Implicit sign conventions cannot be
/// validated against the shipped sign table, so they are refused.
const SUPPORTED_SIGN_MODE: &str = "explicit";
/// The validated Hadamard block sizes the upstream runtime has validated. Other block sizes
/// have no parity reference, so accepting them would be unvalidated guessing.
pub(crate) const SUPPORTED_BLOCK_SIZES: [u32; 4] = [512, 1024, 2048, 4096];
/// The only supported GDN activation layout. Ungrouped folded GDN output is
/// unimplemented upstream and has no parity reference here.
pub(crate) const SUPPORTED_GDN_LAYOUT_IS_GROUPED: bool = true;

/// Bounded document size guard for the manifest itself.
pub(crate) const MAXIMUM_HADAMARD_CONTRACT_BYTES: usize = 4 * 1024 * 1024;
/// Bound on the number of rotated or inverse tensor names. The shipped 27B
/// pack rotates 401 projections; the bound leaves orders of magnitude of
/// headroom while keeping pathological documents off the parse path.
const MAXIMUM_MANIFEST_NAME_COUNT: usize = 65_536;
/// Bound on how many distinct activation widths the sign table may declare.
const MAXIMUM_SIGN_TABLE_ENTRY_COUNT: usize = 64;
/// Bound on any single sign-vector width. The shipped widths are the hidden,
/// grouped-GDN-value, and MLP-intermediate dimensions of a 27B model; nothing
/// legitimate approaches this bound.
const MAXIMUM_SIGN_VECTOR_WIDTH: u32 = 1_048_576;

/// Parsed and validated Hadamard transform contract for a folded artifact.
///
/// The contract answers the serving path's three questions: which tensors are
/// rotated, which single tensor carries the inverse transform, and which
/// per-width sign vector the activation transform must use.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HadamardContract {
    block_size: u32,
    gdn_layout_is_grouped: bool,
    rotated_tensor_names: BTreeSet<String>,
    inverse_tensor_names: BTreeSet<String>,
    sign_vectors_by_width: BTreeMap<u32, Vec<i8>>,
}

impl HadamardContract {
    /// Parses and independently validates the manifest against the artifact's
    /// declared tensor names. Every manifest reference must resolve, the
    /// forward and inverse sets must not overlap, and the sign table must be
    /// internally consistent.
    pub fn from_json_bytes(
        contract_bytes: &[u8],
        artifact_tensor_names: &BTreeSet<String>,
    ) -> Result<Self, HadamardContractError> {
        if contract_bytes.len() > MAXIMUM_HADAMARD_CONTRACT_BYTES {
            return Err(HadamardContractError::ContractTooLarge {
                actual_contract_bytes: contract_bytes.len(),
                maximum_contract_bytes: MAXIMUM_HADAMARD_CONTRACT_BYTES,
            });
        }
        let document = serde_json::from_slice::<HadamardContractDocument>(contract_bytes)
            .map_err(HadamardContractError::DeserializeContract)?;
        Self::from_validated_document(document, artifact_tensor_names)
    }

    fn from_validated_document(
        document: HadamardContractDocument,
        artifact_tensor_names: &BTreeSet<String>,
    ) -> Result<Self, HadamardContractError> {
        if document.version != SUPPORTED_CONTRACT_VERSION {
            return Err(HadamardContractError::UnsupportedVersion {
                actual_version: document.version,
            });
        }
        if document.transform != SUPPORTED_TRANSFORM {
            return Err(HadamardContractError::UnsupportedTransform {
                actual_transform: document.transform,
            });
        }
        if document.axis != SUPPORTED_AXIS {
            return Err(HadamardContractError::UnsupportedAxis {
                actual_axis: document.axis,
            });
        }
        if document.sign_mode != SUPPORTED_SIGN_MODE {
            return Err(HadamardContractError::UnsupportedSignMode {
                actual_sign_mode: document.sign_mode,
            });
        }
        if !SUPPORTED_BLOCK_SIZES.contains(&document.block_size) {
            return Err(HadamardContractError::UnsupportedBlockSize {
                actual_block_size: document.block_size,
            });
        }
        if document.gdn_v_grouped != SUPPORTED_GDN_LAYOUT_IS_GROUPED {
            return Err(HadamardContractError::UnsupportedGdnLayout {
                actual_gdn_layout_is_grouped: document.gdn_v_grouped,
            });
        }
        let rotated_tensor_names =
            validated_manifest_set(&document.weight_names, artifact_tensor_names)?;
        let inverse_tensor_names =
            validated_manifest_set(&document.inverse_weight_names, artifact_tensor_names)?;
        if rotated_tensor_names
            .intersection(&inverse_tensor_names)
            .next()
            .is_some()
        {
            return Err(HadamardContractError::ForwardInverseOverlap);
        }
        let sign_values = document
            .sign_values
            .iter()
            .map(|sign_value| {
                normalized_sign_value(sign_value).ok_or(HadamardContractError::InvalidSignValue)
            })
            .collect::<Result<Vec<i8>, HadamardContractError>>()?;
        let sign_vectors_by_width = validated_sign_table(&document.sign_widths, &sign_values)?;
        Ok(Self {
            block_size: document.block_size,
            gdn_layout_is_grouped: document.gdn_v_grouped,
            rotated_tensor_names,
            inverse_tensor_names,
            sign_vectors_by_width,
        })
    }

    /// The validated Hadamard block size in activation elements.
    pub const fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Whether the GDN activation layout is the grouped layout the decode
    /// path already produces.
    pub const fn gdn_layout_is_grouped(&self) -> bool {
        self.gdn_layout_is_grouped
    }

    /// Whether the named tensor was folded in the rotated basis and therefore
    /// requires the matching activation-side transform.
    pub fn is_rotated(&self, tensor_name: &str) -> bool {
        self.rotated_tensor_names.contains(tensor_name)
    }

    /// Whether the named tensor carries the inverse transform, meaning the
    /// runtime applies the transform after the module rather than before.
    pub fn is_inverse(&self, tensor_name: &str) -> bool {
        self.inverse_tensor_names.contains(tensor_name)
    }

    /// Every tensor name declared with the forward (activation-input) transform.
    pub fn rotated_tensor_names(&self) -> &BTreeSet<String> {
        &self.rotated_tensor_names
    }

    /// Every tensor name declared with the inverse (post-module) transform.
    pub fn inverse_tensor_names(&self) -> &BTreeSet<String> {
        &self.inverse_tensor_names
    }

    /// The validated ±1 sign vector for an activation width, if declared.
    pub fn sign_vector(&self, activation_width: u32) -> Option<&[i8]> {
        self.sign_vectors_by_width
            .get(&activation_width)
            .map(Vec::as_slice)
    }
}

fn validated_manifest_set(
    tensor_names: &[String],
    artifact_tensor_names: &BTreeSet<String>,
) -> Result<BTreeSet<String>, HadamardContractError> {
    if tensor_names.is_empty() {
        return Err(HadamardContractError::EmptyManifest);
    }
    if tensor_names.len() > MAXIMUM_MANIFEST_NAME_COUNT {
        return Err(HadamardContractError::ManifestTooLarge {
            actual_name_count: tensor_names.len(),
            maximum_name_count: MAXIMUM_MANIFEST_NAME_COUNT,
        });
    }
    let validated_names = tensor_names
        .iter()
        .map(|tensor_name| {
            if !artifact_tensor_names.contains(tensor_name) {
                return Err(HadamardContractError::UnknownTensorReference {
                    tensor_name: tensor_name.clone(),
                });
            }
            Ok(tensor_name.clone())
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    if validated_names.len() != tensor_names.len() {
        return Err(HadamardContractError::DuplicateTensorReference);
    }
    Ok(validated_names)
}

/// The shipped manifest writes sign values as floats (`1.0`/`-1.0`) while
/// hand-written documents may use integers; both must denote exactly ±1.
fn normalized_sign_value(sign_value: &serde_json::Number) -> Option<i8> {
    if let Some(integer) = sign_value.as_i64() {
        return match integer {
            1 => Some(1),
            -1 => Some(-1),
            _ => None,
        };
    }
    match sign_value.as_f64() {
        Some(1.0) => Some(1),
        Some(-1.0) => Some(-1),
        _ => None,
    }
}

fn validated_sign_table(
    sign_widths: &[u32],
    sign_values: &[i8],
) -> Result<BTreeMap<u32, Vec<i8>>, HadamardContractError> {
    if sign_widths.is_empty() {
        return Err(HadamardContractError::EmptySignTable);
    }
    if sign_widths.len() > MAXIMUM_SIGN_TABLE_ENTRY_COUNT {
        return Err(HadamardContractError::SignTableTooLarge {
            actual_entry_count: sign_widths.len(),
            maximum_entry_count: MAXIMUM_SIGN_TABLE_ENTRY_COUNT,
        });
    }
    let mut sign_vectors_by_width = BTreeMap::new();
    let mut value_offset = 0;
    for sign_width in sign_widths {
        if *sign_width == 0 || *sign_width > MAXIMUM_SIGN_VECTOR_WIDTH {
            return Err(HadamardContractError::InvalidSignWidth {
                actual_sign_width: *sign_width,
            });
        }
        let width_end = value_offset + *sign_width as usize;
        if sign_values.len() < width_end {
            return Err(HadamardContractError::SignValuesTruncated {
                declared_total_width: value_offset + sign_widths.iter().sum::<u32>() as usize,
                actual_value_count: sign_values.len(),
            });
        }
        let sign_vector = &sign_values[value_offset..width_end];
        if sign_vector
            .iter()
            .any(|sign_value| *sign_value != 1 && *sign_value != -1)
        {
            return Err(HadamardContractError::InvalidSignValue);
        }
        if sign_vectors_by_width
            .insert(*sign_width, sign_vector.to_vec())
            .is_some()
        {
            return Err(HadamardContractError::DuplicateSignWidth {
                duplicate_sign_width: *sign_width,
            });
        }
        value_offset = width_end;
    }
    if value_offset != sign_values.len() {
        return Err(HadamardContractError::TrailingSignValues {
            consumed_value_count: value_offset,
            actual_value_count: sign_values.len(),
        });
    }
    Ok(sign_vectors_by_width)
}

/// Typed refusal for a folded-artifact transform contract this runtime does
/// not implement exactly.
#[derive(Debug, Error)]
pub enum HadamardContractError {
    #[error(
        "hadamard contract exceeds the {maximum_contract_bytes}-byte limit: {actual_contract_bytes}"
    )]
    ContractTooLarge {
        actual_contract_bytes: usize,
        maximum_contract_bytes: usize,
    },
    #[error("hadamard contract is not valid JSON: {0}")]
    DeserializeContract(#[source] serde_json::Error),
    #[error("hadamard contract declares unsupported schema version {actual_version}")]
    UnsupportedVersion { actual_version: u32 },
    #[error("hadamard contract declares unsupported transform {actual_transform:?}")]
    UnsupportedTransform { actual_transform: String },
    #[error("hadamard contract declares unsupported application axis {actual_axis:?}")]
    UnsupportedAxis { actual_axis: String },
    #[error("hadamard contract declares unsupported sign mode {actual_sign_mode:?}")]
    UnsupportedSignMode { actual_sign_mode: String },
    #[error("hadamard contract declares unsupported hadamard block size {actual_block_size}")]
    UnsupportedBlockSize { actual_block_size: u32 },
    #[error(
        "hadamard contract declares unsupported GDN layout (grouped={actual_gdn_layout_is_grouped})"
    )]
    UnsupportedGdnLayout { actual_gdn_layout_is_grouped: bool },
    #[error("hadamard contract declares an empty transform manifest")]
    EmptyManifest,
    #[error(
        "hadamard contract manifest exceeds the {maximum_name_count}-name limit: {actual_name_count}"
    )]
    ManifestTooLarge {
        actual_name_count: usize,
        maximum_name_count: usize,
    },
    #[error("hadamard contract references unknown tensor {tensor_name:?}")]
    UnknownTensorReference { tensor_name: String },
    #[error("hadamard contract manifest contains a duplicate tensor reference")]
    DuplicateTensorReference,
    #[error("hadamard contract forward and inverse transform manifests overlap")]
    ForwardInverseOverlap,
    #[error("hadamard contract declares an empty sign table")]
    EmptySignTable,
    #[error(
        "hadamard contract sign table exceeds the {maximum_entry_count}-entry limit: {actual_entry_count}"
    )]
    SignTableTooLarge {
        actual_entry_count: usize,
        maximum_entry_count: usize,
    },
    #[error("hadamard contract declares invalid sign width {actual_sign_width}")]
    InvalidSignWidth { actual_sign_width: u32 },
    #[error(
        "hadamard contract sign values are truncated: declared total width {declared_total_width}, actual values {actual_value_count}"
    )]
    SignValuesTruncated {
        declared_total_width: usize,
        actual_value_count: usize,
    },
    #[error("hadamard contract sign table contains a value outside the ±1 alphabet")]
    InvalidSignValue,
    #[error("hadamard contract sign table declares duplicate width {duplicate_sign_width}")]
    DuplicateSignWidth { duplicate_sign_width: u32 },
    #[error(
        "hadamard contract has trailing sign values: consumed {consumed_value_count}, actual {actual_value_count}"
    )]
    TrailingSignValues {
        consumed_value_count: usize,
        actual_value_count: usize,
    },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HadamardContractDocument {
    #[serde(rename = "prism.hadamard.version")]
    version: u32,
    #[serde(rename = "prism.hadamard.block_size")]
    block_size: u32,
    #[serde(rename = "prism.hadamard.transform")]
    transform: String,
    #[serde(rename = "prism.hadamard.axis")]
    axis: String,
    #[serde(rename = "prism.hadamard.sign_mode")]
    sign_mode: String,
    #[serde(rename = "prism.hadamard.weight_names")]
    weight_names: Vec<String>,
    #[serde(rename = "prism.hadamard.inverse_weight_names")]
    inverse_weight_names: Vec<String>,
    #[serde(rename = "prism.hadamard.sign_widths")]
    sign_widths: Vec<u32>,
    #[serde(rename = "prism.hadamard.sign_values")]
    sign_values: Vec<serde_json::Number>,
    #[serde(rename = "prism.hadamard.gdn_v_grouped")]
    gdn_v_grouped: bool,
}
