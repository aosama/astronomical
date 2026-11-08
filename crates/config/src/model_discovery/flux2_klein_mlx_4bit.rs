//! Shallow discovery for the 4-bit MLX build of FLUX.2 Klein 4B
//! (`mlx-community/flux2-klein-4b-4bit`, issue #1904).
//!
//! The build is a Diffusers tree quantized by mflux, without `model_index.json`
//! or configuration documents. Its transformer weight index identifies it: the
//! quantization metadata mflux writes, plus exactly Klein 4B's 5 double-stream
//! and 20 single-stream blocks (Klein 9B has more, so it is not mistaken for
//! this build). The license comes from the model card's front matter. The
//! serving validator later proves every tensor shape before a load.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::Deserialize;

use super::ModelLicense;
use super::bounded_artifact_file;
use super::classified_artifacts;
use super::flux2_klein::{
    Flux2KleinDirectoryEvidence, Flux2KleinDirectoryVerificationError, flux2_klein_capabilities,
    is_safe_safetensors_path, required_weight_size,
};
use super::model_card_front_matter::has_front_matter_value;

pub(super) const CANONICAL_MODEL_ID: &str = "flux2-klein-4b-4bit";
pub(super) const PROVIDER_MODEL_ID: &str = "mlx-community/flux2-klein-4b-4bit";
/// The file that identifies the build, both on disk and in a Hub manifest.
pub const FLUX2_KLEIN_MLX_4BIT_TRANSFORMER_INDEX_FILE_NAME: &str =
    "transformer/model.safetensors.index.json";
const TEXT_ENCODER_INDEX_FILE_NAME: &str = "text_encoder/model.safetensors.index.json";
const VAE_INDEX_FILE_NAME: &str = "vae/model.safetensors.index.json";
const MODEL_CARD_FILE_NAME: &str = "README.md";
const REQUIRED_TOKENIZER_FILES: [&str; 3] = [
    "tokenizer/chat_template.jinja",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
];
const MAXIMUM_INDEX_BYTES: u64 = 4 * 1024 * 1024;
const MAXIMUM_MODEL_CARD_BYTES: u64 = 256 * 1024;
const MAXIMUM_TOKENIZER_BYTES: u64 = 64 * 1024 * 1024;
const QUANTIZATION_LEVEL: &str = "4";
const QUANTIZATION_GROUP_SIZE: &str = "64";
const KLEIN_4B_DOUBLE_STREAM_BLOCK_COUNT: usize = 5;
const KLEIN_4B_SINGLE_STREAM_BLOCK_COUNT: usize = 20;
const KLEIN_4B_TEXT_ENCODER_LAYER_COUNT: usize = 36;

/// The index mflux writes beside each component's shards. Fields are optional
/// so another tool's index parses and is simply not classified as this build.
#[derive(Deserialize)]
struct MfluxWeightIndex {
    #[serde(default)]
    metadata: MfluxQuantizationMetadata,
    #[serde(default)]
    weight_map: BTreeMap<String, String>,
}

#[derive(Default, Deserialize)]
struct MfluxQuantizationMetadata {
    quantization_level: Option<String>,
    quantization_group_size: Option<String>,
    mflux_version: Option<String>,
}

impl MfluxWeightIndex {
    fn has_reviewed_quantization(&self) -> bool {
        self.metadata.quantization_level.as_deref() == Some(QUANTIZATION_LEVEL)
            && self.metadata.quantization_group_size.as_deref() == Some(QUANTIZATION_GROUP_SIZE)
            && self
                .metadata
                .mflux_version
                .as_deref()
                .is_some_and(|mflux_version| !mflux_version.is_empty())
    }

    fn block_indices(&self, block_prefix: &str) -> BTreeSet<usize> {
        let mut block_indices = BTreeSet::new();
        for tensor_name in self.weight_map.keys() {
            let block_index = tensor_name
                .strip_prefix(block_prefix)
                .and_then(|block_suffix| block_suffix.split_once('.'))
                .and_then(|(block_index, _)| block_index.parse::<usize>().ok());
            if let Some(block_index) = block_index {
                block_indices.insert(block_index);
            }
        }
        block_indices
    }
}

/// Whether a transformer weight index is this build's. Shared by disk
/// discovery and the download preflight so the two cannot drift.
///
/// # Errors
/// Returns the parse failure when the bytes are not JSON of the index shape.
pub fn classifies_transformer_index(index_bytes: &[u8]) -> Result<bool, serde_json::Error> {
    let transformer_index: MfluxWeightIndex = serde_json::from_slice(index_bytes)?;
    Ok(is_klein_4b_transformer_index(&transformer_index))
}

fn is_klein_4b_transformer_index(transformer_index: &MfluxWeightIndex) -> bool {
    transformer_index.has_reviewed_quantization()
        && transformer_index.block_indices("transformer_blocks.")
            == (0..KLEIN_4B_DOUBLE_STREAM_BLOCK_COUNT).collect()
        && transformer_index.block_indices("single_transformer_blocks.")
            == (0..KLEIN_4B_SINGLE_STREAM_BLOCK_COUNT).collect()
        && transformer_index
            .weight_map
            .contains_key("time_guidance_embed.linear_1.weight")
}

/// Whether `directory` carries the build's transformer index. Such a folder is
/// a pipeline root even without `model_index.json`, so scans stop there.
pub(super) fn has_build_marker(directory: &Path) -> bool {
    directory
        .join(FLUX2_KLEIN_MLX_4BIT_TRANSFORMER_INDEX_FILE_NAME)
        .is_file()
}

/// Whether `directory` holds this build. An unreadable or foreign index is not
/// this build, so classification can still fall back to `config.json`.
pub(super) fn classifies_directory(directory: &Path) -> bool {
    let Ok(index_bytes) = bounded_artifact_file::read_bounded_nonempty_file(
        &directory.join(FLUX2_KLEIN_MLX_4BIT_TRANSFORMER_INDEX_FILE_NAME),
        MAXIMUM_INDEX_BYTES,
    ) else {
        return false;
    };
    classifies_transformer_index(&index_bytes).unwrap_or(false)
}

/// Rereads one exact directory as the 4-bit build.
pub(super) fn verify_model_directory_evidence(
    model_directory: &Path,
) -> Result<Flux2KleinDirectoryEvidence, Flux2KleinDirectoryVerificationError> {
    let transformer_index = read_index(
        model_directory,
        FLUX2_KLEIN_MLX_4BIT_TRANSFORMER_INDEX_FILE_NAME,
        "transformer",
    )?;
    require_index(
        is_klein_4b_transformer_index(&transformer_index),
        "transformer",
    )?;
    let text_encoder_index = read_index(
        model_directory,
        TEXT_ENCODER_INDEX_FILE_NAME,
        "text encoder",
    )?;
    require_index(
        text_encoder_index.has_reviewed_quantization()
            && text_encoder_index.block_indices("layers.")
                == (0..KLEIN_4B_TEXT_ENCODER_LAYER_COUNT).collect(),
        "text encoder",
    )?;
    let vae_index = read_index(model_directory, VAE_INDEX_FILE_NAME, "VAE")?;
    require_index(
        vae_index.has_reviewed_quantization()
            && vae_index.weight_map.contains_key("decoder.conv_in.weight"),
        "VAE",
    )?;
    validate_model_card(model_directory)?;
    validate_tokenizer_files(model_directory)?;
    let model_size_bytes = measure_weight_bytes(
        model_directory,
        [
            ("transformer", "transformer", &transformer_index),
            ("text_encoder", "text encoder", &text_encoder_index),
            ("vae", "VAE", &vae_index),
        ],
    )?;
    let (provider_model_id, revision) = provenance(model_directory)?;
    Ok(Flux2KleinDirectoryEvidence {
        canonical_model_id: CANONICAL_MODEL_ID.to_owned(),
        provider_model_id,
        revision,
        license: ModelLicense::Apache20,
        capabilities: flux2_klein_capabilities(),
        model_size_bytes,
    })
}

fn read_index(
    model_directory: &Path,
    index_file_name: &str,
    component: &'static str,
) -> Result<MfluxWeightIndex, Flux2KleinDirectoryVerificationError> {
    bounded_artifact_file::read_json(&model_directory.join(index_file_name), MAXIMUM_INDEX_BYTES)
        .map_err(
            |_| Flux2KleinDirectoryVerificationError::InvalidQuantizedWeightIndex { component },
        )
}

fn require_index(
    index_is_reviewed: bool,
    component: &'static str,
) -> Result<(), Flux2KleinDirectoryVerificationError> {
    index_is_reviewed
        .then_some(())
        .ok_or(Flux2KleinDirectoryVerificationError::InvalidQuantizedWeightIndex { component })
}

/// The build ships no license file; its model card declares Apache-2.0 and
/// names the BF16 model it was quantized from.
fn validate_model_card(model_directory: &Path) -> Result<(), Flux2KleinDirectoryVerificationError> {
    let model_card_bytes = bounded_artifact_file::read_bounded_nonempty_file(
        &model_directory.join(MODEL_CARD_FILE_NAME),
        MAXIMUM_MODEL_CARD_BYTES,
    )
    .map_err(|_| Flux2KleinDirectoryVerificationError::InvalidLicense)?;
    let model_card = std::str::from_utf8(&model_card_bytes)
        .map_err(|_| Flux2KleinDirectoryVerificationError::InvalidLicense)?;
    (has_front_matter_value(model_card, "license", "apache-2.0")
        && has_front_matter_value(
            model_card,
            "base_model",
            "black-forest-labs/FLUX.2-klein-4B",
        ))
    .then_some(())
    .ok_or(Flux2KleinDirectoryVerificationError::InvalidLicense)
}

fn validate_tokenizer_files(
    model_directory: &Path,
) -> Result<(), Flux2KleinDirectoryVerificationError> {
    for tokenizer_file in REQUIRED_TOKENIZER_FILES {
        bounded_artifact_file::read_bounded_nonempty_file(
            &model_directory.join(tokenizer_file),
            MAXIMUM_TOKENIZER_BYTES,
        )
        .map_err(
            |_| Flux2KleinDirectoryVerificationError::MissingOrInvalidSidecar {
                sidecar: tokenizer_file,
            },
        )?;
    }
    Ok(())
}

/// Bytes on disk across every shard the three indexes name.
fn measure_weight_bytes(
    model_directory: &Path,
    component_indexes: [(&'static str, &'static str, &MfluxWeightIndex); 3],
) -> Result<u64, Flux2KleinDirectoryVerificationError> {
    let mut model_size_bytes = 0_u64;
    for (component_folder, component, component_index) in component_indexes {
        let shard_file_names = component_index.weight_map.values().collect::<BTreeSet<_>>();
        for shard_file_name in shard_file_names {
            if !is_safe_safetensors_path(shard_file_name) {
                return Err(
                    Flux2KleinDirectoryVerificationError::InvalidQuantizedWeightIndex { component },
                );
            }
            let shard_path = model_directory.join(component_folder).join(shard_file_name);
            model_size_bytes = model_size_bytes
                .checked_add(required_weight_size(&shard_path, component)?)
                .ok_or(Flux2KleinDirectoryVerificationError::ModelSizeOverflow)?;
        }
    }
    Ok(model_size_bytes)
}

/// The library's provenance record when Astronomical downloaded the build,
/// else the Hugging Face snapshot the directory sits in. A record naming a
/// repository other than this build's is refused here, because serving would
/// refuse that provenance at load.
fn provenance(
    model_directory: &Path,
) -> Result<(String, String), Flux2KleinDirectoryVerificationError> {
    if let Some((provider_model_id, revision)) =
        classified_artifacts::immutable_model_provenance(model_directory)
    {
        return names_this_build(&provider_model_id)
            .then_some((provider_model_id, revision))
            .ok_or(Flux2KleinDirectoryVerificationError::UnsupportedProviderModelId);
    }
    let revision = classified_artifacts::immutable_file_revision(
        model_directory,
        FLUX2_KLEIN_MLX_4BIT_TRANSFORMER_INDEX_FILE_NAME,
    )
    .ok_or(Flux2KleinDirectoryVerificationError::MissingRevision)?;
    Ok((PROVIDER_MODEL_ID.to_owned(), revision))
}

fn names_this_build(provider_model_id: &str) -> bool {
    provider_model_id.rsplit('/').next() == Some(CANONICAL_MODEL_ID)
}

#[cfg(test)]
#[path = "flux2_klein_mlx_4bit_tests.rs"]
mod tests;
