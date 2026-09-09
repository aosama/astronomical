//! Revision-manifest publication and completeness validation.
//!
//! A revision is complete only when every manifest-declared file exists,
//! matches its declared byte count, and hashes to its declared content —
//! with the resident weight bundle as the single-copy layout marker. These
//! gates run at preparation time and before reusing an existing revision.

use std::{fs, path::Path};

use astronomical_model_serving::QuantizedExpertLayerPlan;

use crate::aligned_expert_pack::AlignedExpertPackError;
use crate::aligned_expert_pack_preparer::AlignedExpertPackPreparationError;
use crate::per_expert_pack::{
    per_expert_pack_relative_path, read_per_expert_pack_header, validate_per_expert_pack_header,
    validate_per_expert_pack_payload,
};
use crate::revision_manifest::{StreamingModelManifest, expert_file_entry, sha256_hex_digest};
use crate::streaming_model_preparer::quantization_mode_name;

/// Writes the revision manifest: every published file declared with its
/// measured size and content hash, so integrity verification works
/// standalone on a downloaded revision with no source artifact present.
pub(super) fn write_manifest(
    source_model_id: &str,
    streaming_model_id: &str,
    model_revision: &str,
    layer_plans: &[QuantizedExpertLayerPlan],
    staging_model_directory: &Path,
    mut report_hashing_progress: impl FnMut(usize, usize),
) -> Result<(), AlignedExpertPackPreparationError> {
    let first_layer_plan = layer_plans
        .first()
        .ok_or(AlignedExpertPackPreparationError::EmptyLayerPlans)?;
    let total_expert_file_count: usize = layer_plans
        .iter()
        .map(|layer_plan| layer_plan.expert_capacity)
        .sum();
    let mut resident_files = Vec::new();
    super::streaming_model_resident_bundle::append_declared_resident_files(
        staging_model_directory,
        staging_model_directory,
        &mut resident_files,
    )?;
    resident_files.sort_by(|left, right| left.file_name.cmp(&right.file_name));
    let mut expert_files = Vec::new();
    for (layer_index, layer_plan) in layer_plans.iter().enumerate() {
        for expert_id in 0..layer_plan.expert_capacity {
            expert_files.push(expert_file_entry(
                staging_model_directory,
                layer_index,
                expert_id,
                per_expert_pack_relative_path(layer_index, expert_id),
            )?);
            let hashed_file_count = expert_files.len();
            if hashed_file_count == 1
                || hashed_file_count == total_expert_file_count
                || hashed_file_count.is_multiple_of(256)
            {
                report_hashing_progress(hashed_file_count, total_expert_file_count);
            }
        }
    }
    StreamingModelManifest::new(
        streaming_model_id,
        source_model_id,
        model_revision,
        first_layer_plan.expert_capacity,
        quantization_mode_name(first_layer_plan),
        first_layer_plan.quantization_bits,
        first_layer_plan.quantization_group_size,
        resident_files,
        expert_files,
    )
    .write_to_revision_directory(staging_model_directory)?;
    Ok(())
}

/// Validates an existing revision for reuse: manifest identity, the
/// resident-bundle marker, every declared file's hash, and every expert
/// header against the planning geometry.
pub(super) fn validate_complete_streaming_model(
    _source_model_id: &str,
    streaming_model_id: &str,
    model_revision: &str,
    layer_plans: &[QuantizedExpertLayerPlan],
    model_directory: &Path,
) -> Result<(), AlignedExpertPackPreparationError> {
    let streaming_model_manifest =
        StreamingModelManifest::read_from_revision_directory(model_directory)?;
    if streaming_model_manifest.model_id != streaming_model_id
        || streaming_model_manifest.model_revision != model_revision
    {
        return Err(AlignedExpertPackPreparationError::InvalidExistingRevision {
            revision_directory: model_directory.to_path_buf(),
        });
    }
    // The single-copy layout is identified by the resident weight bundle;
    // a revision that still carries the superseded shard layout must be
    // rebuilt rather than reused.
    if !streaming_model_manifest
        .resident_files
        .iter()
        .any(|resident_file| resident_file.file_name == "resident.safetensors")
    {
        return Err(AlignedExpertPackPreparationError::InvalidExistingRevision {
            revision_directory: model_directory.to_path_buf(),
        });
    }
    for resident_file in &streaming_model_manifest.resident_files {
        let resident_file_path = model_directory.join(&resident_file.file_name);
        if !resident_file_path.is_file()
            || fs::metadata(&resident_file_path)?.len() != resident_file.expected_file_byte_count
            || sha256_hex_digest(&resident_file_path)? != resident_file.content_sha256
        {
            return Err(AlignedExpertPackPreparationError::InvalidExistingRevision {
                revision_directory: model_directory.to_path_buf(),
            });
        }
    }
    let total_expert_file_count: usize = layer_plans
        .iter()
        .map(|layer_plan| layer_plan.expert_capacity)
        .sum();
    if streaming_model_manifest.expert_files.len() != total_expert_file_count {
        return Err(AlignedExpertPackPreparationError::InvalidExistingRevision {
            revision_directory: model_directory.to_path_buf(),
        });
    }
    for expert_file in &streaming_model_manifest.expert_files {
        let expert_file_path = model_directory.join(&expert_file.file_name);
        if sha256_hex_digest(&expert_file_path)? != expert_file.content_sha256 {
            return Err(AlignedExpertPackPreparationError::InvalidExistingRevision {
                revision_directory: model_directory.to_path_buf(),
            });
        }
        let layer_plan = layer_plans.get(expert_file.layer_index).ok_or_else(|| {
            AlignedExpertPackPreparationError::InvalidExistingRevision {
                revision_directory: model_directory.to_path_buf(),
            }
        })?;
        read_validated_expert_header(
            &expert_file_path,
            layer_plan,
            streaming_model_id,
            model_revision,
            expert_file.layer_index,
            expert_file.expert_id,
        )?;
    }
    Ok(())
}

pub(super) fn read_validated_expert_header(
    pack_path: &Path,
    layer_plan: &QuantizedExpertLayerPlan,
    streaming_model_id: &str,
    model_revision: &str,
    layer_index: usize,
    expert_id: usize,
) -> Result<crate::per_expert_pack::PerExpertPackHeader, AlignedExpertPackError> {
    let pack_header = read_per_expert_pack_header(pack_path)?;
    validate_per_expert_pack_header(
        pack_path,
        &pack_header,
        layer_plan,
        streaming_model_id,
        model_revision,
        layer_index,
        expert_id,
    )?;
    validate_per_expert_pack_payload(pack_path, &pack_header, layer_plan)?;
    Ok(pack_header)
}
