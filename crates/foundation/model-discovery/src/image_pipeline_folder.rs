//! What every image family's folder check shares (issue #2473, Step 5): the
//! evidence a verified folder yields, how weight files are sized, which shard
//! paths are safe to follow, and how verified evidence becomes a discovered
//! model. Each family keeps its own documents and geometry rules.

use std::fs::{self, File};
use std::io::Read;
use std::path::{Component, Path};

use serde::Deserialize;

use super::{
    DiscoveredModel, ImageGenerationCapabilities, ModelCapabilities, ModelFamily, ModelLicense,
    bounded_artifact_file, classified_artifacts,
};

/// Trusted shallow evidence reread from one selected image folder.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImageDirectoryEvidence {
    pub canonical_model_id: String,
    pub provider_model_id: String,
    pub revision: String,
    pub license: ModelLicense,
    pub capabilities: ImageGenerationCapabilities,
    pub model_size_bytes: u64,
}

/// A verified image folder as a discovered model. It is served under the
/// canonical id its evidence verifies, never the folder's name.
#[must_use]
pub fn discovered_image_model(
    model_directory: &Path,
    model_family: ModelFamily,
    verified_evidence: ImageDirectoryEvidence,
) -> DiscoveredModel {
    DiscoveredModel {
        model_id: verified_evidence.canonical_model_id,
        provider_model_id: Some(verified_evidence.provider_model_id),
        model_family,
        revision: verified_evidence.revision,
        model_directory: model_directory.to_path_buf(),
        capabilities: ModelCapabilities::ImageGeneration(verified_evidence.capabilities),
        license: Some(verified_evidence.license),
        model_size_bytes: verified_evidence.model_size_bytes,
    }
}

/// The size of a weight file that exists and is not empty.
#[must_use]
pub fn nonempty_file_size(weight_path: &Path) -> Option<u64> {
    let weight_metadata = fs::metadata(weight_path).ok()?;
    (weight_metadata.is_file() && weight_metadata.len() > 0).then_some(weight_metadata.len())
}

/// The bytes of tensor data a safetensors file holds after its header, or
/// `None` when the file is unreadable or holds no tensor data.
#[must_use]
pub fn safetensors_payload_size(weight_path: &Path) -> Option<u64> {
    let weight_file_size_bytes = fs::metadata(weight_path).ok()?.len();
    let mut weight_file = File::open(weight_path).ok()?;
    let mut header_length_bytes = [0_u8; 8];
    weight_file.read_exact(&mut header_length_bytes).ok()?;
    let header_size_bytes = u64::from_le_bytes(header_length_bytes);
    weight_file_size_bytes
        .checked_sub(8)
        .and_then(|remaining_bytes| remaining_bytes.checked_sub(header_size_bytes))
        .filter(|payload_size_bytes| *payload_size_bytes > 0)
}

/// Whether a shard path from a weight index stays inside the folder: a
/// relative `.safetensors` path of plain folder names.
#[must_use]
pub fn is_safe_safetensors_path(shard_path: &str) -> bool {
    let shard_file_path = Path::new(shard_path);
    !shard_path.is_empty()
        && !shard_path.contains('\\')
        && !shard_file_path.is_absolute()
        && shard_file_path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
        && shard_file_path
            .extension()
            .and_then(|extension| extension.to_str())
            == Some("safetensors")
}

/// Where a pipeline folder came from: the provider id and revision the
/// library recorded, or the family's provider id and the revision Hugging
/// Face recorded for `model_index.json`. `None` when no immutable revision
/// is on record.
#[must_use]
pub fn pipeline_provenance(
    model_directory: &Path,
    family_provider_model_id: &str,
) -> Option<(String, String)> {
    match classified_artifacts::immutable_model_provenance(model_directory) {
        Some(library_provenance) => Some(library_provenance),
        None => {
            let revision =
                classified_artifacts::immutable_file_revision(model_directory, "model_index.json")?;
            Some((family_provider_model_id.to_owned(), revision))
        }
    }
}

/// The settings of a reviewed flow-match Euler scheduler that differ between
/// image families. Every other setting is the same for all of them.
pub struct FlowMatchSchedulerProfile {
    pub max_image_seq_len: u32,
    pub max_shift: f64,
    pub shift: f64,
    pub shift_terminal: Option<f64>,
}

/// The scheduler settings an image family's `scheduler_config.json` holds.
#[derive(Deserialize)]
struct SchedulerGeometry {
    #[serde(rename = "_class_name")]
    class_name: String,
    base_image_seq_len: u32,
    base_shift: f64,
    invert_sigmas: bool,
    max_image_seq_len: u32,
    max_shift: f64,
    num_train_timesteps: u32,
    shift: f64,
    shift_terminal: Option<f64>,
    stochastic_sampling: bool,
    time_shift_type: String,
    use_beta_sigmas: bool,
    use_dynamic_shifting: bool,
    use_exponential_sigmas: bool,
    use_karras_sigmas: bool,
}

/// Whether the folder's scheduler is the reviewed flow-match Euler scheduler
/// with the family's profile.
#[must_use]
pub fn is_reviewed_flow_match_scheduler(
    model_directory: &Path,
    maximum_document_bytes: u64,
    scheduler_profile: &FlowMatchSchedulerProfile,
) -> bool {
    let Ok(geometry) = bounded_artifact_file::read_json::<SchedulerGeometry>(
        &model_directory.join("scheduler/scheduler_config.json"),
        maximum_document_bytes,
    ) else {
        return false;
    };
    geometry.class_name == "FlowMatchEulerDiscreteScheduler"
        && geometry.base_image_seq_len == 256
        && geometry.base_shift == 0.5
        && !geometry.invert_sigmas
        && geometry.max_image_seq_len == scheduler_profile.max_image_seq_len
        && geometry.max_shift == scheduler_profile.max_shift
        && geometry.num_train_timesteps == 1_000
        && geometry.shift == scheduler_profile.shift
        && geometry.shift_terminal == scheduler_profile.shift_terminal
        && !geometry.stochastic_sampling
        && geometry.time_shift_type == "exponential"
        && !geometry.use_beta_sigmas
        && geometry.use_dynamic_shifting
        && !geometry.use_exponential_sigmas
        && !geometry.use_karras_sigmas
}
