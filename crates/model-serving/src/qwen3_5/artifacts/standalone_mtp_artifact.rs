//! Descriptor-backed validation for independently packaged Qwen MTP weights.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::artifact_validation::{
    ValidatedRequiredFile, ValidatedSafetensorsSource, read_bounded_required_file_bytes,
    validate_required_file,
};
use crate::{
    ArtifactValidationError, OptiQQuantizationProfile, Qwen3_5Config, RequiredFileProfile,
    TensorDeclarationOrigin, TensorFeature, TensorInventory, TensorInventoryError, TensorLocation,
    TensorProfile, TensorSemanticRole, TensorSourceId,
};

use super::mtp_storage_fingerprint::standalone_mtp_storage_fingerprint;
use super::{
    Qwen3_5StandaloneMtpConfig, Qwen3_5StandaloneMtpConfigError, StandaloneMtpNamespaceError,
    normalize_qwen3_5_standalone_mtp_tensor_name, qwen3_5_mtp_tensor_profiles,
};

const MAXIMUM_CONFIG_BYTES: u64 = 4 * 1024 * 1024;
const MAXIMUM_INDEX_BYTES: u64 = 16 * 1024 * 1024;

/// Validates one standalone MTP package against exact executor tensor profiles.
pub struct Qwen3_5StandaloneMtpArtifactValidator<'config> {
    target_config: &'config Qwen3_5Config,
    model_id: String,
    discovered_revision: String,
}

impl<'config> Qwen3_5StandaloneMtpArtifactValidator<'config> {
    #[must_use]
    pub fn new(
        target_config: &'config Qwen3_5Config,
        model_id: impl Into<String>,
        discovered_revision: impl Into<String>,
    ) -> Self {
        Self {
            target_config,
            model_id: model_id.into(),
            discovered_revision: discovered_revision.into(),
        }
    }

    pub fn validate(
        self,
        model_directory: impl AsRef<Path>,
    ) -> Result<ValidatedQwen3_5StandaloneMtpArtifact, Qwen3_5StandaloneMtpArtifactValidationError>
    {
        let model_directory = model_directory.as_ref();
        let config_file = validate_file(model_directory, "config.json")?;
        let tokenizer_file = validate_file(model_directory, "tokenizer.json")?;
        let config_bytes = read_bounded_required_file_bytes(&config_file, MAXIMUM_CONFIG_BYTES)?;
        let tokenizer_bytes = read_bounded_required_file_bytes(&tokenizer_file, 32 * 1024 * 1024)?;
        let config = Qwen3_5StandaloneMtpConfig::from_json_bytes(&config_bytes)?;
        let source_declarations = resolve_source_declarations(model_directory)?;
        let mut sources = Vec::with_capacity(source_declarations.len());
        for (source_index, source_declaration) in source_declarations.iter().enumerate() {
            let source_id = TensorSourceId::new(
                u32::try_from(source_index + 1)
                    .map_err(|_| Qwen3_5StandaloneMtpArtifactValidationError::TooManySources)?,
            );
            let required_file = validate_file(model_directory, &source_declaration.file_name)?;
            let source = ValidatedSafetensorsSource::parse(source_id, required_file)?;
            if let Some(indexed_stored_names) = source_declaration.indexed_stored_names.as_ref() {
                let physical_stored_names = source
                    .stored_tensor_names()
                    .cloned()
                    .collect::<BTreeSet<_>>();
                if &physical_stored_names != indexed_stored_names {
                    return Err(
                        Qwen3_5StandaloneMtpArtifactValidationError::IndexInventoryMismatch {
                            file_name: source_declaration.file_name.clone(),
                        },
                    );
                }
            }
            sources.push(source);
        }

        let mut inventory = TensorInventory::new();
        let mut canonical_tensor_names = BTreeSet::new();
        for source in &sources {
            for stored_name in source.stored_tensor_names() {
                let canonical_name = normalize_qwen3_5_standalone_mtp_tensor_name(stored_name)?;
                canonical_tensor_names.insert(canonical_name.clone());
                inventory.insert(TensorLocation::new(
                    canonical_name,
                    stored_name.clone(),
                    source.source_id(),
                    TensorSemanticRole::MultiTokenPrediction,
                    TensorDeclarationOrigin::StandaloneAuxiliary,
                    Some(TensorFeature::MultiTokenPrediction),
                ))?;
            }
        }

        // The current target-local storage resolver already supports mixed native
        // modules; cloning prevents standalone evidence from mutating the target.
        let mut profile_config = self.target_config.clone();
        apply_standalone_storage_contract(
            &mut profile_config,
            &canonical_tensor_names,
            config.quantization_profile(),
        )?;
        let tensor_profiles = qwen3_5_mtp_tensor_profiles(&profile_config);
        for source in &sources {
            source.validate_inventory_profiles(&inventory, &tensor_profiles)?;
        }
        let total_payload_bytes = sources.iter().try_fold(0_u64, |total, source| {
            total
                .checked_add(source.payload_bytes())
                .ok_or(Qwen3_5StandaloneMtpArtifactValidationError::PayloadOverflow)
        })?;
        let storage_fingerprint = standalone_mtp_storage_fingerprint(
            &self.discovered_revision,
            &inventory,
            &tensor_profiles,
            &sources,
        );
        Ok(ValidatedQwen3_5StandaloneMtpArtifact {
            model_id: self.model_id,
            discovered_revision: self.discovered_revision,
            config,
            tokenizer_bytes,
            storage_fingerprint,
            total_payload_bytes,
            tensor_profiles,
            inventory,
            sources,
            binding_config: profile_config,
            model_directory: model_directory.to_path_buf(),
            _config_file: config_file,
            _tokenizer_file: tokenizer_file,
        })
    }
}

/// Independently validated standalone MTP artifact and retained source ownership.
pub struct ValidatedQwen3_5StandaloneMtpArtifact {
    model_id: String,
    discovered_revision: String,
    config: Qwen3_5StandaloneMtpConfig,
    tokenizer_bytes: Vec<u8>,
    storage_fingerprint: String,
    total_payload_bytes: u64,
    tensor_profiles: Vec<TensorProfile>,
    inventory: TensorInventory,
    sources: Vec<ValidatedSafetensorsSource>,
    binding_config: Qwen3_5Config,
    model_directory: PathBuf,
    _config_file: ValidatedRequiredFile,
    _tokenizer_file: ValidatedRequiredFile,
}

impl ValidatedQwen3_5StandaloneMtpArtifact {
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }
    #[must_use]
    pub fn discovered_revision(&self) -> &str {
        &self.discovered_revision
    }
    #[must_use]
    pub const fn config(&self) -> &Qwen3_5StandaloneMtpConfig {
        &self.config
    }
    #[must_use]
    pub fn tokenizer_bytes(&self) -> &[u8] {
        &self.tokenizer_bytes
    }
    #[must_use]
    pub fn storage_fingerprint(&self) -> &str {
        &self.storage_fingerprint
    }
    #[must_use]
    pub const fn total_payload_bytes(&self) -> u64 {
        self.total_payload_bytes
    }
    #[must_use]
    pub fn tensor_profiles(&self) -> &[TensorProfile] {
        &self.tensor_profiles
    }
    #[must_use]
    pub const fn tensor_inventory(&self) -> &TensorInventory {
        &self.inventory
    }
    #[must_use]
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    pub(crate) fn into_binding_parts(
        self,
    ) -> Result<Qwen3_5StandaloneMtpBindingParts, ArtifactValidationError> {
        let mut source_files = Vec::with_capacity(self.sources.len());
        let mut source_file_name_by_source_id = BTreeMap::new();
        for source in self.sources {
            let source_id = source.source_id();
            source_file_name_by_source_id.insert(source_id, source.file_name().to_owned());
            source_files.push((source_id, source.into_validated_weights_file()?));
        }
        Ok(Qwen3_5StandaloneMtpBindingParts {
            binding_config: self.binding_config,
            tensor_inventory: self.inventory,
            source_files,
            model_directory: self.model_directory,
            source_file_name_by_source_id,
        })
    }
}

pub(crate) struct Qwen3_5StandaloneMtpBindingParts {
    pub(crate) binding_config: Qwen3_5Config,
    pub(crate) tensor_inventory: TensorInventory,
    pub(crate) source_files: Vec<(TensorSourceId, crate::ValidatedWeightsFile)>,
    pub(crate) model_directory: PathBuf,
    pub(crate) source_file_name_by_source_id: BTreeMap<TensorSourceId, String>,
}

/// Deep standalone artifact validation failure.
#[derive(Debug, thiserror::Error)]
pub enum Qwen3_5StandaloneMtpArtifactValidationError {
    #[error("standalone MTP file validation failed")]
    Artifact(#[from] ArtifactValidationError),
    #[error("standalone MTP config validation failed")]
    Config(#[from] Qwen3_5StandaloneMtpConfigError),
    #[error("standalone MTP tensor namespace validation failed")]
    Namespace(#[from] StandaloneMtpNamespaceError),
    #[error("standalone MTP tensor inventory is conflicting")]
    Inventory(#[from] TensorInventoryError),
    #[error("standalone MTP package must contain exactly one single-file or indexed layout")]
    ConflictingOrMissingLayout,
    #[error("standalone MTP index is malformed or empty")]
    MalformedIndex,
    #[error("standalone MTP index does not match physical tensors in '{file_name}'")]
    IndexInventoryMismatch { file_name: String },
    #[error("standalone MTP affine companions require declared quantization geometry")]
    MissingQuantizationGeometry,
    #[error("standalone MTP source count exceeds the supported range")]
    TooManySources,
    #[error("standalone MTP payload accounting overflowed")]
    PayloadOverflow,
}

fn validate_file(
    model_directory: &Path,
    relative_file_name: &str,
) -> Result<ValidatedRequiredFile, ArtifactValidationError> {
    validate_required_file(
        model_directory,
        &RequiredFileProfile {
            file_name: relative_file_name.to_owned(),
            size_bytes: 0,
        },
    )
}

fn resolve_source_declarations(
    model_directory: &Path,
) -> Result<Vec<StandaloneSourceDeclaration>, Qwen3_5StandaloneMtpArtifactValidationError> {
    let has_single_file = model_directory.join("model.safetensors").is_file();
    let has_index = model_directory
        .join("model.safetensors.index.json")
        .is_file();
    match (has_single_file, has_index) {
        (true, false) => Ok(vec![StandaloneSourceDeclaration {
            file_name: "model.safetensors".to_owned(),
            indexed_stored_names: None,
        }]),
        (false, true) | (true, true) => {
            let index_file = validate_file(model_directory, "model.safetensors.index.json")?;
            let index_bytes = read_bounded_required_file_bytes(&index_file, MAXIMUM_INDEX_BYTES)?;
            let index: StandaloneIndex = serde_json::from_slice(&index_bytes)
                .map_err(|_| Qwen3_5StandaloneMtpArtifactValidationError::MalformedIndex)?;
            if index.weight_map.is_empty() {
                return Err(Qwen3_5StandaloneMtpArtifactValidationError::MalformedIndex);
            }
            let mut stored_names_by_file_name = BTreeMap::<String, BTreeSet<String>>::new();
            for (stored_name, file_name) in index.weight_map {
                stored_names_by_file_name
                    .entry(file_name)
                    .or_default()
                    .insert(stored_name);
            }
            let source_declarations = stored_names_by_file_name
                .into_iter()
                .map(
                    |(file_name, indexed_stored_names)| StandaloneSourceDeclaration {
                        file_name,
                        indexed_stored_names: Some(indexed_stored_names),
                    },
                )
                .collect::<Vec<_>>();
            if has_single_file
                && (source_declarations.len() != 1
                    || source_declarations[0].file_name != "model.safetensors")
            {
                return Err(
                    Qwen3_5StandaloneMtpArtifactValidationError::ConflictingOrMissingLayout,
                );
            }
            Ok(source_declarations)
        }
        _ => Err(Qwen3_5StandaloneMtpArtifactValidationError::ConflictingOrMissingLayout),
    }
}

#[derive(Deserialize)]
struct StandaloneIndex {
    weight_map: BTreeMap<String, String>,
}

struct StandaloneSourceDeclaration {
    file_name: String,
    indexed_stored_names: Option<BTreeSet<String>>,
}

fn apply_standalone_storage_contract(
    profile_config: &mut Qwen3_5Config,
    canonical_tensor_names: &BTreeSet<String>,
    quantization_profile: Option<OptiQQuantizationProfile>,
) -> Result<(), Qwen3_5StandaloneMtpArtifactValidationError> {
    for canonical_weight_name in canonical_tensor_names
        .iter()
        .filter(|tensor_name| tensor_name.ends_with(".weight"))
    {
        let module_name = canonical_weight_name
            .strip_suffix(".weight")
            .expect("weight suffix was checked");
        let has_scales = canonical_tensor_names.contains(&format!("{module_name}.scales"));
        let has_biases = canonical_tensor_names.contains(&format!("{module_name}.biases"));
        let module_quantization_profile = if has_scales || has_biases {
            quantization_profile
                .ok_or(Qwen3_5StandaloneMtpArtifactValidationError::MissingQuantizationGeometry)?
        } else {
            OptiQQuantizationProfile::unquantized()
        };
        profile_config.set_mtp_module_quantization_profile(
            module_name.to_owned(),
            module_quantization_profile,
        );
    }
    Ok(())
}
