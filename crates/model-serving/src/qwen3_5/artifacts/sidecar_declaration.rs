use std::collections::{BTreeMap, HashSet};
use std::path::{Component, Path};

use thiserror::Error;

use crate::artifact_validation::{
    RequiredFileProfile, TensorDeclarationOrigin, TensorFeature, TensorInventory, TensorLocation,
    TensorProfile, TensorSemanticRole, TensorSourceId, ValidatedSafetensorsSource,
    validate_required_file,
};

const MAXIMUM_SIDECAR_RELATIVE_PATH_BYTES: usize = 4_096;
const MTP_STORED_PREFIX: &str = "mtp.";
const MTP_CANONICAL_PREFIX: &str = "language_model.mtp.";
// The one Qwen architecture sidecar receives a reserved opaque identity. Main-index sources are
// assigned upward from one, and the bounded index document cannot contain enough file names to
// reach this value, so the source cannot alias an indexed physical file.
const MTP_SIDECAR_SOURCE_ID: TensorSourceId = TensorSourceId::new(u32::MAX);

/// Validated Qwen-specific declaration from `mlx_lm_extra_tensors.mtp_file`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3_5MtpSidecarDeclaration {
    relative_path: String,
}

impl Qwen3_5MtpSidecarDeclaration {
    /// Validates declaration syntax before any filesystem operation.
    pub fn parse(relative_path: &str) -> Result<Self, Qwen3_5MtpSidecarDeclarationError> {
        let path = Path::new(relative_path);
        let has_only_normal_components = path
            .components()
            .all(|component| matches!(component, Component::Normal(_)));
        let has_safetensors_suffix =
            path.extension().and_then(|extension| extension.to_str()) == Some("safetensors");
        let has_ambiguous_text_component = relative_path
            .split('/')
            .any(|component| component.is_empty() || matches!(component, "." | ".."));
        if relative_path.is_empty()
            || relative_path.len() > MAXIMUM_SIDECAR_RELATIVE_PATH_BYTES
            || path.is_absolute()
            || !has_only_normal_components
            || !has_safetensors_suffix
            || has_ambiguous_text_component
            || relative_path.contains('\\')
            || relative_path
                .split('/')
                .any(|component| component.ends_with(':'))
        {
            return Err(Qwen3_5MtpSidecarDeclarationError::UnsafeRelativePath);
        }
        Ok(Self {
            relative_path: relative_path.to_owned(),
        })
    }

    #[must_use]
    pub fn relative_path(&self) -> &str {
        &self.relative_path
    }
}

/// Bounded declaration failure that never includes a local path.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum Qwen3_5MtpSidecarDeclarationError {
    #[error("MTP sidecar declaration must be a bounded relative SafeTensors path")]
    UnsafeRelativePath,
}

/// Hermetic optional-sidecar validation outcome.
#[derive(Debug, Default)]
pub struct Qwen3_5MtpSidecarValidationOutcome {
    inventory: TensorInventory,
    payload_bytes: u64,
    is_available: bool,
}

/// Already-open optional Qwen MTP source and its bounded header name mapping.
pub(crate) struct Qwen3_5MtpSidecarCandidate {
    source: ValidatedSafetensorsSource,
    canonical_to_stored_name: BTreeMap<String, String>,
}

impl Qwen3_5MtpSidecarCandidate {
    pub(crate) fn open(
        model_directory: &Path,
        declaration: &Qwen3_5MtpSidecarDeclaration,
    ) -> Result<Self, ()> {
        let required_file = validate_required_file(
            model_directory,
            &RequiredFileProfile {
                file_name: declaration.relative_path().to_owned(),
                size_bytes: 0,
            },
        )
        .map_err(|_| ())?;
        let source = ValidatedSafetensorsSource::parse(MTP_SIDECAR_SOURCE_ID, required_file)
            .map_err(|_| ())?;
        let mut canonical_to_stored_name = BTreeMap::new();
        for stored_name in source.stored_tensor_names() {
            let suffix = stored_name.strip_prefix(MTP_STORED_PREFIX).ok_or(())?;
            let canonical_name = format!("{MTP_CANONICAL_PREFIX}{suffix}");
            if canonical_to_stored_name
                .insert(canonical_name, stored_name.clone())
                .is_some()
            {
                return Err(());
            }
        }
        Ok(Self {
            source,
            canonical_to_stored_name,
        })
    }

    pub(crate) fn canonical_names(&self) -> impl Iterator<Item = &String> {
        self.canonical_to_stored_name.keys()
    }

    pub(crate) fn validate(
        self,
        canonical_profiles: &[TensorProfile],
        existing_canonical_names: &[String],
    ) -> Result<ValidatedQwen3_5MtpSidecar, ()> {
        let profile_by_canonical_name = canonical_profiles
            .iter()
            .map(|profile| (profile.name.as_str(), profile))
            .collect::<BTreeMap<_, _>>();
        if self.canonical_to_stored_name.len() != canonical_profiles.len()
            || self.canonical_to_stored_name.keys().any(|canonical_name| {
                !profile_by_canonical_name.contains_key(canonical_name.as_str())
            })
        {
            return Err(());
        }
        let existing_names = existing_canonical_names.iter().collect::<HashSet<_>>();
        let mut inventory = TensorInventory::new();
        for (canonical_name, stored_name) in &self.canonical_to_stored_name {
            if existing_names.contains(canonical_name) {
                return Err(());
            }
            inventory
                .insert(TensorLocation::new(
                    canonical_name.clone(),
                    stored_name.clone(),
                    self.source.source_id(),
                    TensorSemanticRole::MultiTokenPrediction,
                    TensorDeclarationOrigin::ArchitectureSidecar,
                    Some(TensorFeature::MultiTokenPrediction),
                ))
                .map_err(|_| ())?;
        }
        self.source
            .validate_inventory_profiles(&inventory, canonical_profiles)
            .map_err(|_| ())?;
        Ok(ValidatedQwen3_5MtpSidecar {
            source: self.source,
            inventory,
        })
    }
}

/// Validated optional sidecar ownership transferred into the complete artifact.
pub(crate) struct ValidatedQwen3_5MtpSidecar {
    pub(crate) source: ValidatedSafetensorsSource,
    pub(crate) inventory: TensorInventory,
}

impl Qwen3_5MtpSidecarValidationOutcome {
    #[must_use]
    pub const fn is_available(&self) -> bool {
        self.is_available
    }

    #[must_use]
    pub fn source_count(&self) -> usize {
        self.inventory.source_ids().count()
    }

    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.inventory.tensor_count()
    }

    #[must_use]
    pub const fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }

    #[must_use]
    pub fn stored_name(&self, canonical_name: &str) -> Option<&str> {
        self.inventory
            .location(canonical_name)
            .map(TensorLocation::stored_name)
    }
}

/// Validates one optional Qwen MTP sidecar without allowing it to reject target serving.
#[doc(hidden)]
pub fn validate_qwen3_5_mtp_sidecar_for_tests(
    model_directory: &Path,
    declaration: &Qwen3_5MtpSidecarDeclaration,
    canonical_profiles: &[TensorProfile],
    existing_canonical_names: &[String],
) -> Qwen3_5MtpSidecarValidationOutcome {
    validate_optional_sidecar(
        model_directory,
        declaration,
        canonical_profiles,
        existing_canonical_names,
    )
    .unwrap_or_default()
}

fn validate_optional_sidecar(
    model_directory: &Path,
    declaration: &Qwen3_5MtpSidecarDeclaration,
    canonical_profiles: &[TensorProfile],
    existing_canonical_names: &[String],
) -> Result<Qwen3_5MtpSidecarValidationOutcome, ()> {
    let validated_sidecar = Qwen3_5MtpSidecarCandidate::open(model_directory, declaration)?
        .validate(canonical_profiles, existing_canonical_names)?;
    Ok(Qwen3_5MtpSidecarValidationOutcome {
        inventory: validated_sidecar.inventory,
        payload_bytes: validated_sidecar.source.payload_bytes(),
        is_available: true,
    })
}
