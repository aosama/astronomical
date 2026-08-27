use std::collections::{BTreeMap, HashSet};
use std::path::{Component, Path};

use thiserror::Error;

use crate::artifact_validation::{
    RequiredFileProfile, TensorDeclarationOrigin, TensorDtype, TensorFeature, TensorInventory,
    TensorLocation, TensorProfile, TensorSemanticRole, TensorSourceId, ValidatedSafetensorsSource,
    validate_required_file,
};
use crate::safetensors::SafetensorsTensorView;

const MAXIMUM_SIDECAR_RELATIVE_PATH_BYTES: usize = 4_096;
const MTP_STORED_PREFIX: &str = "mtp.";
const MTP_CANONICAL_PREFIX: &str = "language_model.mtp.";
// The one Qwen architecture sidecar receives a reserved opaque identity. Main-index sources are
// assigned upward from one, and the bounded index document cannot contain enough file names to
// reach this value, so the source cannot alias an indexed physical file.
const MTP_SIDECAR_SOURCE_ID: TensorSourceId = TensorSourceId::new(u32::MAX);

/// Structured validation failure for an optional Qwen MTP sidecar.
///
/// Replaces the previous unit error so `mtp_unavailable_reason` can surface a
/// human-readable cause (for example which tensor is missing or which dtype
/// mismatched) instead of a silent target-only fallback.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum Qwen3_5MtpSidecarValidationError {
    #[error("MTP sidecar file '{relative_path}' is unavailable or unparseable")]
    SidecarFileUnavailable { relative_path: String },

    #[error("sidecar tensor name '{tensor_name}' does not use the 'mtp.' stored prefix")]
    UnknownStoredTensor { tensor_name: String },

    #[error("sidecar declares duplicate canonical tensor '{canonical_name}'")]
    DuplicateCanonicalTensor { canonical_name: String },

    #[error("sidecar tensor '{canonical_name}' collides with a target tensor")]
    TargetTensorCollision { canonical_name: String },

    #[error("expected tensor {tensor_name} not found in sidecar")]
    MissingProfileTensor { tensor_name: String },

    #[error("sidecar tensor inventory conflict for '{canonical_name}'")]
    InventoryConflict { canonical_name: String },

    #[error("{detail}")]
    ProfileValidationFailed { tensor_name: String, detail: String },
}

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
    ) -> Result<Self, Qwen3_5MtpSidecarValidationError> {
        tracing::debug!(
            sidecar_path = declaration.relative_path(),
            "Qwen3_5MtpSidecarCandidate::open start"
        );
        let required_file = validate_required_file(
            model_directory,
            &RequiredFileProfile {
                file_name: declaration.relative_path().to_owned(),
                size_bytes: 0,
            },
        )
        .map_err(|_error| {
            tracing::debug!(
                sidecar_path = declaration.relative_path(),
                error_stage = "validate_required_file",
                "sidecar open failed at validate_required_file"
            );
            Qwen3_5MtpSidecarValidationError::SidecarFileUnavailable {
                relative_path: declaration.relative_path().to_owned(),
            }
        })?;
        tracing::debug!(
            sidecar_path = declaration.relative_path(),
            "validate_required_file succeeded, parsing safetensors"
        );
        let source = ValidatedSafetensorsSource::parse(MTP_SIDECAR_SOURCE_ID, required_file)
            .map_err(|_error| {
                tracing::debug!(
                    sidecar_path = declaration.relative_path(),
                    error_stage = "ValidatedSafetensorsSource::parse",
                    "sidecar open failed at parse"
                );
                Qwen3_5MtpSidecarValidationError::SidecarFileUnavailable {
                    relative_path: declaration.relative_path().to_owned(),
                }
            })?;
        tracing::debug!(
            sidecar_path = declaration.relative_path(),
            "safetensors parse succeeded"
        );
        let mut canonical_to_stored_name = BTreeMap::new();
        for stored_name in source.stored_tensor_names() {
            let suffix = stored_name.strip_prefix(MTP_STORED_PREFIX).ok_or_else(|| {
                Qwen3_5MtpSidecarValidationError::UnknownStoredTensor {
                    tensor_name: stored_name.clone(),
                }
            })?;
            let canonical_name = format!("{MTP_CANONICAL_PREFIX}{suffix}");
            if canonical_to_stored_name
                .insert(canonical_name.clone(), stored_name.clone())
                .is_some()
            {
                return Err(Qwen3_5MtpSidecarValidationError::DuplicateCanonicalTensor {
                    canonical_name,
                });
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
    ) -> Result<ValidatedQwen3_5MtpSidecar, Qwen3_5MtpSidecarValidationError> {
        let profile_by_canonical_name = canonical_profiles
            .iter()
            .map(|profile| (profile.name.as_str(), profile))
            .collect::<BTreeMap<_, _>>();
        // An optional sidecar may carry tensors beyond the generated profile set (for example a
        // quantized head that enumerates per-expert weights). Such additional tensors are accepted
        // as future extensibility; only the canonical names described by profiles must validate.
        let existing_names = existing_canonical_names.iter().collect::<HashSet<_>>();
        let mut inventory = TensorInventory::new();
        for (canonical_name, stored_name) in &self.canonical_to_stored_name {
            if existing_names.contains(canonical_name) {
                return Err(Qwen3_5MtpSidecarValidationError::TargetTensorCollision {
                    canonical_name: canonical_name.clone(),
                });
            }
            let Some(profile) = profile_by_canonical_name.get(canonical_name.as_str()) else {
                tracing::debug!(
                    canonical_name,
                    "optional MTP sidecar tensor is not described by any profile; ignoring"
                );
                continue;
            };
            let metadata = self
                .source
                .stored_tensor_view(stored_name)
                .expect("stored tensor metadata must exist after a successful sidecar parse");
            validate_tensor_profile(profile, metadata).map_err(|detail| {
                Qwen3_5MtpSidecarValidationError::ProfileValidationFailed {
                    tensor_name: canonical_name.clone(),
                    detail,
                }
            })?;
            inventory
                .insert(TensorLocation::new(
                    canonical_name.clone(),
                    stored_name.clone(),
                    self.source.source_id(),
                    TensorSemanticRole::MultiTokenPrediction,
                    TensorDeclarationOrigin::ArchitectureSidecar,
                    Some(TensorFeature::MultiTokenPrediction),
                ))
                .map_err(|_| Qwen3_5MtpSidecarValidationError::InventoryConflict {
                    canonical_name: canonical_name.clone(),
                })?;
        }
        // Every generated profile must have a matching stored tensor in the sidecar.
        for profile in canonical_profiles {
            if !self.canonical_to_stored_name.contains_key(&profile.name) {
                return Err(Qwen3_5MtpSidecarValidationError::MissingProfileTensor {
                    tensor_name: profile.name.clone(),
                });
            }
        }
        Ok(ValidatedQwen3_5MtpSidecar {
            source: self.source,
            inventory,
        })
    }
}

/// Validates one tensor profile against the sidecar's retained header metadata,
/// returning a human-readable detail string on mismatch.
fn validate_tensor_profile(
    profile: &TensorProfile,
    metadata: &SafetensorsTensorView,
) -> Result<(), String> {
    let expected_dtype = match profile.dtype {
        TensorDtype::AffineQuantizationFloat | TensorDtype::ModelFloat => "float (F16/BF16/F32)",
        TensorDtype::BFloat16 => "BF16",
        TensorDtype::Float32 => "F32",
        TensorDtype::UInt32 => "U32",
    };
    let dtype_matches = match profile.dtype {
        TensorDtype::AffineQuantizationFloat | TensorDtype::ModelFloat => {
            matches!(metadata.dtype.as_str(), "F16" | "BF16" | "F32")
        }
        TensorDtype::BFloat16 => metadata.dtype == "BF16",
        TensorDtype::Float32 => metadata.dtype == "F32",
        TensorDtype::UInt32 => metadata.dtype == "U32",
    };
    if !dtype_matches {
        return Err(format!(
            "dtype mismatch: expected {expected_dtype}, got {} for tensor {}",
            metadata.dtype, profile.name
        ));
    }
    if metadata.shape != profile.shape {
        return Err(format!(
            "shape mismatch: expected {:?}, got {:?} for tensor {}",
            profile.shape, metadata.shape, profile.name
        ));
    }
    Ok(())
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
) -> Result<Qwen3_5MtpSidecarValidationOutcome, Qwen3_5MtpSidecarValidationError> {
    let validated_sidecar = Qwen3_5MtpSidecarCandidate::open(model_directory, declaration)?
        .validate(canonical_profiles, existing_canonical_names)?;
    Ok(Qwen3_5MtpSidecarValidationOutcome {
        inventory: validated_sidecar.inventory,
        payload_bytes: validated_sidecar.source.payload_bytes(),
        is_available: true,
    })
}

/// Surface the structured validation outcome for hermetic tests that assert diagnostics.
#[doc(hidden)]
pub fn validate_qwen3_5_mtp_sidecar_result_for_tests(
    model_directory: &Path,
    declaration: &Qwen3_5MtpSidecarDeclaration,
    canonical_profiles: &[TensorProfile],
    existing_canonical_names: &[String],
) -> Result<Qwen3_5MtpSidecarValidationOutcome, Qwen3_5MtpSidecarValidationError> {
    validate_optional_sidecar(
        model_directory,
        declaration,
        canonical_profiles,
        existing_canonical_names,
    )
}
