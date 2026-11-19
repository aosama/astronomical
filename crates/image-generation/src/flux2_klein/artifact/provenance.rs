//! Which published builds an artifact may claim to be, and their license.

use super::super::mlx_4bit_build::{
    FLUX2_KLEIN_MLX_4BIT_MODEL_ID, FLUX2_KLEIN_MLX_4BIT_PROVIDER_MODEL_ID,
    FLUX2_KLEIN_MLX_4BIT_REVISION, identifies_mlx_4bit_build,
};
use super::super::weight_layout::Flux2KleinWeightLayout;
use super::Flux2KleinArtifactError;

pub const FLUX2_KLEIN_OFFICIAL_MODEL_ID: &str = "FLUX.2-klein-4B";
pub const FLUX2_KLEIN_PROVIDER_MODEL_ID: &str = "black-forest-labs/FLUX.2-klein-4B";
pub const FLUX2_KLEIN_OFFICIAL_REVISION: &str = "e7b7dc27f91deacad38e78976d1f2b499d76a294";
/// Both builds carry the upstream Apache-2.0 license.
pub(super) const OFFICIAL_LICENSE_IDENTIFIER: &str = "Apache-2.0";

/// Discovery provenance required before local bytes may claim a reviewed build.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Flux2KleinArtifactProvenance {
    model_id: String,
    revision: String,
    license_identifier: String,
}

impl Flux2KleinArtifactProvenance {
    pub fn new(
        model_id: impl Into<String>,
        revision: impl Into<String>,
        license_identifier: impl Into<String>,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            revision: revision.into(),
            license_identifier: license_identifier.into(),
        }
    }

    /// The reviewed BF16 Diffusers build.
    pub fn official() -> Self {
        Self::new(
            FLUX2_KLEIN_PROVIDER_MODEL_ID,
            FLUX2_KLEIN_OFFICIAL_REVISION,
            OFFICIAL_LICENSE_IDENTIFIER,
        )
    }

    /// The reviewed 4-bit MLX build (issue #1904).
    pub fn mlx_4bit() -> Self {
        Self::new(
            FLUX2_KLEIN_MLX_4BIT_PROVIDER_MODEL_ID,
            FLUX2_KLEIN_MLX_4BIT_REVISION,
            OFFICIAL_LICENSE_IDENTIFIER,
        )
    }

    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    #[must_use]
    pub fn revision(&self) -> &str {
        &self.revision
    }

    /// The model id the build serves under, as discovery names it, so the
    /// engine and the worker's selected configuration agree on one identity.
    #[must_use]
    pub fn serving_model_id(&self) -> &'static str {
        if identifies_mlx_4bit_build(&self.model_id) {
            FLUX2_KLEIN_MLX_4BIT_MODEL_ID
        } else {
            FLUX2_KLEIN_OFFICIAL_MODEL_ID
        }
    }

    pub(super) fn into_revision(self) -> String {
        self.revision
    }
}

/// License metadata exposed without treating a directory or model name as authority.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Flux2KleinLicense;

impl Flux2KleinLicense {
    pub const fn identifier(&self) -> &'static str {
        OFFICIAL_LICENSE_IDENTIFIER
    }
    pub const fn display_name(&self) -> &'static str {
        "Apache License 2.0"
    }
}

/// Architecture files and tensors prove Klein 4B. Provenance only has to name
/// the build whose layout the files have, record an immutable revision, and
/// keep Apache-2.0. One Hub SHA is not a gate.
pub(super) fn validate_provenance(
    provenance: &Flux2KleinArtifactProvenance,
    weight_layout: Flux2KleinWeightLayout,
) -> Result<(), Flux2KleinArtifactError> {
    if provenance_supports_layout(provenance, weight_layout) {
        return Ok(());
    }
    Err(Flux2KleinArtifactError::UnsupportedProvenance {
        model_id: provenance.model_id.clone(),
        revision: provenance.revision.clone(),
        license_identifier: provenance.license_identifier.clone(),
    })
}

fn provenance_supports_layout(
    provenance: &Flux2KleinArtifactProvenance,
    weight_layout: Flux2KleinWeightLayout,
) -> bool {
    provenance.license_identifier == OFFICIAL_LICENSE_IDENTIFIER
        && names_build_with_layout(&provenance.model_id, weight_layout)
        && records_immutable_revision(&provenance.revision)
}

fn names_build_with_layout(model_id: &str, weight_layout: Flux2KleinWeightLayout) -> bool {
    match weight_layout {
        Flux2KleinWeightLayout::DiffusersBf16 => identifies_bf16_build(model_id),
        Flux2KleinWeightLayout::MlxAffine4Bit => identifies_mlx_4bit_build(model_id),
    }
}

fn identifies_bf16_build(model_id: &str) -> bool {
    model_id == FLUX2_KLEIN_OFFICIAL_MODEL_ID
        || model_id == FLUX2_KLEIN_PROVIDER_MODEL_ID
        || model_id.rsplit('/').next() == Some(FLUX2_KLEIN_OFFICIAL_MODEL_ID)
}

fn records_immutable_revision(revision: &str) -> bool {
    revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::{Flux2KleinArtifactProvenance, validate_provenance};
    use crate::flux2_klein::weight_layout::Flux2KleinWeightLayout;

    #[test]
    fn should_accept_each_build_under_its_own_layout() {
        assert!(
            validate_provenance(
                &Flux2KleinArtifactProvenance::official(),
                Flux2KleinWeightLayout::DiffusersBf16
            )
            .is_ok()
        );
        assert!(
            validate_provenance(
                &Flux2KleinArtifactProvenance::mlx_4bit(),
                Flux2KleinWeightLayout::MlxAffine4Bit
            )
            .is_ok()
        );
    }

    #[test]
    fn should_refuse_a_build_whose_files_have_the_other_layout() {
        assert!(
            validate_provenance(
                &Flux2KleinArtifactProvenance::official(),
                Flux2KleinWeightLayout::MlxAffine4Bit
            )
            .is_err()
        );
        assert!(
            validate_provenance(
                &Flux2KleinArtifactProvenance::mlx_4bit(),
                Flux2KleinWeightLayout::DiffusersBf16
            )
            .is_err()
        );
    }

    #[test]
    fn should_serve_each_build_under_its_discovered_model_id() {
        assert_eq!(
            Flux2KleinArtifactProvenance::official().serving_model_id(),
            "FLUX.2-klein-4B"
        );
        assert_eq!(
            Flux2KleinArtifactProvenance::mlx_4bit().serving_model_id(),
            "flux2-klein-4b-4bit"
        );
    }

    #[test]
    fn should_refuse_a_mutable_revision_or_another_license() {
        let branch_revision =
            Flux2KleinArtifactProvenance::new("flux2-klein-4b-4bit", "main", "Apache-2.0");
        let other_license = Flux2KleinArtifactProvenance::new(
            "flux2-klein-4b-4bit",
            "860e87183ceb29e39627c0612ebd66d8ea66e68c",
            "MIT",
        );
        for provenance in [branch_revision, other_license] {
            assert!(
                validate_provenance(&provenance, Flux2KleinWeightLayout::MlxAffine4Bit).is_err()
            );
        }
    }
}
