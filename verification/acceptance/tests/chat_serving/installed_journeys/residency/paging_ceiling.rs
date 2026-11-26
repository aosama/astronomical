//! The memory ceiling that fits a sparse artifact's expert weights but not its
//! activation headroom, derived from the artifact's own geometry so the
//! journeys that force streaming never hard-code a machine-specific size.

use std::path::Path;

use model_serving::{
    CompleteResidencyHeadroomBoundary, ModelMemoryGeometry, Qwen3_5ArtifactValidator,
    model_memory_geometry_from_validated_artifact,
};

pub(super) struct DerivedPagingCeiling {
    pub(super) model_geometry: ModelMemoryGeometry,
    pub(super) required_headroom_bytes: u64,
    pub(super) paging_ceiling_bytes: u64,
}

impl DerivedPagingCeiling {
    pub(super) fn paging_ceiling_limit_bytes(&self) -> usize {
        usize::try_from(self.paging_ceiling_bytes).expect("the paging ceiling should fit usize")
    }
}

pub(super) fn derive_paging_ceiling(model_directory: &Path) -> DerivedPagingCeiling {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, 20_480)
        .expect("the sparse artifact should validate");
    let (model_geometry, required_headroom_bytes) =
        model_memory_geometry_from_validated_artifact(&validated_artifact, model_directory)
            .expect("the sparse artifact should expose disk RAM geometry");
    let paging_ceiling_bytes = CompleteResidencyHeadroomBoundary::from_model_geometry(
        model_geometry,
        required_headroom_bytes,
    )
    .paging_ceiling_bytes()
    .expect(
        "this sparse artifact must have a ceiling that fits expert weights but not activation headroom",
    );
    DerivedPagingCeiling {
        model_geometry,
        required_headroom_bytes,
        paging_ceiling_bytes,
    }
}
