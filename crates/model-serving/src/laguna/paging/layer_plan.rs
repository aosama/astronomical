use std::path::PathBuf;

use crate::expert_paging::{QuantizationMode, QuantizedExpertPageManifest, SafetensorsDtype};
use crate::laguna::artifacts::{
    LagunaCanonicalTensorDescriptor, LagunaExpertProjection, LagunaLayerTensorRole,
    LagunaTensorComponent, LagunaTensorContract, LagunaTensorId, LagunaTensorStorageEncoding,
    ValidatedLagunaArtifact,
};
use crate::laguna::normalization::{LagunaFeedForwardDescriptor, LagunaTargetContract};
use crate::memory::ExpertLayerGeometry;
use crate::{PerformanceAttribution, PerformanceOperation};

use super::error::LagunaPagingError;
use super::page_manifest::build_laguna_expert_page_manifest;
use super::source_slices::{
    compact_trailing_shape, expert_source_slices, map_dtype, parameter_name, projection_name,
};

/// One physical expert slice copied from a canonical source descriptor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LagunaExpertSourceSlice {
    pub(super) source_file: PathBuf,
    pub(super) source_file_offset: u64,
    pub(super) source_byte_count: usize,
}

/// One compact page tensor assembled from stacked, per-expert, or fused sources.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LagunaPagedTensorSource {
    pub(super) tensor_id: LagunaTensorId,
    pub(super) projection_name: &'static str,
    pub(super) parameter_name: &'static str,
    pub(super) dtype: SafetensorsDtype,
    pub(super) compact_trailing_shape: Vec<usize>,
    pub(super) bytes_per_expert: usize,
    pub(super) quantization_bits: i32,
    pub(super) quantization_group_size: i32,
    pub(super) expert_slices: Vec<LagunaExpertSourceSlice>,
}

/// One sparse decoder layer ready to emit complete-layer or routed pages.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LagunaSparseLayerPagingPlan {
    paging_slot_index: usize,
    decoder_layer_index: usize,
    expert_capacity: usize,
    experts_per_token: usize,
    quantization_mode: QuantizationMode,
    tensor_sources: Vec<LagunaPagedTensorSource>,
}

impl LagunaSparseLayerPagingPlan {
    /// Returns the dense 0-based slot consumed by the family-neutral residency planner.
    #[must_use]
    pub const fn paging_slot_index(&self) -> usize {
        self.paging_slot_index
    }

    /// Returns the original decoder-layer index from the target contract.
    #[must_use]
    pub const fn decoder_layer_index(&self) -> usize {
        self.decoder_layer_index
    }

    /// Returns the routed expert capacity for this sparse layer.
    #[must_use]
    pub const fn expert_capacity(&self) -> usize {
        self.expert_capacity
    }

    /// Returns the contract-owned top-K used to size a routed decode page.
    #[must_use]
    pub const fn experts_per_token(&self) -> usize {
        self.experts_per_token
    }

    /// Returns affine or native storage for this layer's routed experts.
    #[must_use]
    pub const fn quantization_mode(&self) -> QuantizationMode {
        self.quantization_mode
    }

    pub(super) fn tensor_sources(&self) -> &[LagunaPagedTensorSource] {
        &self.tensor_sources
    }

    /// Returns the contract affine profile for one compact projection name.
    #[cfg(feature = "direct-mlx")]
    pub(super) fn affine_profile_for_projection(&self, projection_name: &str) -> (i32, i32) {
        self.tensor_sources
            .iter()
            .find(|source| {
                source.projection_name == projection_name && source.parameter_name == "weight"
            })
            .map(|source| (source.quantization_bits, source.quantization_group_size))
            .unwrap_or((0, 0))
    }

    /// Returns exact packed bytes for one expert across every routed projection.
    pub fn expert_payload_byte_count(&self) -> Result<u64, LagunaPagingError> {
        self.tensor_sources.iter().try_fold(0_u64, |total, source| {
            let bytes_per_expert = u64::try_from(source.bytes_per_expert).map_err(|_| {
                LagunaPagingError::ExpertPayloadOverflow {
                    layer_index: self.decoder_layer_index,
                }
            })?;
            total
                .checked_add(bytes_per_expert)
                .ok_or(LagunaPagingError::ExpertPayloadOverflow {
                    layer_index: self.decoder_layer_index,
                })
        })
    }

    /// Returns exact packed bytes for every expert in this sparse layer.
    pub fn complete_layer_payload_byte_count(&self) -> Result<u64, LagunaPagingError> {
        let expert_payload_bytes = self.expert_payload_byte_count()?;
        let expert_capacity = u64::try_from(self.expert_capacity).map_err(|_| {
            LagunaPagingError::ExpertPayloadOverflow {
                layer_index: self.decoder_layer_index,
            }
        })?;
        let complete_payload_bytes = expert_payload_bytes.checked_mul(expert_capacity).ok_or(
            LagunaPagingError::ExpertPayloadOverflow {
                layer_index: self.decoder_layer_index,
            },
        )?;
        let source_payload_bytes =
            self.tensor_sources
                .iter()
                .try_fold(0_u64, |total, source| {
                    let slice_bytes =
                        source
                            .expert_slices
                            .iter()
                            .try_fold(0_u64, |inner, slice| {
                                let slice_count =
                                    u64::try_from(slice.source_byte_count).map_err(|_| {
                                        LagunaPagingError::ExpertPayloadOverflow {
                                            layer_index: self.decoder_layer_index,
                                        }
                                    })?;
                                inner.checked_add(slice_count).ok_or(
                                    LagunaPagingError::ExpertPayloadOverflow {
                                        layer_index: self.decoder_layer_index,
                                    },
                                )
                            })?;
                    total
                        .checked_add(slice_bytes)
                        .ok_or(LagunaPagingError::ExpertPayloadOverflow {
                            layer_index: self.decoder_layer_index,
                        })
                })?;
        if complete_payload_bytes != source_payload_bytes {
            return Err(LagunaPagingError::InconsistentCompletePayload {
                layer_index: self.decoder_layer_index,
                complete_payload_bytes,
                expert_payload_bytes,
                expert_capacity: self.expert_capacity,
            });
        }
        Ok(complete_payload_bytes)
    }

    /// Returns exact packed bytes for one routed top-K page from contract geometry.
    pub fn routed_page_payload_byte_count(&self) -> Result<u64, LagunaPagingError> {
        let expert_payload_bytes = self.expert_payload_byte_count()?;
        let experts_per_token = u64::try_from(self.experts_per_token).map_err(|_| {
            LagunaPagingError::ExpertPayloadOverflow {
                layer_index: self.decoder_layer_index,
            }
        })?;
        expert_payload_bytes.checked_mul(experts_per_token).ok_or(
            LagunaPagingError::ExpertPayloadOverflow {
                layer_index: self.decoder_layer_index,
            },
        )
    }

    /// Builds the complete-layer multi-token prefill page for every expert.
    pub fn complete_layer_page(&self) -> Result<QuantizedExpertPageManifest, LagunaPagingError> {
        let complete_expert_ids = (0..self.expert_capacity).collect::<Vec<_>>();
        build_laguna_expert_page_manifest(self, &complete_expert_ids)
    }

    /// Builds a routed one-token decode page for the supplied ascending expert IDs.
    pub fn routed_page(
        &self,
        expert_ids: &[usize],
    ) -> Result<QuantizedExpertPageManifest, LagunaPagingError> {
        build_laguna_expert_page_manifest(self, expert_ids)
    }

    /// Returns family-neutral geometry keyed by the dense paging slot.
    pub fn layer_geometry(&self) -> Result<ExpertLayerGeometry, LagunaPagingError> {
        Ok(ExpertLayerGeometry {
            layer_index: self.paging_slot_index,
            complete_layer_payload_bytes: self.complete_layer_payload_byte_count()?,
            expert_payload_bytes: self.expert_payload_byte_count()?,
            expert_capacity: self.expert_capacity,
            experts_per_token: self.experts_per_token,
        })
    }
}

/// Every sparse Laguna layer converted into pageable expert geometry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LagunaExpertPagingPlan {
    sparse_layers: Vec<LagunaSparseLayerPagingPlan>,
}

impl LagunaExpertPagingPlan {
    /// Builds pageable expert geometry from a validated artifact contract.
    pub fn from_validated_artifact(
        artifact: &ValidatedLagunaArtifact,
        model_directory: &std::path::Path,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Self, LagunaPagingError> {
        Self::from_contracts(
            artifact.target_contract(),
            artifact.tensor_contract(),
            model_directory,
            performance_attribution,
        )
    }

    /// Builds pageable expert geometry from already-normalized contracts.
    pub fn from_contracts(
        target_contract: &LagunaTargetContract,
        tensor_contract: &LagunaTensorContract,
        model_directory: &std::path::Path,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Self, LagunaPagingError> {
        performance_attribution
            .measure_operation(PerformanceOperation::ExpertPagerPlanConstruction, |_| {
                build_paging_plan(target_contract, tensor_contract, model_directory)
            })
    }

    /// Returns sparse layers in decoder order, remapped onto dense paging slots.
    #[must_use]
    pub fn sparse_layers(&self) -> &[LagunaSparseLayerPagingPlan] {
        &self.sparse_layers
    }

    /// Finds the remapped sparse plan for one decoder-layer index.
    #[must_use]
    pub fn sparse_layer_for_decoder(
        &self,
        decoder_layer_index: usize,
    ) -> Option<&LagunaSparseLayerPagingPlan> {
        self.sparse_layers
            .iter()
            .find(|sparse_layer| sparse_layer.decoder_layer_index() == decoder_layer_index)
    }

    /// Returns family-neutral geometry for the existing residency planner.
    pub fn layer_geometries(&self) -> Result<Vec<ExpertLayerGeometry>, LagunaPagingError> {
        self.sparse_layers
            .iter()
            .map(LagunaSparseLayerPagingPlan::layer_geometry)
            .collect()
    }
}

fn build_paging_plan(
    target_contract: &LagunaTargetContract,
    tensor_contract: &LagunaTensorContract,
    model_directory: &std::path::Path,
) -> Result<LagunaExpertPagingPlan, LagunaPagingError> {
    let mut sparse_layers = Vec::new();
    for layer in target_contract.layers() {
        let LagunaFeedForwardDescriptor::Moe(moe) = layer.feed_forward() else {
            continue;
        };
        let paging_slot_index = sparse_layers.len();
        let expert_capacity = usize::try_from(moe.expert_count()).map_err(|_| {
            LagunaPagingError::ExpertPayloadOverflow {
                layer_index: layer.layer_index(),
            }
        })?;
        let experts_per_token = usize::try_from(moe.experts_per_token()).map_err(|_| {
            LagunaPagingError::ExpertPayloadOverflow {
                layer_index: layer.layer_index(),
            }
        })?;
        let mut tensor_sources = Vec::new();
        let mut quantization_mode = QuantizationMode::NativeBfloat16;
        for projection in [
            LagunaExpertProjection::Gate,
            LagunaExpertProjection::Up,
            LagunaExpertProjection::Down,
        ] {
            let weight_id = routed_tensor_id(
                layer.layer_index(),
                projection,
                LagunaTensorComponent::Weight,
            );
            let weight_descriptor = tensor_contract.descriptor(&weight_id).ok_or(
                LagunaPagingError::MissingRoutedExpertTensor {
                    layer_index: layer.layer_index(),
                    tensor_id: weight_id,
                },
            )?;
            let (layer_mode, required_components) =
                required_components(weight_descriptor.storage_encoding(), weight_id)?;
            quantization_mode = layer_mode;
            for component in required_components {
                let tensor_id = routed_tensor_id(layer.layer_index(), projection, component);
                let descriptor = tensor_contract.descriptor(&tensor_id).ok_or(
                    LagunaPagingError::MissingRoutedExpertTensor {
                        layer_index: layer.layer_index(),
                        tensor_id,
                    },
                )?;
                tensor_sources.push(build_paged_tensor_source(
                    descriptor,
                    projection,
                    component,
                    expert_capacity,
                    model_directory,
                )?);
            }
        }
        let layer_plan = LagunaSparseLayerPagingPlan {
            paging_slot_index,
            decoder_layer_index: layer.layer_index(),
            expert_capacity,
            experts_per_token,
            quantization_mode,
            tensor_sources,
        };
        // Touch complete-payload accounting at bind so a later page cannot surprise.
        let _complete_payload_bytes = layer_plan.complete_layer_payload_byte_count()?;
        sparse_layers.push(layer_plan);
    }
    Ok(LagunaExpertPagingPlan { sparse_layers })
}

fn routed_tensor_id(
    layer_index: usize,
    projection: LagunaExpertProjection,
    component: LagunaTensorComponent,
) -> LagunaTensorId {
    LagunaTensorId::Layer {
        layer_index,
        role: LagunaLayerTensorRole::RoutedExpert(projection),
        component,
    }
}

fn required_components(
    storage_encoding: &LagunaTensorStorageEncoding,
    tensor_id: LagunaTensorId,
) -> Result<(QuantizationMode, Vec<LagunaTensorComponent>), LagunaPagingError> {
    match storage_encoding {
        LagunaTensorStorageEncoding::Unquantized => Ok((
            QuantizationMode::NativeBfloat16,
            vec![LagunaTensorComponent::Weight],
        )),
        LagunaTensorStorageEncoding::DirectAffine { .. } => Ok((
            QuantizationMode::Affine,
            vec![
                LagunaTensorComponent::Weight,
                LagunaTensorComponent::Scales,
                LagunaTensorComponent::Biases,
            ],
        )),
        LagunaTensorStorageEncoding::SymmetricPackedAffine { .. }
        | LagunaTensorStorageEncoding::NativeNvfp4 { .. }
        | LagunaTensorStorageEncoding::TwoLevelCompressedNvfp4 { .. }
        | LagunaTensorStorageEncoding::BlockFp8 { .. } => {
            Err(LagunaPagingError::UnsupportedRoutedStorage { tensor_id })
        }
    }
}

fn build_paged_tensor_source(
    descriptor: &LagunaCanonicalTensorDescriptor,
    projection: LagunaExpertProjection,
    component: LagunaTensorComponent,
    expert_capacity: usize,
    model_directory: &std::path::Path,
) -> Result<LagunaPagedTensorSource, LagunaPagingError> {
    let first_source =
        descriptor
            .sources()
            .first()
            .ok_or(LagunaPagingError::MissingSourceInterval {
                tensor_id: descriptor.tensor_id(),
            })?;
    let dtype = map_dtype(first_source.raw_dtype(), descriptor.tensor_id())?;
    let compact_trailing_shape = compact_trailing_shape(descriptor, expert_capacity)?;
    let mut expert_slices = expert_source_slices(descriptor, expert_capacity)?;
    // Bounded reads open these paths later; keep only artifact-local shard names
    // until this join so the pager never invents a developer home directory.
    for expert_slice in &mut expert_slices {
        expert_slice.source_file = model_directory.join(&expert_slice.source_file);
    }
    let bytes_per_expert = expert_slices
        .first()
        .map(|slice| slice.source_byte_count)
        .ok_or(LagunaPagingError::MissingSourceInterval {
            tensor_id: descriptor.tensor_id(),
        })?;
    if expert_slices.len() != expert_capacity
        || expert_slices
            .iter()
            .any(|slice| slice.source_byte_count != bytes_per_expert)
    {
        return Err(LagunaPagingError::ExpertPayloadNotDivisible {
            tensor_id: descriptor.tensor_id(),
        });
    }
    let (quantization_bits, quantization_group_size) =
        quantization_profile(descriptor.storage_encoding());
    Ok(LagunaPagedTensorSource {
        tensor_id: descriptor.tensor_id(),
        projection_name: projection_name(projection),
        parameter_name: parameter_name(component),
        dtype,
        compact_trailing_shape,
        bytes_per_expert,
        quantization_bits,
        quantization_group_size,
        expert_slices,
    })
}

fn quantization_profile(storage_encoding: &LagunaTensorStorageEncoding) -> (i32, i32) {
    match storage_encoding {
        LagunaTensorStorageEncoding::DirectAffine { profile } => {
            (profile.bits() as i32, profile.group_size() as i32)
        }
        LagunaTensorStorageEncoding::Unquantized
        | LagunaTensorStorageEncoding::SymmetricPackedAffine { .. }
        | LagunaTensorStorageEncoding::NativeNvfp4 { .. }
        | LagunaTensorStorageEncoding::TwoLevelCompressedNvfp4 { .. }
        | LagunaTensorStorageEncoding::BlockFp8 { .. } => (0, 0),
    }
}
