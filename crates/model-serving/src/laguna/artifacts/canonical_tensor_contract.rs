use std::collections::BTreeMap;

use ::safetensors::Dtype;

use super::artifact_error::LagunaArtifactValidationError;
use super::direct_storage_validation::{
    required_component_storage_dtype, validate_source_dtypes, validate_source_shapes,
};
use super::exact_storage_binding::{bind_exact_storage, is_exact_storage_source};
use super::expected_tensors::{LagunaExpectedTensor, expected_tensors};
use super::kv_cache_metadata::collect_fp8_kv_cache_metadata;
use super::tensor_assembly::LagunaTensorAssembly;
use super::tensor_id::{
    LagunaExpertProjection, LagunaLayerTensorRole, LagunaTensorComponent, LagunaTensorId,
};
use super::tensor_name_contract::LagunaTensorNameContract;
use super::tensor_storage::LagunaTensorStorageEncoding;
use crate::laguna::{LagunaExecutionDtype, LagunaFeedForwardDescriptor, LagunaTargetContract};

/// One physical source copied from the family-neutral retained-descriptor inventory.
pub(super) struct LocatedRawTensorDescriptor {
    pub(super) shard_file_name: String,
    pub(super) raw_tensor_name: String,
    pub(super) dtype: Dtype,
    pub(super) shape: Vec<usize>,
    pub(super) data_start_offset_bytes: u64,
    pub(super) data_end_offset_bytes: u64,
    pub(super) payload_bytes: u64,
}

/// Name-free operation needed to construct one canonical tensor from its sources.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum LagunaCanonicalTensorAssemblyKind {
    DirectAlias,
    StackedSource,
    PerExpertStack,
    FusedGateUpSource {
        projection: LagunaExpertProjection,
    },
    FusedPerExpertGateUp {
        projection: LagunaExpertProjection,
    },
    ReinterpretPackedBits {
        source_layout: LagunaCanonicalSourceLayout,
    },
    DeriveSymmetricBias {
        source_layout: LagunaCanonicalSourceLayout,
        negative_code_offset: u32,
    },
    NativeNvfp4 {
        source_layout: LagunaCanonicalSourceLayout,
    },
    TwoLevelCompressedNvfp4 {
        source_layout: LagunaCanonicalSourceLayout,
    },
    BlockFp8 {
        source_layout: LagunaCanonicalSourceLayout,
    },
}

/// Packaging retained independently from an exact storage transform.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum LagunaCanonicalSourceLayout {
    Direct,
    Stacked,
    PerExpert,
    FusedStacked { projection: LagunaExpertProjection },
    FusedPerExpert { projection: LagunaExpertProjection },
}

/// Semantic role of one retained source interval in an exact assembly.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum LagunaTensorSourceRole {
    WeightValues,
    WeightCodes,
    PackedWeightCodes,
    GroupScales,
    BlockScales,
    AffineBiases,
    WeightGlobalScale,
    InputGlobalScale,
    LogicalShape,
    AttentionKeyScaleMetadata,
    AttentionValueScaleMetadata,
}

/// Exact retained-file provenance for one canonical tensor source.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LagunaTensorSourceDescriptor {
    pub(super) shard_file_name: String,
    pub(super) raw_tensor_name: String,
    pub(super) data_start_offset_bytes: u64,
    pub(super) data_end_offset_bytes: u64,
    pub(super) raw_shape: Vec<usize>,
    pub(super) raw_dtype: Dtype,
    pub(super) payload_bytes: u64,
    pub(super) role: LagunaTensorSourceRole,
}

impl LagunaTensorSourceDescriptor {
    #[must_use]
    pub fn shard_file_name(&self) -> &str {
        &self.shard_file_name
    }

    /// Returns the raw artifact name only as physical source provenance.
    #[must_use]
    pub fn raw_tensor_name(&self) -> &str {
        &self.raw_tensor_name
    }

    #[must_use]
    pub const fn data_start_offset_bytes(&self) -> u64 {
        self.data_start_offset_bytes
    }

    #[must_use]
    pub const fn data_end_offset_bytes(&self) -> u64 {
        self.data_end_offset_bytes
    }

    #[must_use]
    pub fn raw_shape(&self) -> &[usize] {
        &self.raw_shape
    }

    #[must_use]
    pub const fn raw_dtype(&self) -> Dtype {
        self.raw_dtype
    }

    #[must_use]
    pub const fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }

    #[must_use]
    pub const fn role(&self) -> LagunaTensorSourceRole {
        self.role
    }
}

/// Explicitly retained artifact metadata that cannot become an executable model weight.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LagunaNonExecutableMetadataDescriptor {
    tensor_id: LagunaTensorId,
    sources: Vec<LagunaTensorSourceDescriptor>,
}

impl LagunaNonExecutableMetadataDescriptor {
    pub(super) fn new(
        tensor_id: LagunaTensorId,
        sources: Vec<LagunaTensorSourceDescriptor>,
    ) -> Self {
        Self { tensor_id, sources }
    }

    #[must_use]
    pub const fn tensor_id(&self) -> LagunaTensorId {
        self.tensor_id
    }

    #[must_use]
    pub fn sources(&self) -> &[LagunaTensorSourceDescriptor] {
        &self.sources
    }
}

/// Fully validated logical tensor plus exact physical source records.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LagunaCanonicalTensorDescriptor {
    tensor_id: LagunaTensorId,
    canonical_module_name: Option<String>,
    logical_shape: Vec<usize>,
    execution_dtype: Dtype,
    storage_dtype: Dtype,
    storage_encoding: LagunaTensorStorageEncoding,
    assembly_kind: LagunaCanonicalTensorAssemblyKind,
    sources: Vec<LagunaTensorSourceDescriptor>,
}

impl LagunaCanonicalTensorDescriptor {
    #[must_use]
    pub const fn tensor_id(&self) -> LagunaTensorId {
        self.tensor_id
    }

    /// Returns the wrapper-free executable module used for profile lookup.
    #[must_use]
    pub fn canonical_module_name(&self) -> Option<&str> {
        self.canonical_module_name.as_deref()
    }

    #[must_use]
    pub fn logical_shape(&self) -> &[usize] {
        &self.logical_shape
    }

    #[must_use]
    pub const fn execution_dtype(&self) -> Dtype {
        self.execution_dtype
    }

    #[must_use]
    pub const fn storage_dtype(&self) -> Dtype {
        self.storage_dtype
    }

    /// Returns native or direct-affine storage plus the exact affine profile.
    #[must_use]
    pub const fn storage_encoding(&self) -> &LagunaTensorStorageEncoding {
        &self.storage_encoding
    }

    #[must_use]
    pub const fn assembly_kind(&self) -> LagunaCanonicalTensorAssemblyKind {
        self.assembly_kind
    }

    #[must_use]
    pub fn sources(&self) -> &[LagunaTensorSourceDescriptor] {
        &self.sources
    }
}

/// Deterministic structured tensor contract consumed by future Laguna construction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LagunaTensorContract {
    descriptors: BTreeMap<LagunaTensorId, LagunaCanonicalTensorDescriptor>,
    non_executable_metadata: Vec<LagunaNonExecutableMetadataDescriptor>,
}

impl LagunaTensorContract {
    #[must_use]
    pub const fn descriptors(&self) -> &BTreeMap<LagunaTensorId, LagunaCanonicalTensorDescriptor> {
        &self.descriptors
    }

    #[must_use]
    pub fn descriptor(
        &self,
        tensor_id: &LagunaTensorId,
    ) -> Option<&LagunaCanonicalTensorDescriptor> {
        self.descriptors.get(tensor_id)
    }

    #[must_use]
    pub fn non_executable_metadata(&self) -> &[LagunaNonExecutableMetadataDescriptor] {
        &self.non_executable_metadata
    }
}

pub(super) fn build_canonical_tensor_contract(
    target_contract: &LagunaTargetContract,
    tensor_name_contract: &LagunaTensorNameContract,
    located_tensors: &BTreeMap<String, LocatedRawTensorDescriptor>,
) -> Result<LagunaTensorContract, LagunaArtifactValidationError> {
    let mut expected_tensors = expected_tensors(target_contract)?;
    for tensor_id in tensor_name_contract.assemblies().keys().copied() {
        insert_optional_router_correction_bias(target_contract, tensor_id, &mut expected_tensors)?;
    }
    for tensor_id in tensor_name_contract.assemblies().keys() {
        if tensor_component(*tensor_id) == LagunaTensorComponent::ZeroPoint {
            return Err(
                LagunaArtifactValidationError::UnsupportedAsymmetricStorage {
                    tensor_id: *tensor_id,
                },
            );
        }
        if !expected_tensors.contains_key(tensor_id)
            && !is_exact_storage_source(target_contract.storage(), *tensor_id, &expected_tensors)
        {
            return Err(LagunaArtifactValidationError::UnexpectedCanonicalTensor {
                tensor_id: *tensor_id,
            });
        }
    }

    let execution_dtype = execution_dtype(target_contract.model().execution_dtype());
    let mut descriptors = BTreeMap::new();
    for (tensor_id, expected_tensor) in expected_tensors {
        if let Some(exact_binding) = bind_exact_storage(
            tensor_id,
            &expected_tensor,
            tensor_name_contract,
            located_tensors,
        )? {
            descriptors.insert(
                tensor_id,
                LagunaCanonicalTensorDescriptor {
                    tensor_id,
                    canonical_module_name: expected_tensor.canonical_module_name,
                    logical_shape: expected_tensor.logical_shape,
                    execution_dtype,
                    storage_dtype: exact_binding.storage_dtype,
                    storage_encoding: exact_binding.storage_encoding,
                    assembly_kind: exact_binding.assembly_kind,
                    sources: exact_binding.sources,
                },
            );
            continue;
        }
        let assembly = tensor_name_contract
            .assemblies()
            .get(&tensor_id)
            .ok_or(LagunaArtifactValidationError::ExpectedTensorMissing { tensor_id })?;
        let sources = resolve_sources(
            tensor_id,
            assembly,
            located_tensors,
            ordinary_source_role(
                tensor_component(tensor_id),
                &expected_tensor.storage_encoding,
            ),
        )?;
        let component = tensor_component(tensor_id);
        validate_source_shapes(
            tensor_id,
            &expected_tensor.logical_shape,
            component,
            &expected_tensor.storage_encoding,
            assembly,
            &sources,
        )?;
        let required_storage_dtype =
            required_component_storage_dtype(component, &expected_tensor.storage_encoding);
        let physical_storage_dtype =
            validate_source_dtypes(tensor_id, required_storage_dtype, &sources)?;
        descriptors.insert(
            tensor_id,
            LagunaCanonicalTensorDescriptor {
                tensor_id,
                canonical_module_name: expected_tensor.canonical_module_name,
                logical_shape: expected_tensor.logical_shape,
                execution_dtype,
                storage_dtype: physical_storage_dtype,
                storage_encoding: expected_tensor.storage_encoding,
                assembly_kind: assembly_kind(assembly),
                sources,
            },
        );
    }
    let non_executable_metadata = collect_fp8_kv_cache_metadata(
        target_contract.storage(),
        target_contract,
        tensor_name_contract,
        located_tensors,
    )?;
    Ok(LagunaTensorContract {
        descriptors,
        non_executable_metadata,
    })
}

fn insert_optional_router_correction_bias(
    target_contract: &LagunaTargetContract,
    tensor_id: LagunaTensorId,
    expected_tensors: &mut BTreeMap<LagunaTensorId, LagunaExpectedTensor>,
) -> Result<(), LagunaArtifactValidationError> {
    let LagunaTensorId::Layer {
        layer_index,
        role: LagunaLayerTensorRole::RouterCorrectionBias,
        component: LagunaTensorComponent::Weight,
    } = tensor_id
    else {
        return Ok(());
    };
    let layer = target_contract
        .layers()
        .get(layer_index)
        .ok_or(LagunaArtifactValidationError::TensorGeometryOverflow)?;
    let LagunaFeedForwardDescriptor::Moe(moe_descriptor) = layer.feed_forward() else {
        return Err(LagunaArtifactValidationError::UnexpectedCanonicalTensor { tensor_id });
    };
    let expert_count = usize::try_from(moe_descriptor.expert_count())
        .map_err(|_| LagunaArtifactValidationError::TensorGeometryOverflow)?;
    expected_tensors.insert(
        tensor_id,
        LagunaExpectedTensor {
            logical_shape: vec![expert_count],
            canonical_module_name: None,
            storage_encoding: LagunaTensorStorageEncoding::Unquantized,
        },
    );
    Ok(())
}

pub(super) fn resolve_sources(
    tensor_id: LagunaTensorId,
    assembly: &LagunaTensorAssembly,
    located_tensors: &BTreeMap<String, LocatedRawTensorDescriptor>,
    role: LagunaTensorSourceRole,
) -> Result<Vec<LagunaTensorSourceDescriptor>, LagunaArtifactValidationError> {
    let mut sources = Vec::with_capacity(assembly.sources().len());
    for source in assembly.sources() {
        let located_tensor = located_tensors.get(source.raw_name()).ok_or_else(|| {
            LagunaArtifactValidationError::CanonicalSourceMissing {
                tensor_id,
                tensor_name: source.raw_name().to_owned(),
            }
        })?;
        sources.push(LagunaTensorSourceDescriptor {
            shard_file_name: located_tensor.shard_file_name.clone(),
            raw_tensor_name: located_tensor.raw_tensor_name.clone(),
            data_start_offset_bytes: located_tensor.data_start_offset_bytes,
            data_end_offset_bytes: located_tensor.data_end_offset_bytes,
            raw_shape: located_tensor.shape.clone(),
            raw_dtype: located_tensor.dtype,
            payload_bytes: located_tensor.payload_bytes,
            role,
        });
    }
    if sources.is_empty() {
        return Err(LagunaArtifactValidationError::EmptyTensorAssembly { tensor_id });
    }
    Ok(sources)
}

const fn ordinary_source_role(
    component: LagunaTensorComponent,
    storage_encoding: &LagunaTensorStorageEncoding,
) -> LagunaTensorSourceRole {
    match component {
        LagunaTensorComponent::Weight => match storage_encoding {
            LagunaTensorStorageEncoding::DirectAffine { .. } => {
                LagunaTensorSourceRole::PackedWeightCodes
            }
            _ => LagunaTensorSourceRole::WeightValues,
        },
        LagunaTensorComponent::Scales => LagunaTensorSourceRole::GroupScales,
        LagunaTensorComponent::Biases => LagunaTensorSourceRole::AffineBiases,
        LagunaTensorComponent::WeightGlobalScale => LagunaTensorSourceRole::WeightGlobalScale,
        LagunaTensorComponent::InputGlobalScale => LagunaTensorSourceRole::InputGlobalScale,
        LagunaTensorComponent::LogicalShape => LagunaTensorSourceRole::LogicalShape,
        LagunaTensorComponent::AttentionKeyScaleMetadata => {
            LagunaTensorSourceRole::AttentionKeyScaleMetadata
        }
        LagunaTensorComponent::AttentionValueScaleMetadata => {
            LagunaTensorSourceRole::AttentionValueScaleMetadata
        }
        LagunaTensorComponent::ZeroPoint => LagunaTensorSourceRole::AffineBiases,
    }
}

const fn tensor_component(tensor_id: LagunaTensorId) -> LagunaTensorComponent {
    match tensor_id {
        LagunaTensorId::Global { component, .. } | LagunaTensorId::Layer { component, .. } => {
            component
        }
    }
}

const fn assembly_kind(assembly: &LagunaTensorAssembly) -> LagunaCanonicalTensorAssemblyKind {
    match assembly {
        LagunaTensorAssembly::DirectAlias { .. } => LagunaCanonicalTensorAssemblyKind::DirectAlias,
        LagunaTensorAssembly::StackedSource { .. } => {
            LagunaCanonicalTensorAssemblyKind::StackedSource
        }
        LagunaTensorAssembly::PerExpertStack { .. } => {
            LagunaCanonicalTensorAssemblyKind::PerExpertStack
        }
        LagunaTensorAssembly::FusedGateUpSource { projection, .. } => {
            LagunaCanonicalTensorAssemblyKind::FusedGateUpSource {
                projection: *projection,
            }
        }
        LagunaTensorAssembly::FusedPerExpertGateUp { projection, .. } => {
            LagunaCanonicalTensorAssemblyKind::FusedPerExpertGateUp {
                projection: *projection,
            }
        }
    }
}

const fn execution_dtype(execution_dtype: LagunaExecutionDtype) -> Dtype {
    match execution_dtype {
        LagunaExecutionDtype::Float16 => Dtype::F16,
        LagunaExecutionDtype::Bfloat16 => Dtype::BF16,
        LagunaExecutionDtype::Float32 => Dtype::F32,
    }
}
