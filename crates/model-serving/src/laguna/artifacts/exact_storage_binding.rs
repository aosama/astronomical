use std::collections::BTreeMap;

use ::safetensors::Dtype;

use super::artifact_error::LagunaArtifactValidationError;
use super::canonical_tensor_contract::{
    LagunaCanonicalSourceLayout, LagunaCanonicalTensorAssemblyKind, LagunaTensorSourceDescriptor,
    LagunaTensorSourceRole, LocatedRawTensorDescriptor,
};
use super::direct_storage_validation::validate_source_dtypes;
use super::exact_storage_validation::{
    validate_block_scales, validate_declared_block_geometry, validate_exact_sources,
    validate_group_scales, validate_last_axis_divided, validate_last_axis_divided_with_dtypes,
    validate_scalar_sources, validate_symmetric_codes,
};
use super::expected_tensors::LagunaExpectedTensor;
use super::tensor_assembly::LagunaTensorAssembly;
use super::tensor_id::{LagunaTensorComponent, LagunaTensorId};
use super::tensor_name_contract::LagunaTensorNameContract;
use super::tensor_storage::LagunaTensorStorageEncoding;
use crate::laguna::LagunaStorageDescriptor;

/// Inventory-backed recipe that retains source intervals without reading payload bytes.
pub(super) struct LagunaExactStorageBinding {
    pub(super) storage_dtype: Dtype,
    pub(super) storage_encoding: LagunaTensorStorageEncoding,
    pub(super) assembly_kind: LagunaCanonicalTensorAssemblyKind,
    pub(super) sources: Vec<LagunaTensorSourceDescriptor>,
}

pub(super) fn bind_exact_storage(
    tensor_id: LagunaTensorId,
    expected_tensor: &LagunaExpectedTensor,
    name_contract: &LagunaTensorNameContract,
    located_tensors: &BTreeMap<String, LocatedRawTensorDescriptor>,
) -> Result<Option<LagunaExactStorageBinding>, LagunaArtifactValidationError> {
    // Norms, routers, and other non-matrix owners remain ordinary model-float tensors.
    if expected_tensor.canonical_module_name.is_none() {
        return Ok(None);
    }
    match &expected_tensor.storage_encoding {
        LagunaTensorStorageEncoding::SymmetricPackedAffine { .. } => {
            bind_symmetric(tensor_id, expected_tensor, name_contract, located_tensors).map(Some)
        }
        LagunaTensorStorageEncoding::NativeNvfp4 { .. } => {
            bind_native_nvfp4(tensor_id, expected_tensor, name_contract, located_tensors).map(Some)
        }
        LagunaTensorStorageEncoding::TwoLevelCompressedNvfp4 { .. } => {
            bind_two_level_nvfp4(tensor_id, expected_tensor, name_contract, located_tensors)
                .map(Some)
        }
        LagunaTensorStorageEncoding::BlockFp8 { .. } => {
            bind_block_fp8(tensor_id, expected_tensor, name_contract, located_tensors).map(Some)
        }
        LagunaTensorStorageEncoding::Unquantized
        | LagunaTensorStorageEncoding::DirectAffine { .. } => Ok(None),
    }
}

pub(super) fn is_exact_storage_source(
    storage: &LagunaStorageDescriptor,
    tensor_id: LagunaTensorId,
    expected_tensors: &BTreeMap<LagunaTensorId, LagunaExpectedTensor>,
) -> bool {
    let component = component(tensor_id);
    if matches!(
        component,
        LagunaTensorComponent::AttentionKeyScaleMetadata
            | LagunaTensorComponent::AttentionValueScaleMetadata
    ) {
        return storage.has_fp8_kv_cache();
    }
    let weight_id = with_component(tensor_id, LagunaTensorComponent::Weight);
    let Some(weight) = expected_tensors.get(&weight_id) else {
        return false;
    };
    match (&weight.storage_encoding, component) {
        (
            LagunaTensorStorageEncoding::SymmetricPackedAffine { .. },
            LagunaTensorComponent::LogicalShape,
        ) => true,
        (
            LagunaTensorStorageEncoding::TwoLevelCompressedNvfp4 { .. },
            LagunaTensorComponent::Scales
            | LagunaTensorComponent::WeightGlobalScale
            | LagunaTensorComponent::InputGlobalScale
            | LagunaTensorComponent::LogicalShape,
        ) => true,
        (LagunaTensorStorageEncoding::BlockFp8 { .. }, LagunaTensorComponent::Scales) => true,
        _ => false,
    }
}

fn bind_symmetric(
    tensor_id: LagunaTensorId,
    expected_tensor: &LagunaExpectedTensor,
    name_contract: &LagunaTensorNameContract,
    located_tensors: &BTreeMap<String, LocatedRawTensorDescriptor>,
) -> Result<LagunaExactStorageBinding, LagunaArtifactValidationError> {
    let component = component(tensor_id);
    let weight_id = with_component(tensor_id, LagunaTensorComponent::Weight);
    let scale_id = with_component(tensor_id, LagunaTensorComponent::Scales);
    let (source_id, source_role, assembly_kind) = match component {
        LagunaTensorComponent::Weight => (
            weight_id,
            LagunaTensorSourceRole::PackedWeightCodes,
            LagunaCanonicalTensorAssemblyKind::ReinterpretPackedBits {
                source_layout: source_layout(required_assembly(name_contract, weight_id)?),
            },
        ),
        LagunaTensorComponent::Scales => (
            scale_id,
            LagunaTensorSourceRole::GroupScales,
            ordinary_kind(required_assembly(name_contract, scale_id)?),
        ),
        LagunaTensorComponent::Biases => (
            scale_id,
            LagunaTensorSourceRole::GroupScales,
            LagunaCanonicalTensorAssemblyKind::DeriveSymmetricBias {
                source_layout: source_layout(required_assembly(name_contract, scale_id)?),
                negative_code_offset: 8,
            },
        ),
        _ => return Err(LagunaArtifactValidationError::UnexpectedCanonicalTensor { tensor_id }),
    };
    let assembly = required_assembly(name_contract, source_id)?;
    let source_matrix_shape = matrix_source_shape(&expected_tensor.logical_shape, assembly)?;
    let mut sources = resolve_sources(source_id, assembly, located_tensors, source_role)?;
    let storage_dtype = match component {
        LagunaTensorComponent::Weight => {
            // I32 and U8 are both exact source packings, while the future consumer
            // sees one canonical U32 bit representation without source conversion.
            validate_symmetric_codes(tensor_id, &source_matrix_shape, &sources)?;
            Dtype::U32
        }
        LagunaTensorComponent::Scales | LagunaTensorComponent::Biases => {
            let source_dtype = validate_source_dtypes(tensor_id, None, &sources)?;
            validate_group_scales(tensor_id, &source_matrix_shape, 32, source_dtype, &sources)?;
            source_dtype
        }
        _ => return Err(LagunaArtifactValidationError::UnexpectedCanonicalTensor { tensor_id }),
    };
    if component == LagunaTensorComponent::Weight {
        append_optional_sources(
            with_component(tensor_id, LagunaTensorComponent::LogicalShape),
            LagunaTensorSourceRole::LogicalShape,
            name_contract,
            located_tensors,
            &mut sources,
            Dtype::I64,
            &[2],
        )?;
    }
    Ok(LagunaExactStorageBinding {
        storage_dtype,
        storage_encoding: expected_tensor.storage_encoding.clone(),
        assembly_kind,
        sources,
    })
}

fn bind_native_nvfp4(
    tensor_id: LagunaTensorId,
    expected_tensor: &LagunaExpectedTensor,
    name_contract: &LagunaTensorNameContract,
    located_tensors: &BTreeMap<String, LocatedRawTensorDescriptor>,
) -> Result<LagunaExactStorageBinding, LagunaArtifactValidationError> {
    let assembly = required_assembly(name_contract, tensor_id)?;
    let source_matrix_shape = matrix_source_shape(&expected_tensor.logical_shape, assembly)?;
    let (role, dtype, divisor) = match component(tensor_id) {
        LagunaTensorComponent::Weight => (LagunaTensorSourceRole::WeightCodes, Dtype::U32, 8),
        LagunaTensorComponent::Scales => (LagunaTensorSourceRole::GroupScales, Dtype::U8, 16),
        _ => return Err(LagunaArtifactValidationError::UnexpectedCanonicalTensor { tensor_id }),
    };
    let sources = resolve_sources(tensor_id, assembly, located_tensors, role)?;
    validate_last_axis_divided(tensor_id, &source_matrix_shape, divisor, dtype, &sources)?;
    Ok(LagunaExactStorageBinding {
        storage_dtype: dtype,
        storage_encoding: expected_tensor.storage_encoding.clone(),
        assembly_kind: LagunaCanonicalTensorAssemblyKind::NativeNvfp4 {
            source_layout: source_layout(assembly),
        },
        sources,
    })
}

fn bind_two_level_nvfp4(
    tensor_id: LagunaTensorId,
    expected_tensor: &LagunaExpectedTensor,
    name_contract: &LagunaTensorNameContract,
    located_tensors: &BTreeMap<String, LocatedRawTensorDescriptor>,
) -> Result<LagunaExactStorageBinding, LagunaArtifactValidationError> {
    let weight_assembly = required_assembly(name_contract, tensor_id)?;
    let source_matrix_shape = matrix_source_shape(&expected_tensor.logical_shape, weight_assembly)?;
    let mut sources = resolve_sources(
        tensor_id,
        weight_assembly,
        located_tensors,
        LagunaTensorSourceRole::PackedWeightCodes,
    )?;
    validate_last_axis_divided(tensor_id, &source_matrix_shape, 2, Dtype::U8, &sources)?;
    let scale_id = with_component(tensor_id, LagunaTensorComponent::Scales);
    let scale_assembly = required_assembly(name_contract, scale_id)?;
    let scale_matrix_shape = matrix_source_shape(&expected_tensor.logical_shape, scale_assembly)?;
    let scale_sources = resolve_sources(
        scale_id,
        scale_assembly,
        located_tensors,
        LagunaTensorSourceRole::GroupScales,
    )?;
    // Some writers preserve E4M3 while MLX-oriented writers expose the same encoded bytes as U8.
    validate_last_axis_divided_with_dtypes(
        scale_id,
        &scale_matrix_shape,
        16,
        &[Dtype::F8_E4M3, Dtype::U8],
        &scale_sources,
    )?;
    sources.extend(scale_sources);
    append_required_scalar_sources(
        with_component(tensor_id, LagunaTensorComponent::WeightGlobalScale),
        LagunaTensorSourceRole::WeightGlobalScale,
        name_contract,
        located_tensors,
        &mut sources,
    )?;
    append_optional_scalar_sources(
        with_component(tensor_id, LagunaTensorComponent::InputGlobalScale),
        LagunaTensorSourceRole::InputGlobalScale,
        name_contract,
        located_tensors,
        &mut sources,
        Dtype::F32,
    )?;
    append_optional_sources(
        with_component(tensor_id, LagunaTensorComponent::LogicalShape),
        LagunaTensorSourceRole::LogicalShape,
        name_contract,
        located_tensors,
        &mut sources,
        Dtype::I64,
        &[2],
    )?;
    Ok(LagunaExactStorageBinding {
        storage_dtype: Dtype::U8,
        storage_encoding: expected_tensor.storage_encoding.clone(),
        assembly_kind: LagunaCanonicalTensorAssemblyKind::TwoLevelCompressedNvfp4 {
            source_layout: source_layout(weight_assembly),
        },
        sources,
    })
}

fn bind_block_fp8(
    tensor_id: LagunaTensorId,
    expected_tensor: &LagunaExpectedTensor,
    name_contract: &LagunaTensorNameContract,
    located_tensors: &BTreeMap<String, LocatedRawTensorDescriptor>,
) -> Result<LagunaExactStorageBinding, LagunaArtifactValidationError> {
    let weight_assembly = required_assembly(name_contract, tensor_id)?;
    let source_matrix_shape = matrix_source_shape(&expected_tensor.logical_shape, weight_assembly)?;
    let mut sources = resolve_sources(
        tensor_id,
        weight_assembly,
        located_tensors,
        LagunaTensorSourceRole::WeightCodes,
    )?;
    validate_exact_sources(tensor_id, &source_matrix_shape, Dtype::F8_E4M3, &sources)?;
    let scale_id = with_component(tensor_id, LagunaTensorComponent::Scales);
    let scale_assembly = required_assembly(name_contract, scale_id)?;
    let scale_sources = resolve_sources(
        scale_id,
        scale_assembly,
        located_tensors,
        LagunaTensorSourceRole::BlockScales,
    )?;
    let (block_row_extent, block_column_extent) =
        validate_block_scales(tensor_id, &source_matrix_shape, &scale_sources)?;
    validate_declared_block_geometry(
        tensor_id,
        &expected_tensor.storage_encoding,
        block_row_extent,
        block_column_extent,
    )?;
    sources.extend(scale_sources);
    Ok(LagunaExactStorageBinding {
        storage_dtype: Dtype::F8_E4M3,
        storage_encoding: expected_tensor.storage_encoding.clone(),
        assembly_kind: LagunaCanonicalTensorAssemblyKind::BlockFp8 {
            source_layout: source_layout(weight_assembly),
        },
        sources,
    })
}

fn append_required_scalar_sources(
    source_id: LagunaTensorId,
    role: LagunaTensorSourceRole,
    name_contract: &LagunaTensorNameContract,
    located_tensors: &BTreeMap<String, LocatedRawTensorDescriptor>,
    destination: &mut Vec<LagunaTensorSourceDescriptor>,
) -> Result<(), LagunaArtifactValidationError> {
    let assembly = required_assembly(name_contract, source_id)?;
    let sources = resolve_sources(source_id, assembly, located_tensors, role)?;
    validate_scalar_sources(source_id, Dtype::F32, &sources)?;
    destination.extend(sources);
    Ok(())
}

fn append_optional_scalar_sources(
    source_id: LagunaTensorId,
    role: LagunaTensorSourceRole,
    name_contract: &LagunaTensorNameContract,
    located_tensors: &BTreeMap<String, LocatedRawTensorDescriptor>,
    destination: &mut Vec<LagunaTensorSourceDescriptor>,
    dtype: Dtype,
) -> Result<(), LagunaArtifactValidationError> {
    let Some(assembly) = name_contract.assemblies().get(&source_id) else {
        return Ok(());
    };
    let sources = resolve_sources(source_id, assembly, located_tensors, role)?;
    validate_scalar_sources(source_id, dtype, &sources)?;
    destination.extend(sources);
    Ok(())
}

fn append_optional_sources(
    source_id: LagunaTensorId,
    role: LagunaTensorSourceRole,
    name_contract: &LagunaTensorNameContract,
    located_tensors: &BTreeMap<String, LocatedRawTensorDescriptor>,
    destination: &mut Vec<LagunaTensorSourceDescriptor>,
    dtype: Dtype,
    shape: &[usize],
) -> Result<(), LagunaArtifactValidationError> {
    let Some(assembly) = name_contract.assemblies().get(&source_id) else {
        return Ok(());
    };
    let sources = resolve_sources(source_id, assembly, located_tensors, role)?;
    validate_exact_sources(source_id, shape, dtype, &sources)?;
    destination.extend(sources);
    Ok(())
}

fn resolve_sources(
    tensor_id: LagunaTensorId,
    assembly: &LagunaTensorAssembly,
    located_tensors: &BTreeMap<String, LocatedRawTensorDescriptor>,
    role: LagunaTensorSourceRole,
) -> Result<Vec<LagunaTensorSourceDescriptor>, LagunaArtifactValidationError> {
    let mut sources = Vec::with_capacity(assembly.sources().len());
    for source in assembly.sources() {
        let raw_name = source.raw_name();
        let located = located_tensors.get(raw_name).ok_or_else(|| {
            LagunaArtifactValidationError::CanonicalSourceMissing {
                tensor_id,
                tensor_name: raw_name.to_owned(),
            }
        })?;
        sources.push(LagunaTensorSourceDescriptor {
            shard_file_name: located.shard_file_name.clone(),
            raw_tensor_name: located.raw_tensor_name.clone(),
            data_start_offset_bytes: located.data_start_offset_bytes,
            data_end_offset_bytes: located.data_end_offset_bytes,
            raw_shape: located.shape.clone(),
            raw_dtype: located.dtype,
            payload_bytes: located.payload_bytes,
            role,
        });
    }
    if sources.is_empty() {
        return Err(LagunaArtifactValidationError::EmptyTensorAssembly { tensor_id });
    }
    Ok(sources)
}

fn required_assembly(
    name_contract: &LagunaTensorNameContract,
    tensor_id: LagunaTensorId,
) -> Result<&LagunaTensorAssembly, LagunaArtifactValidationError> {
    name_contract
        .assemblies()
        .get(&tensor_id)
        .ok_or(LagunaArtifactValidationError::ExpectedTensorMissing { tensor_id })
}

fn matrix_source_shape(
    logical_shape: &[usize],
    assembly: &LagunaTensorAssembly,
) -> Result<Vec<usize>, LagunaArtifactValidationError> {
    match assembly {
        LagunaTensorAssembly::DirectAlias { .. } | LagunaTensorAssembly::StackedSource { .. } => {
            Ok(logical_shape.to_vec())
        }
        LagunaTensorAssembly::PerExpertStack { .. } => logical_shape
            .get(1..)
            .map(<[usize]>::to_vec)
            .ok_or(LagunaArtifactValidationError::TensorGeometryOverflow),
        LagunaTensorAssembly::FusedGateUpSource { .. } => fused_shape(logical_shape),
        LagunaTensorAssembly::FusedPerExpertGateUp { .. } => logical_shape
            .get(1..)
            .ok_or(LagunaArtifactValidationError::TensorGeometryOverflow)
            .and_then(fused_shape),
    }
}

fn fused_shape(logical_shape: &[usize]) -> Result<Vec<usize>, LagunaArtifactValidationError> {
    let mut shape = logical_shape.to_vec();
    let output_axis = shape
        .len()
        .checked_sub(2)
        .ok_or(LagunaArtifactValidationError::TensorGeometryOverflow)?;
    shape[output_axis] = shape[output_axis]
        .checked_mul(2)
        .ok_or(LagunaArtifactValidationError::TensorGeometryOverflow)?;
    Ok(shape)
}

const fn source_layout(assembly: &LagunaTensorAssembly) -> LagunaCanonicalSourceLayout {
    match assembly {
        LagunaTensorAssembly::DirectAlias { .. } => LagunaCanonicalSourceLayout::Direct,
        LagunaTensorAssembly::StackedSource { .. } => LagunaCanonicalSourceLayout::Stacked,
        LagunaTensorAssembly::PerExpertStack { .. } => LagunaCanonicalSourceLayout::PerExpert,
        LagunaTensorAssembly::FusedGateUpSource { projection, .. } => {
            LagunaCanonicalSourceLayout::FusedStacked {
                projection: *projection,
            }
        }
        LagunaTensorAssembly::FusedPerExpertGateUp { projection, .. } => {
            LagunaCanonicalSourceLayout::FusedPerExpert {
                projection: *projection,
            }
        }
    }
}

const fn ordinary_kind(assembly: &LagunaTensorAssembly) -> LagunaCanonicalTensorAssemblyKind {
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

const fn component(tensor_id: LagunaTensorId) -> LagunaTensorComponent {
    match tensor_id {
        LagunaTensorId::Global { component, .. } | LagunaTensorId::Layer { component, .. } => {
            component
        }
    }
}

const fn with_component(
    tensor_id: LagunaTensorId,
    replacement: LagunaTensorComponent,
) -> LagunaTensorId {
    match tensor_id {
        LagunaTensorId::Global { role, .. } => LagunaTensorId::Global {
            role,
            component: replacement,
        },
        LagunaTensorId::Layer {
            layer_index, role, ..
        } => LagunaTensorId::Layer {
            layer_index,
            role,
            component: replacement,
        },
    }
}
