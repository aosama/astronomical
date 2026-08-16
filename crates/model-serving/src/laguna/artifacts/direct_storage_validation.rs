use ::safetensors::Dtype;

use super::artifact_error::LagunaArtifactValidationError;
use super::canonical_tensor_contract::LagunaTensorSourceDescriptor;
use super::tensor_assembly::LagunaTensorAssembly;
use super::tensor_id::{LagunaTensorComponent, LagunaTensorId};
use super::tensor_storage::LagunaTensorStorageEncoding;

pub(super) fn validate_source_shapes(
    tensor_id: LagunaTensorId,
    logical_shape: &[usize],
    component: LagunaTensorComponent,
    storage_encoding: &LagunaTensorStorageEncoding,
    assembly: &LagunaTensorAssembly,
    sources: &[LagunaTensorSourceDescriptor],
) -> Result<(), LagunaArtifactValidationError> {
    let source_execution_shape = match assembly {
        LagunaTensorAssembly::DirectAlias { .. } | LagunaTensorAssembly::StackedSource { .. } => {
            logical_shape.to_vec()
        }
        LagunaTensorAssembly::PerExpertStack { .. } => logical_shape
            .get(1..)
            .ok_or(LagunaArtifactValidationError::TensorGeometryOverflow)?
            .to_vec(),
        LagunaTensorAssembly::FusedGateUpSource { .. } => fused_gate_up_shape(logical_shape)?,
        LagunaTensorAssembly::FusedPerExpertGateUp { .. } => {
            let per_expert_shape = logical_shape
                .get(1..)
                .ok_or(LagunaArtifactValidationError::TensorGeometryOverflow)?;
            fused_gate_up_shape(per_expert_shape)?
        }
    };
    let expected_source_shape = physical_source_shape(
        tensor_id,
        &source_execution_shape,
        component,
        storage_encoding,
    )?;
    let expected_source_count = match assembly {
        LagunaTensorAssembly::DirectAlias { .. }
        | LagunaTensorAssembly::StackedSource { .. }
        | LagunaTensorAssembly::FusedGateUpSource { .. } => 1,
        LagunaTensorAssembly::PerExpertStack { .. }
        | LagunaTensorAssembly::FusedPerExpertGateUp { .. } => logical_shape
            .first()
            .copied()
            .ok_or(LagunaArtifactValidationError::TensorGeometryOverflow)?,
    };
    if sources.len() != expected_source_count {
        return Err(LagunaArtifactValidationError::TensorShapeMismatch {
            tensor_id,
            expected_shape: logical_shape.to_vec(),
            actual_shape: sources
                .first()
                .map(|source| source.raw_shape.clone())
                .unwrap_or_default(),
        });
    }
    for source in sources {
        if source.raw_shape != expected_source_shape {
            return Err(LagunaArtifactValidationError::TensorShapeMismatch {
                tensor_id,
                expected_shape: expected_source_shape,
                actual_shape: source.raw_shape.clone(),
            });
        }
    }
    Ok(())
}

pub(super) fn validate_source_dtypes(
    tensor_id: LagunaTensorId,
    required_dtype: Option<Dtype>,
    sources: &[LagunaTensorSourceDescriptor],
) -> Result<Dtype, LagunaArtifactValidationError> {
    let first_dtype = sources
        .first()
        .map(|source| source.raw_dtype)
        .ok_or(LagunaArtifactValidationError::EmptyTensorAssembly { tensor_id })?;
    if sources.iter().any(|source| source.raw_dtype != first_dtype) {
        return Err(LagunaArtifactValidationError::MixedAssemblyDtypes { tensor_id });
    }
    let is_supported_dtype = required_dtype.map_or_else(
        || matches!(first_dtype, Dtype::F16 | Dtype::BF16 | Dtype::F32),
        |expected_dtype| first_dtype == expected_dtype,
    );
    if !is_supported_dtype {
        return Err(LagunaArtifactValidationError::TensorDtypeMismatch {
            tensor_id,
            expected_dtype: required_dtype.unwrap_or(Dtype::F16),
            actual_dtype: first_dtype,
        });
    }
    Ok(first_dtype)
}

pub(super) const fn required_component_storage_dtype(
    component: LagunaTensorComponent,
    storage_encoding: &LagunaTensorStorageEncoding,
) -> Option<Dtype> {
    match (storage_encoding, component) {
        (LagunaTensorStorageEncoding::DirectAffine { .. }, LagunaTensorComponent::Weight) => {
            Some(Dtype::U32)
        }
        _ => None,
    }
}

fn physical_source_shape(
    tensor_id: LagunaTensorId,
    source_execution_shape: &[usize],
    component: LagunaTensorComponent,
    storage_encoding: &LagunaTensorStorageEncoding,
) -> Result<Vec<usize>, LagunaArtifactValidationError> {
    let LagunaTensorStorageEncoding::DirectAffine { profile } = storage_encoding else {
        return Ok(source_execution_shape.to_vec());
    };
    let mut physical_shape = source_execution_shape.to_vec();
    let logical_input_width = physical_shape
        .last_mut()
        .ok_or(LagunaArtifactValidationError::TensorGeometryOverflow)?;
    let input_width = *logical_input_width;
    let physical_input_width = match component {
        LagunaTensorComponent::Weight => {
            let bit_width = usize::try_from(profile.bits())
                .map_err(|_| LagunaArtifactValidationError::TensorGeometryOverflow)?;
            let packed_input_bits = input_width
                .checked_mul(bit_width)
                .ok_or(LagunaArtifactValidationError::TensorGeometryOverflow)?;
            if !packed_input_bits.is_multiple_of(32) {
                return Err(invalid_affine_dimension(tensor_id, input_width, profile));
            }
            packed_input_bits / 32
        }
        LagunaTensorComponent::Scales | LagunaTensorComponent::Biases => {
            let group_size = usize::try_from(profile.group_size())
                .map_err(|_| LagunaArtifactValidationError::TensorGeometryOverflow)?;
            if group_size == 0 || !input_width.is_multiple_of(group_size) {
                return Err(invalid_affine_dimension(tensor_id, input_width, profile));
            }
            input_width / group_size
        }
        _ => return Err(LagunaArtifactValidationError::UnexpectedCanonicalTensor { tensor_id }),
    };
    *logical_input_width = physical_input_width;
    Ok(physical_shape)
}

fn invalid_affine_dimension(
    tensor_id: LagunaTensorId,
    logical_input_width: usize,
    profile: &crate::laguna::LagunaAffineProfile,
) -> LagunaArtifactValidationError {
    LagunaArtifactValidationError::InvalidAffineDimension {
        tensor_id,
        logical_input_width,
        bit_width: profile.bits(),
        group_size: profile.group_size(),
    }
}

fn fused_gate_up_shape(
    logical_shape: &[usize],
) -> Result<Vec<usize>, LagunaArtifactValidationError> {
    let mut fused_shape = logical_shape.to_vec();
    let intermediate_axis = fused_shape
        .len()
        .checked_sub(2)
        .ok_or(LagunaArtifactValidationError::TensorGeometryOverflow)?;
    fused_shape[intermediate_axis] = fused_shape[intermediate_axis]
        .checked_mul(2)
        .ok_or(LagunaArtifactValidationError::TensorGeometryOverflow)?;
    Ok(fused_shape)
}
