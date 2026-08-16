use ::safetensors::Dtype;

use super::artifact_error::LagunaArtifactValidationError;
use super::canonical_tensor_contract::LagunaTensorSourceDescriptor;
use super::tensor_id::LagunaTensorId;
use super::tensor_storage::LagunaTensorStorageEncoding;

pub(super) fn validate_symmetric_codes(
    tensor_id: LagunaTensorId,
    matrix_shape: &[usize],
    sources: &[LagunaTensorSourceDescriptor],
) -> Result<(), LagunaArtifactValidationError> {
    let first_dtype = sources
        .first()
        .map(|source| source.raw_dtype)
        .ok_or(LagunaArtifactValidationError::EmptyTensorAssembly { tensor_id })?;
    if sources.iter().any(|source| source.raw_dtype != first_dtype) {
        return Err(LagunaArtifactValidationError::MixedAssemblyDtypes { tensor_id });
    }
    let divisor = match first_dtype {
        Dtype::I32 => 8,
        Dtype::U8 => 2,
        actual_dtype => {
            return Err(LagunaArtifactValidationError::TensorDtypeMismatch {
                tensor_id,
                expected_dtype: Dtype::I32,
                actual_dtype,
            });
        }
    };
    validate_last_axis_divided(tensor_id, matrix_shape, divisor, first_dtype, sources)
}

pub(super) fn validate_group_scales(
    tensor_id: LagunaTensorId,
    matrix_shape: &[usize],
    divisor: usize,
    dtype: Dtype,
    sources: &[LagunaTensorSourceDescriptor],
) -> Result<(), LagunaArtifactValidationError> {
    validate_last_axis_divided(tensor_id, matrix_shape, divisor, dtype, sources)
}

pub(super) fn validate_last_axis_divided(
    tensor_id: LagunaTensorId,
    matrix_shape: &[usize],
    divisor: usize,
    dtype: Dtype,
    sources: &[LagunaTensorSourceDescriptor],
) -> Result<(), LagunaArtifactValidationError> {
    let mut expected_shape = matrix_shape.to_vec();
    let input_width = expected_shape
        .last_mut()
        .ok_or(LagunaArtifactValidationError::TensorGeometryOverflow)?;
    if divisor == 0 || !(*input_width).is_multiple_of(divisor) {
        return Err(LagunaArtifactValidationError::TensorShapeMismatch {
            tensor_id,
            expected_shape,
            actual_shape: sources
                .first()
                .map(|source| source.raw_shape.clone())
                .unwrap_or_default(),
        });
    }
    *input_width /= divisor;
    validate_exact_sources(tensor_id, &expected_shape, dtype, sources)
}

pub(super) fn validate_last_axis_divided_with_dtypes(
    tensor_id: LagunaTensorId,
    matrix_shape: &[usize],
    divisor: usize,
    accepted_dtypes: &[Dtype],
    sources: &[LagunaTensorSourceDescriptor],
) -> Result<(), LagunaArtifactValidationError> {
    let mut expected_shape = matrix_shape.to_vec();
    let input_width = expected_shape
        .last_mut()
        .ok_or(LagunaArtifactValidationError::TensorGeometryOverflow)?;
    if divisor == 0 || !(*input_width).is_multiple_of(divisor) {
        return Err(LagunaArtifactValidationError::TensorShapeMismatch {
            tensor_id,
            expected_shape,
            actual_shape: sources
                .first()
                .map(|source| source.raw_shape.clone())
                .unwrap_or_default(),
        });
    }
    *input_width /= divisor;
    validate_sources_with_dtypes(tensor_id, &expected_shape, accepted_dtypes, sources)
}

pub(super) fn validate_scalar_sources(
    tensor_id: LagunaTensorId,
    expected_dtype: Dtype,
    sources: &[LagunaTensorSourceDescriptor],
) -> Result<(), LagunaArtifactValidationError> {
    for source in sources {
        if !source.raw_shape.is_empty() && source.raw_shape != [1] {
            return Err(LagunaArtifactValidationError::TensorShapeMismatch {
                tensor_id,
                expected_shape: Vec::new(),
                actual_shape: source.raw_shape.clone(),
            });
        }
        if source.raw_dtype != expected_dtype {
            return Err(LagunaArtifactValidationError::TensorDtypeMismatch {
                tensor_id,
                expected_dtype,
                actual_dtype: source.raw_dtype,
            });
        }
    }
    Ok(())
}

pub(super) fn validate_block_scales(
    tensor_id: LagunaTensorId,
    matrix_shape: &[usize],
    sources: &[LagunaTensorSourceDescriptor],
) -> Result<(usize, usize), LagunaArtifactValidationError> {
    let scale_shape = sources
        .first()
        .map(|source| source.raw_shape.clone())
        .ok_or(LagunaArtifactValidationError::EmptyTensorAssembly { tensor_id })?;
    if matrix_shape.len() < 2 || scale_shape.len() != matrix_shape.len() {
        return Err(invalid_block_coverage(
            tensor_id,
            matrix_shape,
            &scale_shape,
        ));
    }
    let leading_count = matrix_shape.len() - 2;
    if matrix_shape[..leading_count] != scale_shape[..leading_count]
        || scale_shape[leading_count] == 0
        || scale_shape[leading_count + 1] == 0
        || !matrix_shape[leading_count].is_multiple_of(scale_shape[leading_count])
        || !matrix_shape[leading_count + 1].is_multiple_of(scale_shape[leading_count + 1])
    {
        return Err(invalid_block_coverage(
            tensor_id,
            matrix_shape,
            &scale_shape,
        ));
    }
    validate_exact_sources(tensor_id, &scale_shape, Dtype::F32, sources)?;
    Ok((
        matrix_shape[leading_count] / scale_shape[leading_count],
        matrix_shape[leading_count + 1] / scale_shape[leading_count + 1],
    ))
}

pub(super) fn validate_declared_block_geometry(
    tensor_id: LagunaTensorId,
    storage_encoding: &LagunaTensorStorageEncoding,
    actual_block_row_extent: usize,
    actual_block_column_extent: usize,
) -> Result<(), LagunaArtifactValidationError> {
    let LagunaTensorStorageEncoding::BlockFp8 {
        block_row_extent: declared_block_row_extent,
        block_column_extent: declared_block_column_extent,
    } = storage_encoding
    else {
        return Err(LagunaArtifactValidationError::UnexpectedCanonicalTensor { tensor_id });
    };
    if (actual_block_row_extent, actual_block_column_extent)
        != (*declared_block_row_extent, *declared_block_column_extent)
    {
        return Err(LagunaArtifactValidationError::BlockFp8GeometryMismatch {
            tensor_id,
            declared_block_row_extent: *declared_block_row_extent,
            declared_block_column_extent: *declared_block_column_extent,
            actual_block_row_extent,
            actual_block_column_extent,
        });
    }
    Ok(())
}

pub(super) fn validate_exact_sources(
    tensor_id: LagunaTensorId,
    expected_shape: &[usize],
    expected_dtype: Dtype,
    sources: &[LagunaTensorSourceDescriptor],
) -> Result<(), LagunaArtifactValidationError> {
    for source in sources {
        if source.raw_shape != expected_shape {
            return Err(LagunaArtifactValidationError::TensorShapeMismatch {
                tensor_id,
                expected_shape: expected_shape.to_vec(),
                actual_shape: source.raw_shape.clone(),
            });
        }
        if source.raw_dtype != expected_dtype {
            return Err(LagunaArtifactValidationError::TensorDtypeMismatch {
                tensor_id,
                expected_dtype,
                actual_dtype: source.raw_dtype,
            });
        }
    }
    Ok(())
}

fn validate_sources_with_dtypes(
    tensor_id: LagunaTensorId,
    expected_shape: &[usize],
    accepted_dtypes: &[Dtype],
    sources: &[LagunaTensorSourceDescriptor],
) -> Result<(), LagunaArtifactValidationError> {
    let expected_dtype = accepted_dtypes
        .first()
        .copied()
        .ok_or(LagunaArtifactValidationError::TensorGeometryOverflow)?;
    for source in sources {
        if source.raw_shape != expected_shape {
            return Err(LagunaArtifactValidationError::TensorShapeMismatch {
                tensor_id,
                expected_shape: expected_shape.to_vec(),
                actual_shape: source.raw_shape.clone(),
            });
        }
        if !accepted_dtypes.contains(&source.raw_dtype) {
            return Err(LagunaArtifactValidationError::TensorDtypeMismatch {
                tensor_id,
                expected_dtype,
                actual_dtype: source.raw_dtype,
            });
        }
    }
    Ok(())
}

fn invalid_block_coverage(
    tensor_id: LagunaTensorId,
    logical_shape: &[usize],
    scale_shape: &[usize],
) -> LagunaArtifactValidationError {
    LagunaArtifactValidationError::InvalidBlockFp8Coverage {
        tensor_id,
        logical_shape: logical_shape.to_vec(),
        scale_shape: scale_shape.to_vec(),
    }
}
