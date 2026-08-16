use std::path::PathBuf;

use ::safetensors::Dtype;

use crate::expert_paging::SafetensorsDtype;
use crate::laguna::artifacts::{
    LagunaCanonicalTensorAssemblyKind, LagunaCanonicalTensorDescriptor, LagunaExpertProjection,
    LagunaTensorComponent, LagunaTensorId,
};

use super::error::LagunaPagingError;
use super::layer_plan::LagunaExpertSourceSlice;

pub(super) fn expert_source_slices(
    descriptor: &LagunaCanonicalTensorDescriptor,
    expert_capacity: usize,
) -> Result<Vec<LagunaExpertSourceSlice>, LagunaPagingError> {
    match descriptor.assembly_kind() {
        LagunaCanonicalTensorAssemblyKind::StackedSource => {
            stacked_expert_slices(descriptor, expert_capacity, 1)
        }
        LagunaCanonicalTensorAssemblyKind::PerExpertStack => {
            per_expert_slices(descriptor, expert_capacity, 1)
        }
        LagunaCanonicalTensorAssemblyKind::FusedGateUpSource { projection } => {
            fused_stacked_slices(descriptor, expert_capacity, projection)
        }
        LagunaCanonicalTensorAssemblyKind::FusedPerExpertGateUp { projection } => {
            fused_per_expert_slices(descriptor, expert_capacity, projection)
        }
        LagunaCanonicalTensorAssemblyKind::DirectAlias
        | LagunaCanonicalTensorAssemblyKind::ReinterpretPackedBits { .. }
        | LagunaCanonicalTensorAssemblyKind::DeriveSymmetricBias { .. }
        | LagunaCanonicalTensorAssemblyKind::NativeNvfp4 { .. }
        | LagunaCanonicalTensorAssemblyKind::TwoLevelCompressedNvfp4 { .. }
        | LagunaCanonicalTensorAssemblyKind::BlockFp8 { .. } => {
            Err(LagunaPagingError::UnsupportedRoutedStorage {
                tensor_id: descriptor.tensor_id(),
            })
        }
    }
}

pub(super) fn compact_trailing_shape(
    descriptor: &LagunaCanonicalTensorDescriptor,
    expert_capacity: usize,
) -> Result<Vec<usize>, LagunaPagingError> {
    let first_source =
        descriptor
            .sources()
            .first()
            .ok_or(LagunaPagingError::MissingSourceInterval {
                tensor_id: descriptor.tensor_id(),
            })?;
    let raw_shape = first_source.raw_shape();
    let trailing = match descriptor.assembly_kind() {
        LagunaCanonicalTensorAssemblyKind::StackedSource
        | LagunaCanonicalTensorAssemblyKind::FusedGateUpSource { .. } => {
            if raw_shape.first().copied() != Some(expert_capacity) || raw_shape.len() < 2 {
                return Err(LagunaPagingError::ExpertPayloadNotDivisible {
                    tensor_id: descriptor.tensor_id(),
                });
            }
            raw_shape[1..].to_vec()
        }
        LagunaCanonicalTensorAssemblyKind::PerExpertStack
        | LagunaCanonicalTensorAssemblyKind::FusedPerExpertGateUp { .. } => raw_shape.to_vec(),
        _ => {
            return Err(LagunaPagingError::UnsupportedRoutedStorage {
                tensor_id: descriptor.tensor_id(),
            });
        }
    };
    match descriptor.assembly_kind() {
        LagunaCanonicalTensorAssemblyKind::FusedGateUpSource { .. }
        | LagunaCanonicalTensorAssemblyKind::FusedPerExpertGateUp { .. } => {
            if trailing.first().is_none_or(|extent| *extent % 2 != 0) {
                return Err(LagunaPagingError::FusedProjectionNotEven {
                    tensor_id: descriptor.tensor_id(),
                });
            }
            let mut halved = trailing;
            halved[0] /= 2;
            Ok(halved)
        }
        _ => Ok(trailing),
    }
}

pub(super) fn projection_name(projection: LagunaExpertProjection) -> &'static str {
    match projection {
        LagunaExpertProjection::Gate => "gate_proj",
        LagunaExpertProjection::Up => "up_proj",
        LagunaExpertProjection::Down => "down_proj",
    }
}

pub(super) fn parameter_name(component: LagunaTensorComponent) -> &'static str {
    match component {
        LagunaTensorComponent::Weight => "weight",
        LagunaTensorComponent::Scales => "scales",
        LagunaTensorComponent::Biases => "biases",
        LagunaTensorComponent::WeightGlobalScale
        | LagunaTensorComponent::InputGlobalScale
        | LagunaTensorComponent::LogicalShape
        | LagunaTensorComponent::ZeroPoint
        | LagunaTensorComponent::AttentionKeyScaleMetadata
        | LagunaTensorComponent::AttentionValueScaleMetadata => "unsupported",
    }
}

pub(super) fn map_dtype(
    dtype: Dtype,
    tensor_id: LagunaTensorId,
) -> Result<SafetensorsDtype, LagunaPagingError> {
    match dtype {
        Dtype::BOOL => Ok(SafetensorsDtype::Bool),
        Dtype::U8 => Ok(SafetensorsDtype::Uint8),
        Dtype::I8 => Ok(SafetensorsDtype::Int8),
        Dtype::F8_E4M3 => Ok(SafetensorsDtype::Float8E4M3),
        Dtype::I16 => Ok(SafetensorsDtype::Int16),
        Dtype::F16 => Ok(SafetensorsDtype::Float16),
        Dtype::BF16 => Ok(SafetensorsDtype::BFloat16),
        Dtype::I32 => Ok(SafetensorsDtype::Int32),
        Dtype::U32 => Ok(SafetensorsDtype::Uint32),
        Dtype::F32 => Ok(SafetensorsDtype::Float32),
        Dtype::I64 => Ok(SafetensorsDtype::Int64),
        Dtype::U64 => Ok(SafetensorsDtype::Uint64),
        _ => Err(LagunaPagingError::UnsupportedSourceDtype { tensor_id }),
    }
}

fn stacked_expert_slices(
    descriptor: &LagunaCanonicalTensorDescriptor,
    expert_capacity: usize,
    split_denominator: u64,
) -> Result<Vec<LagunaExpertSourceSlice>, LagunaPagingError> {
    let source = descriptor
        .sources()
        .first()
        .ok_or(LagunaPagingError::MissingSourceInterval {
            tensor_id: descriptor.tensor_id(),
        })?;
    let expert_capacity_bytes =
        u64::try_from(expert_capacity).map_err(|_| LagunaPagingError::ExpertPayloadOverflow {
            layer_index: layer_index(descriptor.tensor_id()),
        })?;
    if expert_capacity_bytes == 0 || source.payload_bytes() % expert_capacity_bytes != 0 {
        return Err(LagunaPagingError::ExpertPayloadNotDivisible {
            tensor_id: descriptor.tensor_id(),
        });
    }
    let fused_bytes_per_expert = source.payload_bytes() / expert_capacity_bytes;
    if fused_bytes_per_expert % split_denominator != 0 {
        return Err(LagunaPagingError::FusedProjectionNotEven {
            tensor_id: descriptor.tensor_id(),
        });
    }
    let bytes_per_expert = fused_bytes_per_expert / split_denominator;
    let source_byte_count = usize::try_from(bytes_per_expert).map_err(|_| {
        LagunaPagingError::ExpertPayloadOverflow {
            layer_index: layer_index(descriptor.tensor_id()),
        }
    })?;
    (0..expert_capacity)
        .map(|expert_index| {
            let expert_start = source
                .data_start_offset_bytes()
                .checked_add(u64::try_from(expert_index).ok()? * fused_bytes_per_expert)?;
            Some(LagunaExpertSourceSlice {
                source_file: PathBuf::from(source.shard_file_name()),
                source_file_offset: expert_start,
                source_byte_count,
            })
        })
        .collect::<Option<Vec<_>>>()
        .ok_or(LagunaPagingError::ExpertPayloadOverflow {
            layer_index: layer_index(descriptor.tensor_id()),
        })
}

fn per_expert_slices(
    descriptor: &LagunaCanonicalTensorDescriptor,
    expert_capacity: usize,
    split_denominator: u64,
) -> Result<Vec<LagunaExpertSourceSlice>, LagunaPagingError> {
    if descriptor.sources().len() != expert_capacity {
        return Err(LagunaPagingError::MissingPerExpertSource {
            tensor_id: descriptor.tensor_id(),
            expert_index: descriptor.sources().len(),
        });
    }
    descriptor
        .sources()
        .iter()
        .map(|source| {
            if split_denominator == 0 || source.payload_bytes() % split_denominator != 0 {
                return Err(LagunaPagingError::FusedProjectionNotEven {
                    tensor_id: descriptor.tensor_id(),
                });
            }
            let bytes_per_expert = source.payload_bytes() / split_denominator;
            let source_byte_count = usize::try_from(bytes_per_expert).map_err(|_| {
                LagunaPagingError::ExpertPayloadOverflow {
                    layer_index: layer_index(descriptor.tensor_id()),
                }
            })?;
            Ok(LagunaExpertSourceSlice {
                source_file: PathBuf::from(source.shard_file_name()),
                source_file_offset: source.data_start_offset_bytes(),
                source_byte_count,
            })
        })
        .collect()
}

fn fused_stacked_slices(
    descriptor: &LagunaCanonicalTensorDescriptor,
    expert_capacity: usize,
    projection: LagunaExpertProjection,
) -> Result<Vec<LagunaExpertSourceSlice>, LagunaPagingError> {
    shift_up_half(
        stacked_expert_slices(descriptor, expert_capacity, 2)?,
        projection,
        descriptor.tensor_id(),
    )
}

fn fused_per_expert_slices(
    descriptor: &LagunaCanonicalTensorDescriptor,
    expert_capacity: usize,
    projection: LagunaExpertProjection,
) -> Result<Vec<LagunaExpertSourceSlice>, LagunaPagingError> {
    shift_up_half(
        per_expert_slices(descriptor, expert_capacity, 2)?,
        projection,
        descriptor.tensor_id(),
    )
}

fn shift_up_half(
    mut slices: Vec<LagunaExpertSourceSlice>,
    projection: LagunaExpertProjection,
    tensor_id: LagunaTensorId,
) -> Result<Vec<LagunaExpertSourceSlice>, LagunaPagingError> {
    if !matches!(projection, LagunaExpertProjection::Up) {
        return Ok(slices);
    }
    let first_slice = slices
        .first()
        .ok_or(LagunaPagingError::MissingSourceInterval { tensor_id })?;
    let half_bytes = u64::try_from(first_slice.source_byte_count).map_err(|_| {
        LagunaPagingError::ExpertPayloadOverflow {
            layer_index: layer_index(tensor_id),
        }
    })?;
    for slice in &mut slices {
        slice.source_file_offset = slice.source_file_offset.checked_add(half_bytes).ok_or(
            LagunaPagingError::ExpertPayloadOverflow {
                layer_index: layer_index(tensor_id),
            },
        )?;
    }
    Ok(slices)
}

fn layer_index(tensor_id: LagunaTensorId) -> usize {
    match tensor_id {
        LagunaTensorId::Layer { layer_index, .. } => layer_index,
        LagunaTensorId::Global { .. } => 0,
    }
}
