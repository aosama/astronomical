//! Exact global, resident-block, and operation-local block ownership of the
//! transformer's weights, in either artifact layout (BF16 or 4-bit MLX).

use std::collections::{BTreeMap, BTreeSet};

use astronomical_runtime_integration::MlxRuntime;

use crate::flux2_klein::tensor_store::{
    Flux2KleinTensorStore, Flux2KleinWeight, map_component_weight_files,
};
use crate::flux2_klein::weight_layout::Flux2KleinComponentWeightFiles;

use super::{Flux2KleinTransformerError, Flux2KleinTransformerGeometry};
use mlx_c_rust::{MlxArray, MlxDtype};

enum TransformerTensorSource {
    Descriptor {
        retained_store: Flux2KleinTensorStore,
        streaming_files: Flux2KleinComponentWeightFiles,
    },
    Injected(BTreeMap<String, MlxArray>),
}

impl std::fmt::Debug for TransformerTensorSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Descriptor { .. } => formatter.write_str("TransformerTensorSource::Descriptor"),
            Self::Injected(tensors) => formatter
                .debug_tuple("TransformerTensorSource::Injected")
                .field(&tensors.len())
                .finish(),
        }
    }
}

impl TransformerTensorSource {
    fn weight(
        &self,
        tensor_name: &str,
        logical_shape: &[usize],
    ) -> Result<Flux2KleinWeight, Flux2KleinTransformerError> {
        match self {
            Self::Descriptor { retained_store, .. } => {
                Ok(retained_store.weight(tensor_name, logical_shape)?)
            }
            Self::Injected(tensors) => Ok(Flux2KleinWeight::Dense(
                tensors
                    .get(tensor_name)
                    .ok_or_else(|| Flux2KleinTransformerError::MissingWeight {
                        tensor_name: tensor_name.to_owned(),
                    })?
                    .retain()
                    .map_err(Flux2KleinTransformerError::from)?,
            )),
        }
    }

    fn is_descriptor_backed(&self) -> bool {
        matches!(self, Self::Descriptor { .. })
    }

    /// Maps one block's weights afresh from the files for a single operation,
    /// so a block that is not retained costs memory only while it runs. The
    /// mapping clones file descriptors once per operation, inside
    /// `map_component_weight_files`, and nowhere else.
    fn operation_local_weights(
        &self,
        runtime: &MlxRuntime,
        block_weight_shapes: &[(String, Vec<usize>)],
    ) -> Result<BTreeMap<String, Flux2KleinWeight>, Flux2KleinTransformerError> {
        let operation_store = match self {
            Self::Descriptor {
                streaming_files, ..
            } => Some(map_component_weight_files(streaming_files, runtime, None)?),
            Self::Injected(_) => None,
        };
        let mut block_weights = BTreeMap::new();
        for (tensor_name, logical_shape) in block_weight_shapes {
            let block_weight = match &operation_store {
                Some(store) => store.weight(tensor_name, logical_shape)?,
                None => self.weight(tensor_name, logical_shape)?,
            };
            block_weights.insert(tensor_name.clone(), block_weight);
        }
        Ok(block_weights)
    }
}

enum BlockTensorOwner<'a> {
    Retained(&'a BTreeMap<String, Flux2KleinWeight>),
    OperationLocal(BTreeMap<String, Flux2KleinWeight>),
}

pub(super) struct Flux2KleinBlockWeights<'a> {
    tensors: BlockTensorOwner<'a>,
}

impl Flux2KleinBlockWeights<'_> {
    /// A dense tensor of this block, such as a norm scale.
    pub(super) fn tensor(
        &self,
        tensor_name: &str,
    ) -> Result<&MlxArray, Flux2KleinTransformerError> {
        dense_weight_of_lookup(tensor_name, self.linear(tensor_name))
    }

    /// A projection of this block, dense or 4-bit.
    pub(super) fn linear(
        &self,
        tensor_name: &str,
    ) -> Result<&Flux2KleinWeight, Flux2KleinTransformerError> {
        let weights = match &self.tensors {
            BlockTensorOwner::Retained(weights) => *weights,
            BlockTensorOwner::OperationLocal(weights) => weights,
        };
        weights
            .get(tensor_name)
            .ok_or_else(|| Flux2KleinTransformerError::MissingWeight {
                tensor_name: tensor_name.to_owned(),
            })
    }

    pub(super) fn is_operation_local(&self) -> bool {
        matches!(&self.tensors, BlockTensorOwner::OperationLocal(_))
    }
}

#[derive(Debug)]
pub struct Flux2KleinTransformerWeights {
    global_tensors: BTreeMap<String, Flux2KleinWeight>,
    retained_blocks: BTreeMap<usize, BTreeMap<String, Flux2KleinWeight>>,
    block_weight_shapes: Vec<Vec<(String, Vec<usize>)>>,
    source: TransformerTensorSource,
    payload_bytes: u64,
    resident_payload_bytes: u64,
    tensor_count: usize,
}

impl Flux2KleinTransformerWeights {
    pub fn load(
        runtime: &MlxRuntime,
        weight_files: Flux2KleinComponentWeightFiles,
        geometry: &Flux2KleinTransformerGeometry,
    ) -> Result<Self, Flux2KleinTransformerError> {
        let retained_block_indices = (0..geometry.total_block_count()).collect::<Vec<_>>();
        Self::load_with_residency(runtime, weight_files, geometry, &retained_block_indices)
    }

    pub fn load_with_residency(
        runtime: &MlxRuntime,
        weight_files: Flux2KleinComponentWeightFiles,
        geometry: &Flux2KleinTransformerGeometry,
        retained_block_indices: &[usize],
    ) -> Result<Self, Flux2KleinTransformerError> {
        let source = TransformerTensorSource::Descriptor {
            retained_store: map_component_weight_files(&weight_files, runtime, None)?,
            streaming_files: weight_files,
        };
        Self::bind(source, geometry, retained_block_indices)
    }

    /// Test seam for reduced geometry; production uses descriptor-backed `load`.
    pub fn bind_injected(
        tensors: BTreeMap<String, MlxArray>,
        geometry: &Flux2KleinTransformerGeometry,
    ) -> Result<Self, Flux2KleinTransformerError> {
        let retained_block_indices = (0..geometry.total_block_count()).collect::<Vec<_>>();
        Self::bind_injected_with_residency(tensors, geometry, &retained_block_indices)
    }

    /// Reduced-geometry seam that preserves the production residency topology.
    pub fn bind_injected_with_residency(
        tensors: BTreeMap<String, MlxArray>,
        geometry: &Flux2KleinTransformerGeometry,
        retained_block_indices: &[usize],
    ) -> Result<Self, Flux2KleinTransformerError> {
        let expected_tensor_names = geometry
            .expected_weight_shapes()
            .map(|(tensor_name, _)| tensor_name)
            .collect::<BTreeSet<_>>();
        if let Some(tensor_name) = tensors
            .keys()
            .find(|tensor_name| !expected_tensor_names.contains(*tensor_name))
        {
            return Err(Flux2KleinTransformerError::UnassignedWeight {
                tensor_name: tensor_name.clone(),
            });
        }
        Self::bind(
            TransformerTensorSource::Injected(tensors),
            geometry,
            retained_block_indices,
        )
    }

    fn bind(
        source: TransformerTensorSource,
        geometry: &Flux2KleinTransformerGeometry,
        retained_block_indices: &[usize],
    ) -> Result<Self, Flux2KleinTransformerError> {
        let retained_block_indices =
            validate_retained_block_indices(retained_block_indices, geometry.total_block_count())?;
        let mut global_tensors = BTreeMap::new();
        let mut retained_blocks = BTreeMap::new();
        let mut block_weight_shapes = vec![Vec::new(); geometry.total_block_count()];
        let mut payload_bytes = 0_u64;
        let mut resident_payload_bytes = 0_u64;
        let mut tensor_count = 0_usize;

        for (tensor_name, expected_shape) in geometry.expected_weight_shapes() {
            let tensor = source.weight(&tensor_name, &expected_shape)?;
            validate_weight(&tensor_name, &tensor, &expected_shape)?;
            let tensor_payload_bytes = weight_payload_bytes(&tensor)?;
            payload_bytes = payload_bytes.checked_add(tensor_payload_bytes).ok_or(
                Flux2KleinTransformerError::InvalidInput {
                    description: "transformer payload accounting overflowed",
                },
            )?;
            tensor_count = tensor_count.saturating_add(1);

            if let Some(block_index) = geometry.block_index_for_weight_name(&tensor_name) {
                block_weight_shapes[block_index].push((tensor_name.clone(), expected_shape));
                if retained_block_indices.contains(&block_index) {
                    retained_blocks
                        .entry(block_index)
                        .or_insert_with(BTreeMap::new)
                        .insert(tensor_name, tensor);
                    resident_payload_bytes = resident_payload_bytes
                        .checked_add(tensor_payload_bytes)
                        .ok_or(Flux2KleinTransformerError::InvalidInput {
                            description: "resident transformer payload accounting overflowed",
                        })?;
                }
            } else {
                global_tensors.insert(tensor_name, tensor);
                resident_payload_bytes = resident_payload_bytes
                    .checked_add(tensor_payload_bytes)
                    .ok_or(Flux2KleinTransformerError::InvalidInput {
                    description: "resident transformer payload accounting overflowed",
                })?;
            }
        }

        Ok(Self {
            global_tensors,
            retained_blocks,
            block_weight_shapes,
            source,
            payload_bytes,
            resident_payload_bytes,
            tensor_count,
        })
    }

    /// A dense global tensor, such as a norm scale.
    pub fn tensor(&self, tensor_name: &str) -> Result<&MlxArray, Flux2KleinTransformerError> {
        dense_weight_of_lookup(tensor_name, self.linear(tensor_name))
    }

    /// A global projection, dense or 4-bit.
    pub(crate) fn linear(
        &self,
        tensor_name: &str,
    ) -> Result<&Flux2KleinWeight, Flux2KleinTransformerError> {
        self.global_tensors.get(tensor_name).ok_or_else(|| {
            Flux2KleinTransformerError::MissingWeight {
                tensor_name: tensor_name.to_owned(),
            }
        })
    }

    pub(super) fn bind_block(
        &self,
        runtime: &MlxRuntime,
        block_index: usize,
    ) -> Result<Flux2KleinBlockWeights<'_>, Flux2KleinTransformerError> {
        if let Some(tensors) = self.retained_blocks.get(&block_index) {
            return Ok(Flux2KleinBlockWeights {
                tensors: BlockTensorOwner::Retained(tensors),
            });
        }
        let block_weight_shapes = self.block_weight_shapes.get(block_index).ok_or(
            Flux2KleinTransformerError::InvalidInput {
                description: "transformer block index exceeds geometry",
            },
        )?;
        let tensors = self
            .source
            .operation_local_weights(runtime, block_weight_shapes)?;
        Ok(Flux2KleinBlockWeights {
            tensors: BlockTensorOwner::OperationLocal(tensors),
        })
    }

    pub fn materialize_owned(
        &self,
        runtime: &MlxRuntime,
    ) -> Result<(), Flux2KleinTransformerError> {
        let mut tensors = self
            .global_tensors
            .values()
            .flat_map(Flux2KleinWeight::arrays)
            .collect::<Vec<_>>();
        tensors.extend(
            self.retained_blocks
                .values()
                .flat_map(|block_tensors| block_tensors.values())
                .flat_map(Flux2KleinWeight::arrays),
        );
        runtime.evaluate_arrays(&tensors)?;
        Ok(())
    }

    pub const fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }
    pub const fn resident_payload_bytes(&self) -> u64 {
        self.resident_payload_bytes
    }
    pub const fn tensor_count(&self) -> usize {
        self.tensor_count
    }
    pub fn resident_tensor_count(&self) -> usize {
        self.global_tensors.len()
            + self
                .retained_blocks
                .values()
                .map(BTreeMap::len)
                .sum::<usize>()
    }
    pub fn retained_block_count(&self) -> usize {
        self.retained_blocks.len()
    }
    pub fn is_descriptor_backed(&self) -> bool {
        self.source.is_descriptor_backed()
    }
}

fn validate_retained_block_indices(
    retained_block_indices: &[usize],
    total_block_count: usize,
) -> Result<BTreeSet<usize>, Flux2KleinTransformerError> {
    let retained_indices = retained_block_indices
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if retained_indices.len() != retained_block_indices.len()
        || retained_indices
            .last()
            .is_some_and(|block_index| *block_index >= total_block_count)
    {
        return Err(Flux2KleinTransformerError::InvalidInput {
            description: "retained transformer block indices are duplicated or out of range",
        });
    }
    Ok(retained_indices)
}

/// A dense weight must be BF16 with exactly its configured shape; a 4-bit
/// projection was already checked against its logical shape when bound.
fn validate_weight(
    tensor_name: &str,
    weight: &Flux2KleinWeight,
    expected_shape: &[usize],
) -> Result<(), Flux2KleinTransformerError> {
    let Flux2KleinWeight::Dense(tensor) = weight else {
        return Ok(());
    };
    if tensor.dtype() != MlxDtype::BFloat16 {
        return Err(Flux2KleinTransformerError::WeightDtype {
            tensor_name: tensor_name.to_owned(),
        });
    }
    let expected_signed_shape = expected_shape
        .iter()
        .map(|dimension| i32::try_from(*dimension))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| Flux2KleinTransformerError::InvalidInput {
            description: "weight shape exceeds the MLX integer range",
        })?;
    if tensor.shape() != expected_signed_shape {
        return Err(Flux2KleinTransformerError::WeightShape {
            tensor_name: tensor_name.to_owned(),
            actual_shape: tensor.shape(),
            expected_shape: expected_shape.to_vec(),
        });
    }
    Ok(())
}

/// The bytes a weight occupies as stored: all three packed arrays of a 4-bit
/// projection, so memory accounting sees what is actually resident.
fn weight_payload_bytes(weight: &Flux2KleinWeight) -> Result<u64, Flux2KleinTransformerError> {
    let mut payload_bytes = 0_u64;
    for array in weight.arrays() {
        let array_bytes = u64::try_from(array.byte_count()).map_err(|_| {
            Flux2KleinTransformerError::InvalidInput {
                description: "transformer tensor payload exceeds the u64 range",
            }
        })?;
        payload_bytes = payload_bytes.checked_add(array_bytes).ok_or(
            Flux2KleinTransformerError::InvalidInput {
                description: "transformer payload accounting overflowed",
            },
        )?;
    }
    Ok(payload_bytes)
}

/// The dense array behind a looked-up weight; a missing weight stays missing.
fn dense_weight_of_lookup<'weight>(
    tensor_name: &str,
    looked_up_weight: Result<&'weight Flux2KleinWeight, Flux2KleinTransformerError>,
) -> Result<&'weight MlxArray, Flux2KleinTransformerError> {
    match looked_up_weight {
        Ok(weight) => dense_weight(tensor_name, weight),
        Err(missing_weight) => Err(missing_weight),
    }
}

/// The dense array behind `tensor_name`; a norm scale is never quantized, so a
/// 4-bit weight here means the artifact does not match the engine.
fn dense_weight<'weight>(
    tensor_name: &str,
    weight: &'weight Flux2KleinWeight,
) -> Result<&'weight MlxArray, Flux2KleinTransformerError> {
    weight
        .dense()
        .ok_or_else(|| Flux2KleinTransformerError::WeightDtype {
            tensor_name: tensor_name.to_owned(),
        })
}
