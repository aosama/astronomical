//! Exact BF16 global, resident-block, and operation-local block ownership.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;

use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime, MlxSafetensors};

use crate::ValidatedWeightsFile;

use super::{Flux2KleinTransformerError, Flux2KleinTransformerGeometry};

#[derive(Debug)]
enum TransformerTensorSource {
    Descriptor {
        retained_map: MlxSafetensors,
        streaming_file: File,
    },
    Injected(BTreeMap<String, MlxArray>),
}

impl TransformerTensorSource {
    fn tensor(&self, tensor_name: &str) -> Result<MlxArray, Flux2KleinTransformerError> {
        match self {
            Self::Descriptor { retained_map, .. } => Ok(retained_map.tensor(tensor_name)?),
            Self::Injected(tensors) => tensors
                .get(tensor_name)
                .ok_or_else(|| Flux2KleinTransformerError::MissingWeight {
                    tensor_name: tensor_name.to_owned(),
                })?
                .retain()
                .map_err(Flux2KleinTransformerError::from),
        }
    }

    fn is_descriptor_backed(&self) -> bool {
        matches!(self, Self::Descriptor { .. })
    }

    fn operation_local_tensors(
        &self,
        runtime: &MlxRuntime,
        tensor_names: &[String],
    ) -> Result<BTreeMap<String, MlxArray>, Flux2KleinTransformerError> {
        match self {
            Self::Descriptor { streaming_file, .. } => {
                let block_file = streaming_file
                    .try_clone()
                    .map_err(|source| Flux2KleinTransformerError::DescriptorClone { source })?;
                let block_map = runtime.load_safetensors(block_file, None)?;
                tensor_names
                    .iter()
                    .map(|tensor_name| {
                        block_map
                            .tensor(tensor_name)
                            .map(|tensor| (tensor_name.clone(), tensor))
                            .map_err(Flux2KleinTransformerError::from)
                    })
                    .collect()
            }
            Self::Injected(_) => tensor_names
                .iter()
                .map(|tensor_name| {
                    self.tensor(tensor_name)
                        .map(|tensor| (tensor_name.clone(), tensor))
                })
                .collect(),
        }
    }
}

enum BlockTensorOwner<'a> {
    Retained(&'a BTreeMap<String, MlxArray>),
    OperationLocal(BTreeMap<String, MlxArray>),
}

pub(super) struct Flux2KleinBlockWeights<'a> {
    tensors: BlockTensorOwner<'a>,
}

impl Flux2KleinBlockWeights<'_> {
    pub(super) fn tensor(
        &self,
        tensor_name: &str,
    ) -> Result<&MlxArray, Flux2KleinTransformerError> {
        let tensors = match &self.tensors {
            BlockTensorOwner::Retained(tensors) => *tensors,
            BlockTensorOwner::OperationLocal(tensors) => tensors,
        };
        tensors
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
    global_tensors: BTreeMap<String, MlxArray>,
    retained_blocks: BTreeMap<usize, BTreeMap<String, MlxArray>>,
    block_tensor_names: Vec<Vec<String>>,
    source: TransformerTensorSource,
    payload_bytes: u64,
    resident_payload_bytes: u64,
    tensor_count: usize,
}

impl Flux2KleinTransformerWeights {
    pub fn load(
        runtime: &MlxRuntime,
        weights_file: ValidatedWeightsFile,
        geometry: &Flux2KleinTransformerGeometry,
    ) -> Result<Self, Flux2KleinTransformerError> {
        let retained_block_indices = (0..geometry.total_block_count()).collect::<Vec<_>>();
        Self::load_with_residency(runtime, weights_file, geometry, &retained_block_indices)
    }

    pub fn load_with_residency(
        runtime: &MlxRuntime,
        weights_file: ValidatedWeightsFile,
        geometry: &Flux2KleinTransformerGeometry,
        retained_block_indices: &[usize],
    ) -> Result<Self, Flux2KleinTransformerError> {
        let weights_file = weights_file.into_file();
        let streaming_file = weights_file
            .try_clone()
            .map_err(|source| Flux2KleinTransformerError::DescriptorClone { source })?;
        let source = TransformerTensorSource::Descriptor {
            retained_map: runtime.load_safetensors(weights_file, None)?,
            streaming_file,
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
        let mut block_tensor_names = vec![Vec::new(); geometry.total_block_count()];
        let mut payload_bytes = 0_u64;
        let mut resident_payload_bytes = 0_u64;
        let mut tensor_count = 0_usize;

        for (tensor_name, expected_shape) in geometry.expected_weight_shapes() {
            let tensor = source.tensor(&tensor_name)?;
            validate_tensor(&tensor_name, &tensor, expected_shape)?;
            let tensor_payload_bytes = u64::try_from(tensor.byte_count()).map_err(|_| {
                Flux2KleinTransformerError::InvalidInput {
                    description: "transformer tensor payload exceeds the u64 range",
                }
            })?;
            payload_bytes = payload_bytes.checked_add(tensor_payload_bytes).ok_or(
                Flux2KleinTransformerError::InvalidInput {
                    description: "transformer payload accounting overflowed",
                },
            )?;
            tensor_count = tensor_count.saturating_add(1);

            if let Some(block_index) = geometry.block_index_for_weight_name(&tensor_name) {
                block_tensor_names[block_index].push(tensor_name.clone());
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
            block_tensor_names,
            source,
            payload_bytes,
            resident_payload_bytes,
            tensor_count,
        })
    }

    pub fn tensor(&self, tensor_name: &str) -> Result<&MlxArray, Flux2KleinTransformerError> {
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
        let tensor_names = self.block_tensor_names.get(block_index).ok_or(
            Flux2KleinTransformerError::InvalidInput {
                description: "transformer block index exceeds geometry",
            },
        )?;
        let tensors = self.source.operation_local_tensors(runtime, tensor_names)?;
        Ok(Flux2KleinBlockWeights {
            tensors: BlockTensorOwner::OperationLocal(tensors),
        })
    }

    pub fn materialize_owned(
        &self,
        runtime: &MlxRuntime,
    ) -> Result<(), Flux2KleinTransformerError> {
        let mut tensors = self.global_tensors.values().collect::<Vec<_>>();
        tensors.extend(
            self.retained_blocks
                .values()
                .flat_map(|block_tensors| block_tensors.values()),
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

fn validate_tensor(
    tensor_name: &str,
    tensor: &MlxArray,
    expected_shape: Vec<usize>,
) -> Result<(), Flux2KleinTransformerError> {
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
            expected_shape,
        });
    }
    Ok(())
}
