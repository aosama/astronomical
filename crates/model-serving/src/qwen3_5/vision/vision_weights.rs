use std::collections::HashMap;

use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime, MlxSafetensors};

use crate::{TensorDtype, TensorProfile};

use super::{Qwen3_5ExecutionError, ValidatedQwen3_5Artifact, qwen3_5_vision_tensor_profiles};

/// Strict tensor binding for the Qwen3.5 vision tower.
///
/// For oQ4, vision tensors come from a separate sidecar file, and
/// `vision_sidecar` holds the owning `MlxSafetensors` to keep the
/// memory-mapped data alive.
///
/// For embedded vision, `Qwen3_5Weights::model_shards` keeps the source
/// maps alive instead.
/// In this case `vision_sidecar` is `None`.
#[derive(Debug)]
pub struct Qwen3_5VisionWeights {
    bound_tensors: HashMap<String, MlxArray>,
    /// The safetensors source that owns the memory-mapped data backing the
    /// bound tensors. Present for sidecar models (oQ4). `None` for embedded-vision
    /// models where the model shards in `Qwen3_5Weights` keep
    /// the data alive.
    #[allow(dead_code)]
    vision_sidecar: Option<MlxSafetensors>,
    total_payload_bytes: u64,
}

impl Qwen3_5VisionWeights {
    /// Loads vision weights from a separate sidecar file (oQ4 model).
    /// Returns None when visual weights are embedded or absent.
    pub fn load_from_sidecar(
        runtime: &MlxRuntime,
        validated_artifact: &mut ValidatedQwen3_5Artifact,
    ) -> Result<Option<Self>, Qwen3_5ExecutionError> {
        let vision_config =
            validated_artifact
                .vision_config()
                .ok_or(Qwen3_5ExecutionError::InvalidInput {
                    description: "validated visual sidecar has no vision configuration",
                })?;
        let vision_tensor_profiles = qwen3_5_vision_tensor_profiles(vision_config);
        let Some(vision_sidecar_file) = validated_artifact.take_vision_sidecar_file()? else {
            // The artifact either embeds visual weights or has none.
            return Ok(None);
        };
        let vision_sidecar = runtime.load_safetensors(vision_sidecar_file.into_file())?;

        let mut bound_tensors = HashMap::with_capacity(vision_tensor_profiles.len());
        let mut actual_payload_bytes = 0_u64;
        for tensor_profile in &vision_tensor_profiles {
            let tensor = vision_sidecar.tensor(&tensor_profile.name)?;
            validate_bound_vision_tensor(tensor_profile, &tensor)?;
            let tensor_payload_bytes = u64::try_from(tensor.byte_count()).map_err(|_| {
                Qwen3_5ExecutionError::InvalidTensor {
                    tensor_name: tensor_profile.name.clone(),
                    description: "vision tensor payload byte count exceeds the u64 range",
                }
            })?;
            actual_payload_bytes = actual_payload_bytes
                .checked_add(tensor_payload_bytes)
                .ok_or(Qwen3_5ExecutionError::InvalidTensor {
                    tensor_name: tensor_profile.name.clone(),
                    description: "vision tensor total payload byte count overflowed",
                })?;
            if bound_tensors
                .insert(tensor_profile.name.clone(), tensor)
                .is_some()
            {
                return Err(Qwen3_5ExecutionError::InvalidTensor {
                    tensor_name: tensor_profile.name.clone(),
                    description: "vision tensor was bound more than once",
                });
            }
        }

        Ok(Some(Self {
            bound_tensors,
            vision_sidecar: Some(vision_sidecar),
            total_payload_bytes: actual_payload_bytes,
        }))
    }

    /// Loads vision weights from model shard safetensors.
    ///
    /// Vision tensors can be embedded in the same shard files as language model
    /// tensors or a dedicated vision-only model shard. The `model_shards`
    /// parameter provides the already-loaded
    /// safetensors objects that keep the memory-mapped data alive.
    /// The `vision_tensor_name_to_shard_index` maps each vision tensor name
    /// to its shard index in `model_shards`.
    pub fn load_from_model_shards(
        vision_config: &super::Qwen3_5VisionConfig,
        model_shards: &[MlxSafetensors],
        vision_tensor_name_to_shard_index: &HashMap<String, usize>,
    ) -> Result<Self, Qwen3_5ExecutionError> {
        let vision_tensor_profiles = qwen3_5_vision_tensor_profiles(vision_config);
        let mut bound_tensors = HashMap::with_capacity(vision_tensor_profiles.len());
        let mut actual_payload_bytes = 0_u64;
        for tensor_profile in &vision_tensor_profiles {
            let shard_index = vision_tensor_name_to_shard_index
                .get(&tensor_profile.name)
                .ok_or_else(|| Qwen3_5ExecutionError::MissingTensor {
                    tensor_name: tensor_profile.name.clone(),
                })?;
            let model_shard = model_shards.get(*shard_index).ok_or_else(|| {
                Qwen3_5ExecutionError::InvalidTensor {
                    tensor_name: tensor_profile.name.clone(),
                    description: "vision tensor resolves outside the loaded model shards",
                }
            })?;
            let tensor = model_shard.tensor(&tensor_profile.name)?;
            validate_bound_vision_tensor(tensor_profile, &tensor)?;
            let tensor_payload_bytes = u64::try_from(tensor.byte_count()).map_err(|_| {
                Qwen3_5ExecutionError::InvalidTensor {
                    tensor_name: tensor_profile.name.clone(),
                    description: "vision tensor payload byte count exceeds the u64 range",
                }
            })?;
            actual_payload_bytes = actual_payload_bytes
                .checked_add(tensor_payload_bytes)
                .ok_or(Qwen3_5ExecutionError::InvalidTensor {
                    tensor_name: tensor_profile.name.clone(),
                    description: "vision tensor total payload byte count overflowed",
                })?;
            if bound_tensors
                .insert(tensor_profile.name.clone(), tensor)
                .is_some()
            {
                return Err(Qwen3_5ExecutionError::InvalidTensor {
                    tensor_name: tensor_profile.name.clone(),
                    description: "vision tensor was bound more than once",
                });
            }
        }

        Ok(Self {
            bound_tensors,
            vision_sidecar: None,
            total_payload_bytes: actual_payload_bytes,
        })
    }

    /// Returns one exact vision tower tensor by name.
    pub fn tensor(&self, tensor_name: &str) -> Result<&MlxArray, Qwen3_5ExecutionError> {
        self.bound_tensors
            .get(tensor_name)
            .ok_or_else(|| Qwen3_5ExecutionError::MissingTensor {
                tensor_name: tensor_name.to_owned(),
            })
    }

    /// Returns the number of exactly bound vision tensors.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.bound_tensors.len()
    }

    /// Returns the complete bound vision payload size without evaluating it.
    #[must_use]
    pub const fn total_payload_bytes(&self) -> u64 {
        self.total_payload_bytes
    }

    #[allow(dead_code)]
    pub(super) fn materialize(&self, runtime: &MlxRuntime) -> Result<(), Qwen3_5ExecutionError> {
        let bound_tensor_references = self.bound_tensors.values().collect::<Vec<_>>();
        Ok(runtime.evaluate_arrays(&bound_tensor_references)?)
    }
}

fn validate_bound_vision_tensor(
    tensor_profile: &TensorProfile,
    tensor: &MlxArray,
) -> Result<(), Qwen3_5ExecutionError> {
    let tensor_dtype_matches_profile = match tensor_profile.dtype {
        TensorDtype::AffineQuantizationFloat => matches!(
            tensor.dtype(),
            MlxDtype::Float16 | MlxDtype::BFloat16 | MlxDtype::Float32
        ),
        TensorDtype::ModelFloat => matches!(
            tensor.dtype(),
            MlxDtype::Float16 | MlxDtype::BFloat16 | MlxDtype::Float32
        ),
        TensorDtype::BFloat16 => tensor.dtype() == MlxDtype::BFloat16,
        TensorDtype::Float32 => tensor.dtype() == MlxDtype::Float32,
        TensorDtype::UInt32 => tensor.dtype() == MlxDtype::UInt32,
    };
    if !tensor_dtype_matches_profile {
        return Err(Qwen3_5ExecutionError::InvalidTensor {
            tensor_name: tensor_profile.name.clone(),
            description: "vision tensor dtype differs from the certified profile",
        });
    }
    let expected_shape = tensor_profile
        .shape
        .iter()
        .map(|dimension| {
            i32::try_from(*dimension).map_err(|_| Qwen3_5ExecutionError::InvalidTensor {
                tensor_name: tensor_profile.name.clone(),
                description: "vision tensor dimension exceeds the MLX integer range",
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if tensor.shape() != expected_shape {
        return Err(Qwen3_5ExecutionError::InvalidTensor {
            tensor_name: tensor_profile.name.clone(),
            description: "vision tensor shape differs from the certified profile",
        });
    }
    Ok(())
}
