use std::collections::HashMap;

use astronomical_runtime_integration::MlxRuntime;

use crate::qwen3_5_core::configuration::Qwen3_5Config;
use crate::qwen3_5_core::quantizations::mlx_quantization::{
    ModuleQuantizationProfile, ModuleWeightCodec,
};
use crate::{TensorDtype, TensorProfile};

use super::vision_tensor_source::Qwen3_5VisionTensorSource;
use super::{Qwen3_5ExecutionError, ValidatedQwen3_5Artifact};
use mlx_bridge::{MlxArray, MlxDtype};

#[cfg(test)]
pub(crate) mod tests;

/// Where one canonical vision tensor physically lives among the tower's
/// source files (vision sidecars, or the model shards of an embedded tower),
/// including its stored spelling for unified-naming checkpoints
/// (`model.visual.*` shards read through canonical `vision_tower.*` names).
#[derive(Clone, Debug)]
pub struct VisionTensorLocation {
    /// Position of the owning file in the tower's source file list.
    pub source_file_position: usize,
    /// Tensor spelling to use for the physical read.
    pub stored_tensor_name: String,
}

/// Strict tensor binding for the Qwen3.5 vision tower.
///
/// The bound tensors are lazy until the first image request evaluates them.
/// After that they hold the tower's bytes, so [`Self::rebind_lazily`] swaps in
/// fresh lazy handles from the retained source files and lets the evaluated
/// ones go: the tower then costs memory only while an image is encoded.
#[derive(Debug)]
pub struct Qwen3_5VisionWeights {
    bound_tensors: HashMap<String, MlxArray>,
    quantization_profiles: HashMap<String, ModuleQuantizationProfile>,
    tensor_profiles: Vec<TensorProfile>,
    /// The tower's own source files. Its lazy maps are loaded apart from the
    /// language shards even for embedded towers, because a map entry keeps an
    /// evaluated tensor alive: sharing the language maps would pin the tower's
    /// bytes for the model's lifetime. The maps themselves are dropped right
    /// after binding; each bound handle keeps its reader alive on its own.
    tensor_source: Qwen3_5VisionTensorSource,
    total_payload_bytes: u64,
}

impl Qwen3_5VisionWeights {
    /// Loads vision weights from indexed separate vision files.
    /// Returns None when visual weights are embedded or absent.
    pub fn load_from_sidecar(
        runtime: &MlxRuntime,
        validated_artifact: &mut ValidatedQwen3_5Artifact,
    ) -> Result<Option<Self>, Qwen3_5ExecutionError> {
        // The config is cloned so the artifact's mutable source-handoff below
        // never conflicts with the bound-tensor loop's config reads.
        let vision_config = validated_artifact
            .vision_config()
            .ok_or(Qwen3_5ExecutionError::InvalidInput {
                description: "validated visual sidecar has no vision configuration",
            })?
            .clone();
        let qwen3_5_config = validated_artifact.config().clone();
        let vision_sidecar_file_names = validated_artifact
            .shard_index()
            .vision_sidecar_file_names()
            .to_vec();
        let vision_sidecar_source_ids =
            validated_artifact.source_ids_for_file_names(&vision_sidecar_file_names)?;
        let vision_sidecar_files =
            validated_artifact.take_safetensors_sources(&vision_sidecar_source_ids)?;
        if vision_sidecar_files.is_empty() {
            // The artifact either embeds visual weights or has none.
            return Ok(None);
        }
        let mut vision_tensor_name_to_location = HashMap::new();
        for (vision_tensor_name, vision_sidecar_file_name) in validated_artifact
            .shard_index()
            .vision_tensor_name_to_shard_file_name()
        {
            let Some(source_file_position) = vision_sidecar_file_names
                .iter()
                .position(|indexed_file_name| indexed_file_name == vision_sidecar_file_name)
            else {
                continue;
            };
            vision_tensor_name_to_location.insert(
                vision_tensor_name.clone(),
                VisionTensorLocation {
                    source_file_position,
                    stored_tensor_name: vision_tensor_name.clone(),
                },
            );
        }
        let tensor_source = Qwen3_5VisionTensorSource::new(
            vision_sidecar_files
                .into_iter()
                .map(crate::artifact_validation::ValidatedWeightsFile::into_file)
                .collect(),
            vision_tensor_name_to_location,
        );
        Self::bind(runtime, &vision_config, &qwen3_5_config, tensor_source).map(Some)
    }

    /// Loads vision weights embedded in the model shard files.
    ///
    /// `model_shard_files` are descriptors of the model shards in the order the
    /// locations' `source_file_position` values index. Vision tensors can be
    /// embedded in the same shard files as language model tensors or in a
    /// dedicated vision-only model shard.
    pub fn load_from_model_shard_files(
        runtime: &MlxRuntime,
        vision_config: &super::Qwen3_5VisionConfig,
        qwen3_5_config: &Qwen3_5Config,
        model_shard_files: Vec<std::fs::File>,
        vision_tensor_name_to_shard_location: HashMap<String, VisionTensorLocation>,
    ) -> Result<Self, Qwen3_5ExecutionError> {
        let tensor_source = Qwen3_5VisionTensorSource::keeping_referenced_files(
            model_shard_files,
            vision_tensor_name_to_shard_location,
        );
        Self::bind(runtime, vision_config, qwen3_5_config, tensor_source)
    }

    fn bind(
        runtime: &MlxRuntime,
        vision_config: &super::Qwen3_5VisionConfig,
        qwen3_5_config: &Qwen3_5Config,
        tensor_source: Qwen3_5VisionTensorSource,
    ) -> Result<Self, Qwen3_5ExecutionError> {
        let tensor_profiles =
            super::qwen3_5_vision_tensor_profiles_for_model(vision_config, qwen3_5_config)
                .map_err(|artifact_error| Qwen3_5ExecutionError::ExpertLayerPlan {
                    description: artifact_error.to_string(),
                })?;
        let quantization_profiles =
            super::vision_tensor_spec::qwen3_5_vision_linear_quantization_profiles(
                vision_config,
                qwen3_5_config,
            );
        let (bound_tensors, total_payload_bytes) =
            bind_vision_tensors(runtime, &tensor_profiles, &tensor_source)?;
        Ok(Self {
            bound_tensors,
            quantization_profiles,
            tensor_profiles,
            tensor_source,
            total_payload_bytes,
        })
    }

    /// Replaces every bound tensor with a fresh lazy handle read from the
    /// retained source files, releasing the evaluated bytes.
    ///
    /// The bytes are freed once no graph still references the previous
    /// handles, and results stay correct either way; the caller releases after
    /// the visual embeddings were evaluated, when nothing does. On error the
    /// previous handles stay bound.
    pub(super) fn rebind_lazily(
        &mut self,
        runtime: &MlxRuntime,
    ) -> Result<(), Qwen3_5ExecutionError> {
        let (bound_tensors, total_payload_bytes) =
            bind_vision_tensors(runtime, &self.tensor_profiles, &self.tensor_source)?;
        if total_payload_bytes != self.total_payload_bytes {
            return Err(Qwen3_5ExecutionError::TensorPayloadMismatch {
                actual_payload_bytes: total_payload_bytes,
                expected_payload_bytes: self.total_payload_bytes,
            });
        }
        self.bound_tensors = bound_tensors;
        Ok(())
    }

    /// Returns one exact vision tower tensor by name.
    pub fn tensor(&self, tensor_name: &str) -> Result<&MlxArray, Qwen3_5ExecutionError> {
        self.bound_tensors
            .get(tensor_name)
            .ok_or_else(|| Qwen3_5ExecutionError::MissingTensor {
                tensor_name: tensor_name.to_owned(),
            })
    }

    pub(crate) fn quantization_profile_for_module(
        &self,
        module_name: &str,
    ) -> Option<ModuleQuantizationProfile> {
        self.quantization_profiles.get(module_name).copied()
    }

    pub(crate) fn linear(
        &self,
        runtime: &MlxRuntime,
        input_states: &MlxArray,
        module_name: &str,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let module_quantization_profile = self.quantization_profile_for_module(module_name).ok_or(
            Qwen3_5ExecutionError::InvalidInput {
                description: "vision linear module is missing its quantization profile",
            },
        )?;
        let linear_bias = self.tensor(&format!("{module_name}.bias"))?;
        if module_quantization_profile.is_unquantized() {
            let linear_weight = self.tensor(&format!("{module_name}.weight"))?;
            let transposed_linear_weight = runtime.transpose_axes(linear_weight, &[1, 0])?;
            // Keep the fused accumulation path for native weights; separating
            // this into matmul plus add changes BF16 rounding in the vision tower.
            return Ok(runtime.addmm(
                linear_bias,
                input_states,
                &transposed_linear_weight,
                1.0,
                1.0,
            )?);
        }
        if !matches!(
            module_quantization_profile.weight_codec,
            ModuleWeightCodec::MlxAffineQuantized
        ) {
            return Err(Qwen3_5ExecutionError::InvalidInput {
                description: "vision linear module declares an unsupported weight codec",
            });
        }
        let quantization_group_size = i32::try_from(module_quantization_profile.group_size)
            .map_err(|_| Qwen3_5ExecutionError::InvalidInput {
                description: "vision affine group size exceeds the MLX integer range",
            })?;
        let quantization_bits = i32::try_from(module_quantization_profile.bits).map_err(|_| {
            Qwen3_5ExecutionError::InvalidInput {
                description: "vision affine bit width exceeds the MLX integer range",
            }
        })?;
        let packed_linear_weight = self.tensor(&format!("{module_name}.weight"))?;
        let quantization_scales = self.tensor(&format!("{module_name}.scales"))?;
        let quantization_biases = self.tensor(&format!("{module_name}.biases"))?;
        // The affine `biases` companion restores quantized weight offsets;
        // `.bias` remains the original learned linear bias and is added after qmm.
        let quantized_projection = runtime.quantized_matmul_affine(
            input_states,
            packed_linear_weight,
            quantization_scales,
            quantization_biases,
            true,
            quantization_group_size,
            quantization_bits,
        )?;
        Ok(runtime.add(&quantized_projection, linear_bias)?)
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

/// Binds every profiled vision tensor from fresh lazy maps of the source
/// files, returning the bound handles and their total payload size. The maps
/// are dropped on return; each bound handle keeps its reader alive.
fn bind_vision_tensors(
    runtime: &MlxRuntime,
    tensor_profiles: &[TensorProfile],
    tensor_source: &Qwen3_5VisionTensorSource,
) -> Result<(HashMap<String, MlxArray>, u64), Qwen3_5ExecutionError> {
    let source_safetensors = tensor_source.load_lazy_safetensors(runtime)?;
    let mut bound_tensors = HashMap::with_capacity(tensor_profiles.len());
    let mut total_payload_bytes = 0_u64;
    for tensor_profile in tensor_profiles {
        let tensor_location = tensor_source.location(&tensor_profile.name)?;
        let source_map = source_safetensors
            .get(tensor_location.source_file_position)
            .ok_or_else(|| Qwen3_5ExecutionError::InvalidTensor {
                tensor_name: tensor_profile.name.clone(),
                description: "vision tensor resolves outside the loaded vision source files",
            })?;
        let tensor = source_map.tensor(&tensor_location.stored_tensor_name)?;
        let tensor = normalize_published_patch_embed_layout(runtime, tensor_profile, tensor)?;
        validate_bound_vision_tensor(tensor_profile, &tensor)?;
        let tensor_payload_bytes = u64::try_from(tensor.byte_count()).map_err(|_| {
            Qwen3_5ExecutionError::InvalidTensor {
                tensor_name: tensor_profile.name.clone(),
                description: "vision tensor payload byte count exceeds the u64 range",
            }
        })?;
        total_payload_bytes = total_payload_bytes
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
    Ok((bound_tensors, total_payload_bytes))
}

/// Normalizes the patch-embed Conv3d weight into the MLX ODHWI execution
/// layout when a checkpoint published it in the upstream PyTorch order
/// `[output, in_channels, kernel_depth, kernel_height, kernel_width]`. The
/// two forms are pure axis permutations of the same stored values, so the
/// normalization changes no numerics; tensors already in the execution
/// layout (and any other shape, which the profile validation then judges)
/// pass through untouched.
fn normalize_published_patch_embed_layout(
    runtime: &MlxRuntime,
    tensor_profile: &TensorProfile,
    tensor: MlxArray,
) -> Result<MlxArray, Qwen3_5ExecutionError> {
    let is_patch_embed_weight = tensor_profile.name == "vision_tower.patch_embed.proj.weight";
    if !is_patch_embed_weight {
        return Ok(tensor);
    }
    let execution_shape: Vec<i32> = tensor_profile
        .shape
        .iter()
        .map(|dimension| *dimension as i32)
        .collect();
    if tensor.shape() == execution_shape {
        return Ok(tensor);
    }
    let published_matches_pytorch_layout =
        tensor_profile
            .equivalent_published_shapes
            .iter()
            .any(|equivalent_shape| {
                let equivalent_shape: Vec<i32> = equivalent_shape
                    .iter()
                    .map(|dimension| *dimension as i32)
                    .collect();
                tensor.shape() == equivalent_shape
            });
    if !published_matches_pytorch_layout {
        return Ok(tensor);
    }
    // [output, in, kT, kH, kW] -> [output, kT, kH, kW, in].
    Ok(runtime.transpose_axes(&tensor, &[0, 2, 3, 4, 1])?)
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
        TensorDtype::Float16 => tensor.dtype() == MlxDtype::Float16,
        TensorDtype::Float32 => tensor.dtype() == MlxDtype::Float32,
        TensorDtype::UInt32 => tensor.dtype() == MlxDtype::UInt32,
    };
    if !tensor_dtype_matches_profile {
        return Err(Qwen3_5ExecutionError::InvalidTensor {
            tensor_name: tensor_profile.name.clone(),
            description: "vision tensor dtype differs from the expected profile",
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
            description: "vision tensor shape differs from the expected profile",
        });
    }
    Ok(())
}
