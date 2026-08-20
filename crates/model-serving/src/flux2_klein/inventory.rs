//! Physical transformer and VAE inventory contracts with retained source intervals.

use std::collections::{BTreeMap, BTreeSet};

use ::safetensors::Dtype;

use crate::artifact_validation::{RawSafetensorsInventory, RawSafetensorsTensorDescriptor};

use super::{Flux2KleinArtifactError, Flux2KleinTransformerConfig};

/// One exact physical tensor source interval for future native loading.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Flux2KleinTensorDescriptor {
    source_file_name: String,
    tensor_name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    data_start_offset_bytes: u64,
    data_end_offset_bytes: u64,
    payload_bytes: u64,
}

impl Flux2KleinTensorDescriptor {
    pub fn source_file_name(&self) -> &str {
        &self.source_file_name
    }
    pub fn tensor_name(&self) -> &str {
        &self.tensor_name
    }
    pub const fn dtype(&self) -> Dtype {
        self.dtype
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub const fn data_start_offset_bytes(&self) -> u64 {
        self.data_start_offset_bytes
    }
    pub const fn data_end_offset_bytes(&self) -> u64 {
        self.data_end_offset_bytes
    }
    pub const fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }

    /// Identifies tensors retained by the production decoder rather than sibling VAE owners.
    pub fn is_owned_by_vae_decoder(&self) -> bool {
        vae_decoder_owns_tensor(self.tensor_name())
    }
}

/// Validated inventory topology plus the exact descriptors needed to load it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Flux2KleinTensorInventory {
    pub(super) descriptors: Vec<Flux2KleinTensorDescriptor>,
    pub(super) payload_bytes: u64,
    pub(super) double_stream_block_count: usize,
    pub(super) single_stream_block_count: usize,
    pub(super) up_block_count: usize,
}

impl Flux2KleinTensorInventory {
    pub fn descriptors(&self) -> &[Flux2KleinTensorDescriptor] {
        &self.descriptors
    }
    pub const fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }
    pub const fn double_stream_block_count(&self) -> usize {
        self.double_stream_block_count
    }
    pub const fn single_stream_block_count(&self) -> usize {
        self.single_stream_block_count
    }
    pub const fn up_block_count(&self) -> usize {
        self.up_block_count
    }

    /// Returns the exact retained tensor payload constructed by the production VAE decoder.
    pub fn vae_decoder_owned_payload_bytes(&self) -> Option<u64> {
        self.descriptors
            .iter()
            .filter(|descriptor| descriptor.is_owned_by_vae_decoder())
            .try_fold(0_u64, |payload_bytes, descriptor| {
                payload_bytes.checked_add(descriptor.payload_bytes())
            })
    }

    /// Counts physical sources represented by the validated descriptor inventory.
    pub fn source_file_count(&self) -> usize {
        self.descriptors
            .iter()
            .map(|descriptor| descriptor.source_file_name())
            .collect::<BTreeSet<_>>()
            .len()
    }
}

pub(super) fn validate_transformer_inventory(
    file_name: &str,
    inventory: RawSafetensorsInventory,
    config: &Flux2KleinTransformerConfig,
) -> Result<Flux2KleinTensorInventory, Flux2KleinArtifactError> {
    let double_stream_block_count = config.double_stream_block_count();
    let single_stream_block_count = config.single_stream_block_count();
    let names = inventory
        .tensor_descriptors
        .iter()
        .map(|tensor| tensor.tensor_name.as_str())
        .collect::<BTreeSet<_>>();
    validate_exact_names(
        &names,
        &transformer_tensor_names(double_stream_block_count, single_stream_block_count),
        "transformer",
    )?;
    let expected_shapes = config.expected_weight_shapes().collect::<BTreeMap<_, _>>();
    let descriptors = validate_component_tensors(
        file_name,
        "transformer",
        inventory.tensor_descriptors,
        |tensor| expected_shapes.get(&tensor.tensor_name) == Some(&tensor.shape),
    )?;
    Ok(Flux2KleinTensorInventory {
        descriptors,
        payload_bytes: inventory.shard_payload_bytes,
        double_stream_block_count,
        single_stream_block_count,
        up_block_count: 0,
    })
}

pub(super) fn validate_vae_inventory(
    file_name: &str,
    inventory: RawSafetensorsInventory,
) -> Result<Flux2KleinTensorInventory, Flux2KleinArtifactError> {
    let names = inventory
        .tensor_descriptors
        .iter()
        .map(|tensor| tensor.tensor_name.as_str())
        .collect::<BTreeSet<_>>();
    validate_exact_names(&names, &vae_tensor_names(), "VAE")?;
    let descriptors = inventory
        .tensor_descriptors
        .into_iter()
        .map(|tensor| {
            let is_batch_counter = tensor.tensor_name == "bn.num_batches_tracked";
            let dtype_is_supported = if is_batch_counter {
                tensor.dtype == Dtype::I64 && tensor.shape.is_empty()
            } else {
                tensor.dtype == Dtype::BF16
            };
            if !dtype_is_supported {
                return Err(Flux2KleinArtifactError::TensorDtype {
                    component: "VAE",
                    tensor_name: tensor.tensor_name,
                });
            }
            if !vae_name_is_supported(&tensor.tensor_name) {
                return Err(Flux2KleinArtifactError::UnsupportedTensor {
                    component: "VAE",
                    tensor_name: tensor.tensor_name,
                });
            }
            Ok(public_descriptor(file_name, tensor))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Flux2KleinTensorInventory {
        descriptors,
        payload_bytes: inventory.shard_payload_bytes,
        double_stream_block_count: 0,
        single_stream_block_count: 0,
        up_block_count: 4,
    })
}

fn validate_component_tensors(
    file_name: &str,
    component: &'static str,
    tensors: Vec<RawSafetensorsTensorDescriptor>,
    shape_is_supported: impl Fn(&RawSafetensorsTensorDescriptor) -> bool,
) -> Result<Vec<Flux2KleinTensorDescriptor>, Flux2KleinArtifactError> {
    tensors
        .into_iter()
        .map(|tensor| {
            if tensor.dtype != Dtype::BF16 {
                return Err(Flux2KleinArtifactError::TensorDtype {
                    component,
                    tensor_name: tensor.tensor_name,
                });
            }
            if !shape_is_supported(&tensor) {
                return Err(Flux2KleinArtifactError::TensorShape {
                    component,
                    tensor_name: tensor.tensor_name,
                });
            }
            Ok(public_descriptor(file_name, tensor))
        })
        .collect()
}

fn vae_name_is_supported(name: &str) -> bool {
    name.starts_with("decoder.")
        || name.starts_with("encoder.")
        || name.starts_with("quant_conv.")
        || name.starts_with("post_quant_conv.")
        || matches!(
            name,
            "bn.num_batches_tracked" | "bn.running_mean" | "bn.running_var"
        )
}

fn vae_decoder_owns_tensor(name: &str) -> bool {
    name.starts_with("decoder.")
        || name.starts_with("post_quant_conv.")
        || matches!(name, "bn.running_mean" | "bn.running_var")
}

fn validate_exact_names(
    actual_names: &BTreeSet<&str>,
    expected_names: &BTreeSet<String>,
    component: &'static str,
) -> Result<(), Flux2KleinArtifactError> {
    if let Some(missing_name) = expected_names
        .iter()
        .find(|name| !actual_names.contains(name.as_str()))
    {
        return Err(Flux2KleinArtifactError::MissingTensor {
            component,
            tensor_name: missing_name.clone(),
        });
    }
    if let Some(extra_name) = actual_names
        .iter()
        .find(|name| !expected_names.contains(**name))
    {
        return Err(Flux2KleinArtifactError::UnsupportedTensor {
            component,
            tensor_name: (*extra_name).to_owned(),
        });
    }
    Ok(())
}

fn transformer_tensor_names(double_count: usize, single_count: usize) -> BTreeSet<String> {
    let mut names = [
        "context_embedder.weight",
        "double_stream_modulation_img.linear.weight",
        "double_stream_modulation_txt.linear.weight",
        "norm_out.linear.weight",
        "proj_out.weight",
        "single_stream_modulation.linear.weight",
        "time_guidance_embed.timestep_embedder.linear_1.weight",
        "time_guidance_embed.timestep_embedder.linear_2.weight",
        "x_embedder.weight",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<BTreeSet<_>>();
    let double_suffixes = [
        "attn.add_k_proj.weight",
        "attn.add_q_proj.weight",
        "attn.add_v_proj.weight",
        "attn.norm_added_k.weight",
        "attn.norm_added_q.weight",
        "attn.norm_k.weight",
        "attn.norm_q.weight",
        "attn.to_add_out.weight",
        "attn.to_k.weight",
        "attn.to_out.0.weight",
        "attn.to_q.weight",
        "attn.to_v.weight",
        "ff.linear_in.weight",
        "ff.linear_out.weight",
        "ff_context.linear_in.weight",
        "ff_context.linear_out.weight",
    ];
    for block_index in 0..double_count {
        for suffix in double_suffixes {
            names.insert(format!("transformer_blocks.{block_index}.{suffix}"));
        }
    }
    let single_suffixes = [
        "attn.norm_k.weight",
        "attn.norm_q.weight",
        "attn.to_out.weight",
        "attn.to_qkv_mlp_proj.weight",
    ];
    for block_index in 0..single_count {
        for suffix in single_suffixes {
            names.insert(format!("single_transformer_blocks.{block_index}.{suffix}"));
        }
    }
    names
}

fn vae_tensor_names() -> BTreeSet<String> {
    let mut names = [
        "decoder.conv_in",
        "decoder.conv_norm_out",
        "decoder.conv_out",
        "encoder.conv_in",
        "encoder.conv_norm_out",
        "encoder.conv_out",
        "quant_conv",
        "post_quant_conv",
    ]
    .into_iter()
    .flat_map(weight_and_bias_names)
    .collect::<BTreeSet<_>>();
    names.extend([
        "bn.num_batches_tracked".to_owned(),
        "bn.running_mean".to_owned(),
        "bn.running_var".to_owned(),
    ]);
    add_mid_block_names(&mut names, "decoder.mid_block");
    add_mid_block_names(&mut names, "encoder.mid_block");
    for block_index in 0..4 {
        for resnet_index in 0..3 {
            add_resnet_names(
                &mut names,
                &format!("decoder.up_blocks.{block_index}.resnets.{resnet_index}"),
                resnet_index == 0 && block_index >= 2,
            );
        }
        if block_index < 3 {
            add_weight_and_bias(
                &mut names,
                &format!("decoder.up_blocks.{block_index}.upsamplers.0.conv"),
            );
        }
    }
    for block_index in 0..4 {
        for resnet_index in 0..2 {
            add_resnet_names(
                &mut names,
                &format!("encoder.down_blocks.{block_index}.resnets.{resnet_index}"),
                resnet_index == 0 && matches!(block_index, 1 | 2),
            );
        }
        if block_index < 3 {
            add_weight_and_bias(
                &mut names,
                &format!("encoder.down_blocks.{block_index}.downsamplers.0.conv"),
            );
        }
    }
    names
}

fn weight_and_bias_names(prefix: &str) -> [String; 2] {
    [format!("{prefix}.weight"), format!("{prefix}.bias")]
}

fn add_weight_and_bias(names: &mut BTreeSet<String>, prefix: &str) {
    names.extend(weight_and_bias_names(prefix));
}

fn add_resnet_names(names: &mut BTreeSet<String>, prefix: &str, has_shortcut: bool) {
    for child in ["conv1", "conv2", "norm1", "norm2"] {
        add_weight_and_bias(names, &format!("{prefix}.{child}"));
    }
    if has_shortcut {
        add_weight_and_bias(names, &format!("{prefix}.conv_shortcut"));
    }
}

fn add_mid_block_names(names: &mut BTreeSet<String>, prefix: &str) {
    for resnet_index in 0..2 {
        add_resnet_names(names, &format!("{prefix}.resnets.{resnet_index}"), false);
    }
    let attention_prefix = format!("{prefix}.attentions.0");
    for child in ["group_norm", "to_k", "to_q", "to_v", "to_out.0"] {
        add_weight_and_bias(names, &format!("{attention_prefix}.{child}"));
    }
}

pub(super) fn public_descriptor(
    file_name: &str,
    tensor: RawSafetensorsTensorDescriptor,
) -> Flux2KleinTensorDescriptor {
    Flux2KleinTensorDescriptor {
        source_file_name: file_name.to_owned(),
        tensor_name: tensor.tensor_name,
        dtype: tensor.dtype,
        shape: tensor.shape,
        data_start_offset_bytes: tensor.data_start_offset_bytes,
        data_end_offset_bytes: tensor.data_end_offset_bytes,
        payload_bytes: tensor.tensor_payload_bytes,
    }
}
