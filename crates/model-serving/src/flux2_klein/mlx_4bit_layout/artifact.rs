//! Validation of the 4-bit build's files: three components of mflux shards,
//! each with its own index, plus the tokenizer.
//!
//! The build ships no Diffusers configuration and no license file. The
//! configuration is the embedded upstream one; the license is established by
//! provenance, which discovery derives from the model card, before this runs.

use std::path::Path;

use crate::artifact_validation::ValidatedWeightsFile;

use super::super::artifact::{
    Flux2KleinArtifactError, Flux2KleinValidatedComponent, Flux2KleinValidatedParts,
    open_named_files, open_required, open_weights, read_required_document,
};
use super::super::inventory::{
    Flux2KleinTensorDescriptor, transformer_tensor_names, vae_tensor_names,
};
use super::super::weight_layout::{
    Flux2KleinComponent, Flux2KleinComponentWeightFiles, Flux2KleinWeightLayout,
};
use super::super::{Flux2KleinTensorInventory, Flux2KleinTransformerConfig};
use super::shard_index::Mlx4BitShardIndex;
use super::tensor_profile::{ExpectedTensors, validate_component_tensors};
use super::upstream_configs::parse_upstream_configs;

const SHARD_INDEX_FILE_NAME: &str = "model.safetensors.index.json";
/// The tokenizer loads from `tokenizer.json` alone; the other two complete the
/// published tokenizer folder and are proven present so a partial download is
/// refused here rather than at the first request.
const TOKENIZER_SIDECAR_FILE_NAMES: [&str; 3] = [
    "tokenizer/chat_template.jinja",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
];
/// The BF16 build counts its batch-norm updates; mflux does not store the
/// counter, and the decoder never reads it.
const VAE_BATCH_COUNTER_TENSOR_NAME: &str = "bn.num_batches_tracked";
/// The embedded text encoder configuration pins 36 decoder layers.
const TEXT_ENCODER_LAYER_COUNT: usize = 36;
const TEXT_ENCODER_LAYER_TENSOR_SUFFIXES: [&str; 11] = [
    "input_layernorm.weight",
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "post_attention_layernorm.weight",
    "self_attn.k_norm.weight",
    "self_attn.k_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
];
const VAE_UP_BLOCK_COUNT: usize = 4;

pub(in crate::flux2_klein) fn validate_mlx_4bit_artifact(
    model_directory: &Path,
) -> Result<Flux2KleinValidatedParts, Flux2KleinArtifactError> {
    let configs = parse_upstream_configs()?;
    let tokenizer_sidecars = open_named_files(model_directory, &TOKENIZER_SIDECAR_FILE_NAMES)?;
    let text_encoder = validate_component(
        model_directory,
        Flux2KleinComponent::TextEncoder,
        &text_encoder_expected_tensors(),
    )?;
    let transformer = validate_component(
        model_directory,
        Flux2KleinComponent::Transformer,
        &transformer_expected_tensors(&configs.transformer),
    )?;
    let vae = validate_component(
        model_directory,
        Flux2KleinComponent::Vae,
        &vae_expected_tensors(),
    )?;
    Ok(Flux2KleinValidatedParts {
        transformer: with_block_counts(
            transformer,
            configs.transformer.double_stream_block_count(),
            configs.transformer.single_stream_block_count(),
            0,
        ),
        vae: with_block_counts(vae, 0, 0, VAE_UP_BLOCK_COUNT),
        configs,
        document_files: std::collections::BTreeMap::new(),
        tokenizer_sidecars,
        text_encoder,
    })
}

/// Opens one component's index and shards and proves its tensor profile.
fn validate_component(
    model_directory: &Path,
    component: Flux2KleinComponent,
    expected_tensors: &ExpectedTensors,
) -> Result<Flux2KleinValidatedComponent, Flux2KleinArtifactError> {
    let folder_name = component.folder_name();
    let index_file_name = format!("{folder_name}/{SHARD_INDEX_FILE_NAME}");
    let index_file = open_required(model_directory, &index_file_name)?;
    let index = Mlx4BitShardIndex::parse(
        &index_file_name,
        &read_required_document(&index_file, &index_file_name)?,
    )?;
    let mut shard_files = Vec::<ValidatedWeightsFile>::new();
    let mut shard_inventories = Vec::new();
    for shard_file_name in &index.shard_file_names {
        let (shard_file, shard_inventory) =
            open_weights(model_directory, &format!("{folder_name}/{shard_file_name}"))?;
        shard_files.push(shard_file);
        shard_inventories.push((shard_file_name.clone(), shard_inventory));
    }
    let descriptors =
        validate_component_tensors(component, expected_tensors, &index, shard_inventories)?;
    Ok(Flux2KleinValidatedComponent {
        weight_files: Flux2KleinComponentWeightFiles::new(
            Flux2KleinWeightLayout::MlxAffine4Bit,
            component,
            shard_files,
        ),
        inventory: logical_inventory(descriptors)?,
    })
}

/// An inventory whose payload is what the engine keeps resident (see
/// `tensor_profile`), so admission budgets the 4-bit build by its real cost.
fn logical_inventory(
    descriptors: Vec<Flux2KleinTensorDescriptor>,
) -> Result<Flux2KleinTensorInventory, Flux2KleinArtifactError> {
    let mut payload_bytes = 0_u64;
    for descriptor in &descriptors {
        payload_bytes = payload_bytes
            .checked_add(descriptor.payload_bytes())
            .ok_or(Flux2KleinArtifactError::PayloadAccountingOverflow)?;
    }
    Ok(Flux2KleinTensorInventory {
        descriptors,
        payload_bytes,
        double_stream_block_count: 0,
        single_stream_block_count: 0,
        up_block_count: 0,
    })
}

fn with_block_counts(
    mut component: Flux2KleinValidatedComponent,
    double_stream_block_count: usize,
    single_stream_block_count: usize,
    up_block_count: usize,
) -> Flux2KleinValidatedComponent {
    component.inventory.double_stream_block_count = double_stream_block_count;
    component.inventory.single_stream_block_count = single_stream_block_count;
    component.inventory.up_block_count = up_block_count;
    component
}

fn transformer_expected_tensors(config: &Flux2KleinTransformerConfig) -> ExpectedTensors {
    let mut expected_tensors = ExpectedTensors::new();
    for (tensor_name, logical_shape) in config.expected_weight_shapes() {
        expected_tensors.insert(tensor_name, Some(logical_shape));
    }
    // The shape profile covers every transformer tensor; the name list is the
    // same one the BF16 inventory requires, so the two stay in step.
    for tensor_name in transformer_tensor_names(
        config.double_stream_block_count(),
        config.single_stream_block_count(),
    ) {
        expected_tensors.entry(tensor_name).or_insert(None);
    }
    expected_tensors
}

/// Shapes are checked when the text encoder binds each tensor, as in the BF16 build.
fn text_encoder_expected_tensors() -> ExpectedTensors {
    let mut expected_tensors = ExpectedTensors::from([
        ("model.embed_tokens.weight".to_owned(), None),
        ("model.norm.weight".to_owned(), None),
    ]);
    for layer_index in 0..TEXT_ENCODER_LAYER_COUNT {
        for suffix in TEXT_ENCODER_LAYER_TENSOR_SUFFIXES {
            expected_tensors.insert(format!("model.layers.{layer_index}.{suffix}"), None);
        }
    }
    expected_tensors
}

/// Shapes are checked when the decoder binds each tensor, as in the BF16 build.
fn vae_expected_tensors() -> ExpectedTensors {
    let mut expected_tensors = ExpectedTensors::new();
    for tensor_name in vae_tensor_names() {
        if tensor_name != VAE_BATCH_COUNTER_TENSOR_NAME {
            expected_tensors.insert(tensor_name, None);
        }
    }
    expected_tensors
}
