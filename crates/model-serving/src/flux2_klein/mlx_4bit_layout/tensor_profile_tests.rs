use std::collections::BTreeMap;

use ::safetensors::Dtype;

use super::{ExpectedTensors, validate_component_tensors};
use crate::artifact_validation::{RawSafetensorsInventory, RawSafetensorsTensorDescriptor};
use crate::flux2_klein::artifact::Flux2KleinArtifactError;
use crate::flux2_klein::inventory::Flux2KleinTensorDescriptor;
use crate::flux2_klein::mlx_4bit_layout::shard_index::Mlx4BitShardIndex;
use crate::flux2_klein::weight_layout::Flux2KleinComponent;

const SHARD_FILE_NAME: &str = "0.safetensors";

fn stored(tensor_name: &str, dtype: Dtype, shape: &[usize]) -> RawSafetensorsTensorDescriptor {
    let value_count = shape.iter().product::<usize>() as u64;
    let bytes_per_value = match dtype {
        Dtype::U32 | Dtype::F32 => 4,
        _ => 2,
    };
    RawSafetensorsTensorDescriptor {
        tensor_name: tensor_name.to_owned(),
        dtype,
        shape: shape.to_vec(),
        data_start_offset_bytes: 0,
        data_end_offset_bytes: value_count * bytes_per_value,
        tensor_payload_bytes: value_count * bytes_per_value,
    }
}

/// A packed `[output_width, input_width]` projection named `stem`.
fn packed(
    stem: &str,
    output_width: usize,
    input_width: usize,
) -> Vec<RawSafetensorsTensorDescriptor> {
    vec![
        stored(
            &format!("{stem}.weight"),
            Dtype::U32,
            &[output_width, input_width / 8],
        ),
        stored(
            &format!("{stem}.scales"),
            Dtype::BF16,
            &[output_width, input_width / 64],
        ),
        stored(
            &format!("{stem}.biases"),
            Dtype::BF16,
            &[output_width, input_width / 64],
        ),
    ]
}

fn validate(
    component: Flux2KleinComponent,
    expected_tensors: &ExpectedTensors,
    stored_tensors: Vec<RawSafetensorsTensorDescriptor>,
) -> Result<Vec<Flux2KleinTensorDescriptor>, Flux2KleinArtifactError> {
    let weight_map = stored_tensors
        .iter()
        .map(|tensor| (tensor.tensor_name.clone(), SHARD_FILE_NAME.to_owned()))
        .collect::<BTreeMap<_, _>>();
    let index = Mlx4BitShardIndex {
        weight_map,
        shard_file_names: vec![SHARD_FILE_NAME.to_owned()],
    };
    let shard_inventory = RawSafetensorsInventory {
        tensor_descriptors: stored_tensors,
        shard_payload_bytes: 0,
    };
    validate_component_tensors(
        component,
        expected_tensors,
        &index,
        vec![(SHARD_FILE_NAME.to_owned(), shard_inventory)],
    )
}

fn expected(entries: &[(&str, Option<&[usize]>)]) -> ExpectedTensors {
    entries
        .iter()
        .map(|(name, shape)| ((*name).to_owned(), shape.map(<[usize]>::to_vec)))
        .collect()
}

#[test]
fn should_describe_a_quantized_transformer_projection_by_diffusers_name_and_packed_payload() {
    let descriptors = validate(
        Flux2KleinComponent::Transformer,
        &expected(&[(
            "transformer_blocks.0.attn.to_out.0.weight",
            Some(&[3_072, 3_072]),
        )]),
        packed("transformer_blocks.0.attn.to_out", 3_072, 3_072),
    )
    .expect("a correctly packed projection should validate");

    assert_eq!(descriptors.len(), 1);
    let projection = &descriptors[0];
    assert_eq!(
        projection.tensor_name(),
        "transformer_blocks.0.attn.to_out.0.weight"
    );
    assert_eq!(projection.shape(), [3_072, 3_072]);
    assert_eq!(projection.dtype(), Dtype::U32);
    let packed_bytes = 3_072 * 384 * 4 + 2 * 3_072 * 48 * 2;
    assert_eq!(projection.payload_bytes(), packed_bytes);
    assert_eq!(projection.source_file_name(), "transformer/0.safetensors");
}

#[test]
fn should_describe_a_quantized_text_projection_at_its_packed_size() {
    let descriptors = validate(
        Flux2KleinComponent::TextEncoder,
        &expected(&[("model.layers.0.mlp.down_proj.weight", None)]),
        packed("layers.0.mlp.down_proj", 2_560, 9_728),
    )
    .expect("a correctly packed text projection should validate");

    assert_eq!(descriptors[0].dtype(), Dtype::U32);
    assert_eq!(descriptors[0].shape(), [2_560, 9_728]);
    assert_eq!(
        descriptors[0].payload_bytes(),
        2_560 * 1_216 * 4 + 2 * 2_560 * 152 * 2
    );
}

#[test]
fn should_describe_a_quantized_vae_projection_at_its_dequantized_bf16_size() {
    let descriptors = validate(
        Flux2KleinComponent::Vae,
        &expected(&[("decoder.mid_block.attentions.0.to_q.weight", None)]),
        packed("decoder.mid_block.attentions.0.to_q", 512, 512),
    )
    .expect("a correctly packed VAE projection should validate");

    assert_eq!(descriptors[0].dtype(), Dtype::BF16);
    assert_eq!(descriptors[0].payload_bytes(), 512 * 512 * 2);
}

#[test]
fn should_describe_a_vae_convolution_kernel_in_the_diffusers_layout() {
    let descriptors = validate(
        Flux2KleinComponent::Vae,
        &expected(&[("decoder.conv_in.weight", None)]),
        vec![stored(
            "decoder.conv_in.weight",
            Dtype::BF16,
            &[512, 3, 3, 32],
        )],
    )
    .expect("an MLX-layout kernel should validate");

    assert_eq!(descriptors[0].shape(), [512, 32, 3, 3]);
}

#[test]
fn should_ignore_the_text_encoder_rotary_inverse_frequencies() {
    let descriptors = validate(
        Flux2KleinComponent::TextEncoder,
        &expected(&[("model.norm.weight", Some(&[2_560]))]),
        vec![
            stored("norm.weight", Dtype::BF16, &[2_560]),
            stored("rotary_emb.inv_freq", Dtype::F32, &[64]),
        ],
    )
    .expect("the stored inverse frequencies should be ignored");

    assert_eq!(descriptors.len(), 1);
}

#[test]
fn should_refuse_rotary_inverse_frequencies_outside_the_text_encoder() {
    let refusal = validate(
        Flux2KleinComponent::Vae,
        &expected(&[("bn.running_mean", None)]),
        vec![
            stored("bn.running_mean", Dtype::BF16, &[128]),
            stored("rotary_emb.inv_freq", Dtype::F32, &[64]),
        ],
    );

    assert!(matches!(
        refusal,
        Err(Flux2KleinArtifactError::UnsupportedTensor { .. })
    ));
}

#[test]
fn should_refuse_a_missing_diffusers_tensor() {
    let refusal = validate(
        Flux2KleinComponent::Transformer,
        &expected(&[("proj_out.weight", None), ("x_embedder.weight", None)]),
        vec![stored("proj_out.weight", Dtype::BF16, &[128, 3_072])],
    );

    assert!(matches!(
        refusal,
        Err(Flux2KleinArtifactError::MissingTensor { tensor_name, .. })
            if tensor_name == "x_embedder.weight"
    ));
}

#[test]
fn should_refuse_scales_without_biases() {
    let mut stored_tensors = packed("proj_out", 128, 3_072);
    stored_tensors.retain(|tensor| !tensor.tensor_name.ends_with(".biases"));

    let refusal = validate(
        Flux2KleinComponent::Transformer,
        &expected(&[("proj_out.weight", None)]),
        stored_tensors,
    );

    assert!(matches!(
        refusal,
        Err(Flux2KleinArtifactError::QuantizedTensorPacking { .. })
    ));
}

#[test]
fn should_refuse_a_group_size_other_than_64() {
    let stored_tensors = vec![
        stored("proj_out.weight", Dtype::U32, &[128, 384]),
        stored("proj_out.scales", Dtype::BF16, &[128, 96]),
        stored("proj_out.biases", Dtype::BF16, &[128, 96]),
    ];

    let refusal = validate(
        Flux2KleinComponent::Transformer,
        &expected(&[("proj_out.weight", None)]),
        stored_tensors,
    );

    assert!(matches!(
        refusal,
        Err(Flux2KleinArtifactError::QuantizedTensorPacking { .. })
    ));
}

#[test]
fn should_refuse_a_dense_tensor_that_is_not_bf16() {
    let refusal = validate(
        Flux2KleinComponent::Vae,
        &expected(&[("bn.running_var", None)]),
        vec![stored("bn.running_var", Dtype::F32, &[128])],
    );

    assert!(matches!(
        refusal,
        Err(Flux2KleinArtifactError::TensorDtype { .. })
    ));
}

#[test]
fn should_refuse_a_projection_whose_shape_disagrees_with_the_configuration() {
    let refusal = validate(
        Flux2KleinComponent::Transformer,
        &expected(&[("proj_out.weight", Some(&[128, 3_072]))]),
        packed("proj_out", 128, 2_048),
    );

    assert!(matches!(
        refusal,
        Err(Flux2KleinArtifactError::TensorShape { .. })
    ));
}

#[test]
fn should_refuse_a_tensor_no_diffusers_name_maps_to() {
    let mut stored_tensors = vec![stored("proj_out.weight", Dtype::BF16, &[128, 3_072])];
    stored_tensors.push(stored("lm_head.weight", Dtype::BF16, &[8, 8]));

    let refusal = validate(
        Flux2KleinComponent::Transformer,
        &expected(&[("proj_out.weight", None)]),
        stored_tensors,
    );

    assert!(matches!(
        refusal,
        Err(Flux2KleinArtifactError::UnsupportedTensor { tensor_name, .. })
            if tensor_name == "lm_head.weight"
    ));
}

#[test]
fn should_refuse_a_tensor_the_index_places_in_another_shard() {
    let index = Mlx4BitShardIndex {
        weight_map: BTreeMap::from([("proj_out.weight".to_owned(), "1.safetensors".to_owned())]),
        shard_file_names: vec!["1.safetensors".to_owned()],
    };
    let shard_inventory = RawSafetensorsInventory {
        tensor_descriptors: vec![stored("proj_out.weight", Dtype::BF16, &[128, 3_072])],
        shard_payload_bytes: 0,
    };

    let refusal = validate_component_tensors(
        Flux2KleinComponent::Transformer,
        &expected(&[("proj_out.weight", None)]),
        &index,
        vec![(SHARD_FILE_NAME.to_owned(), shard_inventory)],
    );

    assert!(matches!(
        refusal,
        Err(Flux2KleinArtifactError::ShardIndexDisagreement { .. })
    ));
}
