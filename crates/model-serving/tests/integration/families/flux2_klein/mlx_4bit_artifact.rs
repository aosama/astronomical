//! Validation of the 4-bit MLX build (`flux2_klein::mlx_4bit_build::artifact`).

use std::fs;

use astronomical_model_serving::{
    Flux2KleinArtifactError, Flux2KleinArtifactProvenance, Flux2KleinArtifactValidator,
    Flux2KleinWeightLayout,
};

use super::mlx_4bit_support::SyntheticMlx4BitArtifact;

#[test]
fn should_validate_the_4bit_tree_into_diffusers_named_inventories() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    SyntheticMlx4BitArtifact::reviewed().write(model_directory.path());

    let artifact = Flux2KleinArtifactValidator::new()
        .validate(
            model_directory.path(),
            Flux2KleinArtifactProvenance::mlx_4bit(),
        )
        .expect("the synthetic 4-bit tree should validate");

    assert_eq!(
        artifact.weight_layout(),
        Flux2KleinWeightLayout::MlxAffine4Bit
    );
    let transformer_inventory = artifact.transformer_inventory();
    assert_eq!(transformer_inventory.descriptors().len(), 169);
    assert_eq!(transformer_inventory.double_stream_block_count(), 5);
    assert_eq!(transformer_inventory.single_stream_block_count(), 20);
    let transformer_names = transformer_inventory
        .descriptors()
        .iter()
        .map(|descriptor| descriptor.tensor_name())
        .collect::<Vec<_>>();
    assert!(transformer_names.contains(&"transformer_blocks.0.attn.to_out.0.weight"));
    assert!(transformer_names.contains(&"single_transformer_blocks.0.attn.to_out.weight"));
    assert!(transformer_names.contains(&"time_guidance_embed.timestep_embedder.linear_1.weight"));
    assert_eq!(artifact.vae_inventory().up_block_count(), 4);
    let decoder_conv_in = artifact
        .vae_inventory()
        .descriptors()
        .iter()
        .find(|descriptor| descriptor.tensor_name() == "decoder.conv_in.weight")
        .expect("the decoder input convolution should be described");
    assert_eq!(decoder_conv_in.shape(), [512, 32, 3, 3]);
    assert_eq!(
        artifact.text_encoder_inventory().descriptors().len(),
        2 + 36 * 11
    );
    assert!(
        artifact
            .text_encoder_inventory()
            .descriptors()
            .iter()
            .any(|descriptor| descriptor.tensor_name() == "model.embed_tokens.weight")
    );
    let retained_files = artifact
        .into_retained_files()
        .expect("descriptor identities should transfer");
    assert_eq!(retained_files.text_encoder_weights().shard_count(), 1);
    assert_eq!(retained_files.tokenizer_sidecars().len(), 3);
}

#[test]
fn should_refuse_the_4bit_tree_under_the_bf16_provenance() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    SyntheticMlx4BitArtifact::reviewed().write(model_directory.path());

    let refusal = Flux2KleinArtifactValidator::new()
        .validate(
            model_directory.path(),
            Flux2KleinArtifactProvenance::official(),
        )
        .expect_err("the BF16 identity must not claim the 4-bit files");

    assert!(matches!(
        refusal,
        Flux2KleinArtifactError::UnsupportedProvenance { .. }
    ));
}

#[test]
fn should_refuse_a_4bit_tree_without_its_tokenizer() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    SyntheticMlx4BitArtifact::reviewed().write(model_directory.path());
    fs::remove_file(model_directory.path().join("tokenizer/tokenizer.json"))
        .expect("the tokenizer should be removable");

    let refusal = Flux2KleinArtifactValidator::new()
        .validate(
            model_directory.path(),
            Flux2KleinArtifactProvenance::mlx_4bit(),
        )
        .expect_err("a partial download must be refused");

    assert!(matches!(
        refusal,
        Flux2KleinArtifactError::ArtifactFile { file_name } if file_name == "tokenizer/tokenizer.json"
    ));
}

#[test]
fn should_refuse_a_tree_quantized_with_another_scheme() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    SyntheticMlx4BitArtifact {
        quantization_level: "8",
    }
    .write(model_directory.path());

    let refusal = Flux2KleinArtifactValidator::new()
        .validate(
            model_directory.path(),
            Flux2KleinArtifactProvenance::mlx_4bit(),
        )
        .expect_err("an 8-bit tree must be refused");

    assert!(matches!(
        refusal,
        Flux2KleinArtifactError::UnsupportedQuantization { .. }
    ));
}

#[test]
fn should_refuse_a_shard_the_index_names_but_the_tree_lacks() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    SyntheticMlx4BitArtifact::reviewed().write(model_directory.path());
    fs::remove_file(model_directory.path().join("vae/0.safetensors"))
        .expect("the VAE shard should be removable");

    let refusal = Flux2KleinArtifactValidator::new()
        .validate(
            model_directory.path(),
            Flux2KleinArtifactProvenance::mlx_4bit(),
        )
        .expect_err("a missing shard must be refused");

    assert!(matches!(
        refusal,
        Flux2KleinArtifactError::ArtifactFile { file_name } if file_name == "vae/0.safetensors"
    ));
}
