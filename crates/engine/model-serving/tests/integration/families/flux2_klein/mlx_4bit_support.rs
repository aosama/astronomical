//! A synthetic 4-bit MLX FLUX.2 Klein tree: mflux names, packed projections,
//! one shard per component, real headers and sparse payloads.

use std::fs;
use std::path::Path;

use serde_json::{Map, json};

use super::support::{
    SyntheticTensor, transformer_tensors, typed_tensor, vae_tensors, write,
    write_sparse_safetensors,
};

const SHARD_FILE_NAME: &str = "0.safetensors";
/// Text shapes are bound and checked by the encoder at load, not by the
/// validator, so the smallest packable projection keeps the fixture small.
const SMALL_PROJECTION_WIDTH: usize = 64;
const TEXT_ENCODER_LAYER_COUNT: usize = 36;
const TEXT_ENCODER_PROJECTION_SUFFIXES: [&str; 7] = [
    "mlp.down_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "self_attn.k_proj",
    "self_attn.o_proj",
    "self_attn.q_proj",
    "self_attn.v_proj",
];
const TEXT_ENCODER_NORM_SUFFIXES: [&str; 4] = [
    "input_layernorm",
    "post_attention_layernorm",
    "self_attn.k_norm",
    "self_attn.q_norm",
];

pub(super) struct SyntheticMlx4BitArtifact {
    pub(super) quantization_level: &'static str,
}

impl SyntheticMlx4BitArtifact {
    pub(super) fn reviewed() -> Self {
        Self {
            quantization_level: "4",
        }
    }

    pub(super) fn write(&self, model_directory: &Path) {
        for (component_folder, tensors) in [
            ("transformer", transformer_tensors_4bit()),
            ("text_encoder", text_encoder_tensors_4bit()),
            ("vae", vae_tensors_4bit()),
        ] {
            fs::create_dir_all(model_directory.join(component_folder))
                .expect("the component folder should be created");
            write_sparse_safetensors(
                model_directory,
                &format!("{component_folder}/{SHARD_FILE_NAME}"),
                &tensors,
            );
            let weight_map = tensors
                .iter()
                .map(|tensor| (tensor.name.clone(), json!(SHARD_FILE_NAME)))
                .collect::<Map<_, _>>();
            let index = json!({
                "metadata": {
                    "quantization_level": self.quantization_level,
                    "quantization_group_size": "64",
                    "mflux_version": "0.17.5"
                },
                "weight_map": weight_map
            });
            write(
                model_directory,
                &format!("{component_folder}/model.safetensors.index.json"),
                &serde_json::to_vec(&index).expect("the index should serialize"),
            );
        }
        fs::create_dir_all(model_directory.join("tokenizer"))
            .expect("the tokenizer folder should be created");
        for tokenizer_file in [
            "chat_template.jinja",
            "tokenizer.json",
            "tokenizer_config.json",
        ] {
            write(
                model_directory,
                &format!("tokenizer/{tokenizer_file}"),
                b"{}\n",
            );
        }
    }
}

/// mflux's names for the transformer (see `mlx_4bit_build::tensor_names`).
fn mflux_transformer_name(diffusers_name: &str) -> String {
    let flattened = diffusers_name.replace(
        "time_guidance_embed.timestep_embedder.",
        "time_guidance_embed.",
    );
    flattened.replace(".to_out.0.", ".to_out.")
}

/// Projections become packed triples; norm scales stay dense BF16.
fn push_stored(stored_tensors: &mut Vec<SyntheticTensor>, name: String, shape: Vec<usize>) {
    let Some(stem) = name.strip_suffix(".weight") else {
        stored_tensors.push(typed_tensor(&name, "BF16", shape));
        return;
    };
    let &[output_width, input_width] = shape.as_slice() else {
        stored_tensors.push(typed_tensor(&name, "BF16", shape));
        return;
    };
    let packed_width = input_width / 8;
    let group_count = input_width / 64;
    stored_tensors.push(typed_tensor(&name, "U32", vec![output_width, packed_width]));
    for role in ["scales", "biases"] {
        stored_tensors.push(typed_tensor(
            &format!("{stem}.{role}"),
            "BF16",
            vec![output_width, group_count],
        ));
    }
}

fn transformer_tensors_4bit() -> Vec<SyntheticTensor> {
    let mut stored_tensors = Vec::new();
    for diffusers_tensor in transformer_tensors() {
        push_stored(
            &mut stored_tensors,
            mflux_transformer_name(&diffusers_tensor.name),
            diffusers_tensor.shape,
        );
    }
    stored_tensors
}

fn text_encoder_tensors_4bit() -> Vec<SyntheticTensor> {
    let projection_shape = vec![SMALL_PROJECTION_WIDTH, SMALL_PROJECTION_WIDTH];
    let mut stored_tensors = vec![
        typed_tensor("norm.weight", "BF16", vec![SMALL_PROJECTION_WIDTH]),
        typed_tensor("rotary_emb.inv_freq", "F32", vec![32]),
    ];
    push_stored(
        &mut stored_tensors,
        "embed_tokens.weight".to_owned(),
        projection_shape.clone(),
    );
    for layer_index in 0..TEXT_ENCODER_LAYER_COUNT {
        for suffix in TEXT_ENCODER_NORM_SUFFIXES {
            stored_tensors.push(typed_tensor(
                &format!("layers.{layer_index}.{suffix}.weight"),
                "BF16",
                vec![SMALL_PROJECTION_WIDTH],
            ));
        }
        for suffix in TEXT_ENCODER_PROJECTION_SUFFIXES {
            push_stored(
                &mut stored_tensors,
                format!("layers.{layer_index}.{suffix}.weight"),
                projection_shape.clone(),
            );
        }
    }
    stored_tensors
}

/// The VAE without its batch-norm counter, with packed attention projections
/// and one MLX-order kernel.
fn vae_tensors_4bit() -> Vec<SyntheticTensor> {
    let mut stored_tensors = Vec::new();
    for diffusers_tensor in vae_tensors() {
        let stored_name = diffusers_tensor.name.replace(".to_out.0.", ".to_out.");
        let is_attention_projection = stored_name.contains(".attentions.0.to_")
            && stored_name.ends_with(".weight")
            && !stored_name.contains("group_norm");
        if diffusers_tensor.name == "bn.num_batches_tracked" {
            continue;
        }
        if is_attention_projection {
            push_stored(
                &mut stored_tensors,
                stored_name,
                vec![SMALL_PROJECTION_WIDTH, SMALL_PROJECTION_WIDTH],
            );
        } else if stored_name == "decoder.conv_in.weight" {
            stored_tensors.push(typed_tensor(&stored_name, "BF16", vec![512, 3, 3, 32]));
        } else {
            stored_tensors.push(typed_tensor(&stored_name, "BF16", diffusers_tensor.shape));
        }
    }
    stored_tensors
}
