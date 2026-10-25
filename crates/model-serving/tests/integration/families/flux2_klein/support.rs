use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

use astronomical_model_serving::Flux2KleinTransformerConfig;
use astronomical_test_harness::fixtures::{
    model_index_json, scheduler_config_json, text_encoder_config_json, transformer_config_json,
    vae_config_json,
};
use serde_json::{Value, json};

pub(super) struct SyntheticFlux2KleinArtifact {
    text_tensors_by_shard: BTreeMap<String, Vec<SyntheticTensor>>,
    text_index: BTreeMap<String, String>,
    pub(super) root_transformer_bytes: Vec<u8>,
    invalid_transformer_shape: bool,
    invalid_transformer_dtype: bool,
}

#[derive(Clone)]
struct SyntheticTensor {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
}

impl SyntheticFlux2KleinArtifact {
    pub(super) fn official() -> Self {
        let first_shard = "text_encoder/model-00001-of-00002.safetensors".to_owned();
        let second_shard = "text_encoder/model-00002-of-00002.safetensors".to_owned();
        let first_tensor = SyntheticTensor {
            name: "model.embed_tokens.weight".to_owned(),
            dtype: "BF16",
            shape: vec![4, 4],
        };
        let second_tensor = SyntheticTensor {
            name: "model.layers.27.input_layernorm.weight".to_owned(),
            dtype: "BF16",
            shape: vec![4],
        };
        Self {
            text_tensors_by_shard: BTreeMap::from([
                (first_shard.clone(), vec![first_tensor.clone()]),
                (second_shard.clone(), vec![second_tensor.clone()]),
            ]),
            text_index: BTreeMap::from([
                (
                    first_tensor.name,
                    "model-00001-of-00002.safetensors".to_owned(),
                ),
                (
                    second_tensor.name,
                    "model-00002-of-00002.safetensors".to_owned(),
                ),
            ]),
            root_transformer_bytes: b"duplicate root payload".to_vec(),
            invalid_transformer_shape: false,
            invalid_transformer_dtype: false,
        }
    }

    pub(super) fn move_indexed_text_tensor_to_wrong_shard(&mut self) {
        let first_tensor_name = "model.embed_tokens.weight";
        self.text_index.insert(
            first_tensor_name.to_owned(),
            "model-00002-of-00002.safetensors".to_owned(),
        );
    }

    pub(super) fn invalidate_transformer_shape(&mut self) {
        self.invalid_transformer_shape = true;
    }

    pub(super) fn invalidate_transformer_dtype(&mut self) {
        self.invalid_transformer_dtype = true;
    }

    pub(super) fn write(&self, model_directory: &Path) {
        for nested_directory in [
            "scheduler",
            "text_encoder",
            "tokenizer",
            "transformer",
            "vae",
        ] {
            fs::create_dir_all(model_directory.join(nested_directory))
                .expect("the nested fixture directory should be created");
        }
        write(model_directory, "model_index.json", &model_index_json());
        write(model_directory, "LICENSE.md", &apache_license_bytes());
        write(
            model_directory,
            "scheduler/scheduler_config.json",
            &scheduler_config_json(),
        );
        write(
            model_directory,
            "text_encoder/config.json",
            &text_encoder_config_json(),
        );
        write(
            model_directory,
            "text_encoder/generation_config.json",
            &serde_json::to_vec(&json!({
                "bos_token_id": 151643,
                "do_sample": true,
                "eos_token_id": [151645, 151643],
                "pad_token_id": 151643,
                "temperature": 0.6,
                "top_k": 20,
                "top_p": 0.95,
                "transformers_version": "4.56.1"
            }))
            .expect("the generation config should serialize"),
        );
        write(
            model_directory,
            "transformer/config.json",
            &transformer_config_json(),
        );
        write(model_directory, "vae/config.json", &vae_config_json());
        for sidecar_name in [
            "added_tokens.json",
            "chat_template.jinja",
            "merges.txt",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ] {
            write(
                model_directory,
                &format!("tokenizer/{sidecar_name}"),
                b"{}\n",
            );
        }
        let text_index = json!({
            "metadata": {
                "total_parameters": self.text_tensors_by_shard.values().flatten().map(SyntheticTensor::parameter_count).sum::<usize>(),
                "total_size": self.text_tensors_by_shard.values().flatten().map(SyntheticTensor::payload_bytes).sum::<usize>()
            },
            "weight_map": self.text_index,
        });
        write(
            model_directory,
            "text_encoder/model.safetensors.index.json",
            &serde_json::to_vec(&text_index).expect("the text index should serialize"),
        );
        for (relative_name, tensors) in &self.text_tensors_by_shard {
            write(model_directory, relative_name, &safetensors_bytes(tensors));
        }
        let mut transformer_tensors = transformer_tensors();
        if self.invalid_transformer_shape {
            transformer_tensors[0].shape[0] += 1;
        }
        if self.invalid_transformer_dtype {
            transformer_tensors[0].dtype = "F16";
        }
        write_sparse_safetensors(
            model_directory,
            "transformer/diffusion_pytorch_model.safetensors",
            &transformer_tensors,
        );
        write(
            model_directory,
            "vae/diffusion_pytorch_model.safetensors",
            &safetensors_bytes(&vae_tensors()),
        );
        write(
            model_directory,
            "flux-2-klein-4b.safetensors",
            &self.root_transformer_bytes,
        );
    }
}

fn transformer_tensors() -> Vec<SyntheticTensor> {
    Flux2KleinTransformerConfig::parse(&transformer_config_json())
        .expect("the official transformer fixture config should parse")
        .expected_weight_shapes()
        .map(|(name, shape)| SyntheticTensor {
            name,
            dtype: "BF16",
            shape,
        })
        .collect()
}

fn vae_tensors() -> Vec<SyntheticTensor> {
    let mut tensors = vec![
        typed_tensor("bn.num_batches_tracked", "I64", vec![]),
        tensor("bn.running_mean", vec![4]),
        tensor("bn.running_var", vec![4]),
    ];
    for prefix in [
        "decoder.conv_in",
        "decoder.conv_norm_out",
        "decoder.conv_out",
        "encoder.conv_in",
        "encoder.conv_norm_out",
        "encoder.conv_out",
        "quant_conv",
        "post_quant_conv",
    ] {
        push_weight_and_bias(&mut tensors, prefix);
    }
    push_mid_block(&mut tensors, "decoder.mid_block");
    push_mid_block(&mut tensors, "encoder.mid_block");
    for block_index in 0..4 {
        for resnet_index in 0..3 {
            push_resnet(
                &mut tensors,
                &format!("decoder.up_blocks.{block_index}.resnets.{resnet_index}"),
                resnet_index == 0 && block_index >= 2,
            );
        }
        if block_index < 3 {
            push_weight_and_bias(
                &mut tensors,
                &format!("decoder.up_blocks.{block_index}.upsamplers.0.conv"),
            );
        }
    }
    for block_index in 0..4 {
        for resnet_index in 0..2 {
            push_resnet(
                &mut tensors,
                &format!("encoder.down_blocks.{block_index}.resnets.{resnet_index}"),
                resnet_index == 0 && matches!(block_index, 1 | 2),
            );
        }
        if block_index < 3 {
            push_weight_and_bias(
                &mut tensors,
                &format!("encoder.down_blocks.{block_index}.downsamplers.0.conv"),
            );
        }
    }
    tensors
}

fn push_weight_and_bias(tensors: &mut Vec<SyntheticTensor>, prefix: &str) {
    tensors.push(tensor(&format!("{prefix}.weight"), vec![4, 4]));
    tensors.push(tensor(&format!("{prefix}.bias"), vec![4]));
}

fn push_resnet(tensors: &mut Vec<SyntheticTensor>, prefix: &str, has_shortcut: bool) {
    for child in ["conv1", "conv2", "norm1", "norm2"] {
        push_weight_and_bias(tensors, &format!("{prefix}.{child}"));
    }
    if has_shortcut {
        push_weight_and_bias(tensors, &format!("{prefix}.conv_shortcut"));
    }
}

fn push_mid_block(tensors: &mut Vec<SyntheticTensor>, prefix: &str) {
    for resnet_index in 0..2 {
        push_resnet(tensors, &format!("{prefix}.resnets.{resnet_index}"), false);
    }
    for child in ["group_norm", "to_k", "to_q", "to_v", "to_out.0"] {
        push_weight_and_bias(tensors, &format!("{prefix}.attentions.0.{child}"));
    }
}

fn tensor(name: &str, shape: Vec<usize>) -> SyntheticTensor {
    typed_tensor(name, "BF16", shape)
}

fn typed_tensor(name: &str, dtype: &'static str, shape: Vec<usize>) -> SyntheticTensor {
    SyntheticTensor {
        name: name.to_owned(),
        dtype,
        shape,
    }
}

fn safetensors_bytes(tensors: &[SyntheticTensor]) -> Vec<u8> {
    let mut payload_offset = 0_usize;
    let mut header = serde_json::Map::new();
    for tensor in tensors {
        let payload_end = payload_offset + tensor.payload_bytes();
        header.insert(
            tensor.name.clone(),
            json!({"dtype": tensor.dtype, "shape": tensor.shape, "data_offsets": [payload_offset, payload_end]}),
        );
        payload_offset = payload_end;
    }
    let header_bytes =
        serde_json::to_vec(&Value::Object(header)).expect("the header should serialize");
    let mut file_bytes = Vec::new();
    file_bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    file_bytes.extend_from_slice(&header_bytes);
    file_bytes.resize(file_bytes.len() + payload_offset, 0);
    file_bytes
}

fn write_sparse_safetensors(
    model_directory: &Path,
    relative_name: &str,
    tensors: &[SyntheticTensor],
) {
    let mut payload_offset = 0_u64;
    let mut header = serde_json::Map::new();
    for tensor in tensors {
        let tensor_payload_bytes = tensor.payload_bytes() as u64;
        let payload_end = payload_offset + tensor_payload_bytes;
        header.insert(
            tensor.name.clone(),
            json!({"dtype": tensor.dtype, "shape": tensor.shape, "data_offsets": [payload_offset, payload_end]}),
        );
        payload_offset = payload_end;
    }
    let header_bytes = serde_json::to_vec(&Value::Object(header))
        .expect("the sparse safetensors header should serialize");
    let file_path = model_directory.join(relative_name);
    let mut file = File::create(file_path).expect("the sparse safetensors file should be created");
    file.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .and_then(|()| file.write_all(&header_bytes))
        .expect("the sparse safetensors header should be written");
    file.set_len(8 + header_bytes.len() as u64 + payload_offset)
        .expect("the sparse safetensors payload extent should be allocated");
}

impl SyntheticTensor {
    fn parameter_count(&self) -> usize {
        self.shape.iter().product()
    }

    fn payload_bytes(&self) -> usize {
        let bytes_per_element = match self.dtype {
            "BF16" => 2,
            "F16" => 2,
            "I64" => 8,
            _ => panic!("the synthetic tensor dtype must have a byte width"),
        };
        self.shape.iter().product::<usize>() * bytes_per_element
    }
}

fn apache_license_bytes() -> Vec<u8> {
    b"Apache License\nVersion 2.0, January 2004\nhttp://www.apache.org/licenses/\nEND OF TERMS AND CONDITIONS\n".to_vec()
}

fn write(model_directory: &Path, relative_name: &str, bytes: &[u8]) {
    fs::write(model_directory.join(relative_name), bytes)
        .expect("the synthetic artifact file should be written");
}
