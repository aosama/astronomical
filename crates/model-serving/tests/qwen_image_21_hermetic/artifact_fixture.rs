//! Synthetic Qwen-Image-2.1 MLX artifact mirroring the reviewed package's wire shape.
//!
//! The config documents are the reviewed package's exact JSON (the strict validator pins them to
//! these values), and the three weight files are sparse — a real safetensors header plus a
//! zero-filled payload extent — so the hermetic suite never materializes the 10.5 GB of weights.
//! Tensor names, dtypes, and shapes come from the crate's own profile generators, the same
//! single source of truth the validator checks against.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

use astronomical_model_serving::{
    QwenImage21TensorProfile, QwenImage21TextEncoderConfig, QwenImage21TransformerConfig,
    QwenImage21VaeConfig, text_encoder_tensor_profiles, transformer_tensor_profiles,
    vae_tensor_profiles,
};
use serde_json::{Value, json};

const PROCESSOR_SIDECARS: [&str; 9] = [
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "video_preprocessor_config.json",
    "vocab.json",
];

/// One synthetic weight component: the reviewed config bytes plus its physical tensors.
pub(super) struct SyntheticComponent {
    pub(super) config_bytes: Vec<u8>,
    pub(super) tensors: Vec<QwenImage21TensorProfile>,
}

impl SyntheticComponent {
    fn payload_bytes(&self) -> u64 {
        self.tensors
            .iter()
            .map(|tensor| tensor.shape.iter().product::<usize>() as u64 * dtype_bytes(tensor.dtype))
            .sum()
    }
}

pub(super) struct SyntheticQwenImage21Artifact {
    transformer: SyntheticComponent,
    text_encoder: SyntheticComponent,
    vae: SyntheticComponent,
    pipeline_class: String,
    index_total_size_delta: i64,
    index_shard_override: Option<String>,
    delete_vae_weights: bool,
}

impl SyntheticQwenImage21Artifact {
    pub(super) fn reviewed() -> Self {
        let transformer_config = transformer_config_json();
        let text_encoder_config = text_encoder_config_json();
        let vae_config = vae_config_json();
        let transformer_parsed = QwenImage21TransformerConfig::parse(&transformer_config)
            .expect("the reviewed transformer config should parse for the fixture");
        let text_encoder_parsed = QwenImage21TextEncoderConfig::parse(&text_encoder_config)
            .expect("the reviewed text encoder config should parse for the fixture");
        let vae_parsed = QwenImage21VaeConfig::parse(&vae_config)
            .expect("the reviewed vae config should parse for the fixture");
        Self {
            transformer: SyntheticComponent {
                config_bytes: transformer_config,
                tensors: transformer_tensor_profiles(&transformer_parsed),
            },
            text_encoder: SyntheticComponent {
                config_bytes: text_encoder_config,
                tensors: text_encoder_tensor_profiles(&text_encoder_parsed),
            },
            vae: SyntheticComponent {
                config_bytes: vae_config,
                tensors: vae_tensor_profiles(&vae_parsed),
            },
            pipeline_class: "QwenImage21Pipeline".to_owned(),
            index_total_size_delta: 0,
            index_shard_override: None,
            delete_vae_weights: false,
        }
    }

    pub(super) fn set_pipeline_class(&mut self, class_name: &str) {
        self.pipeline_class = class_name.to_owned();
    }

    pub(super) fn use_bits_eight_transformer_quantization(&mut self) {
        // The crate's parser itself rejects 8-bit quantization, so the fixture writes the
        // tampered bytes directly: the validator must reject them during config parsing, long
        // before the tensor profiles matter.
        self.transformer.config_bytes = transformer_config_with_bits(8);
    }

    pub(super) fn drop_transformer_tensor(&mut self) {
        // Drop a top-level tensor so the rejection is unambiguous in the error name.
        self.transformer
            .tensors
            .retain(|tensor| tensor.tensor_name != "proj_out.weight");
    }

    pub(super) fn add_extra_transformer_tensor(&mut self) {
        self.transformer.tensors.push(QwenImage21TensorProfile {
            tensor_name: "transformer_blocks.0.attn.extra_projection.weight".to_owned(),
            dtype: "U32",
            shape: vec![4096, 512],
        });
    }

    pub(super) fn flip_first_vae_dtype(&mut self) {
        if let Some(first_tensor) = self.vae.tensors.first_mut() {
            first_tensor.dtype = "F16";
        }
    }

    pub(super) fn widen_first_transformer_tensor(&mut self) {
        if let Some(first_tensor) = self.transformer.tensors.first_mut() {
            if let Some(last_dimension) = first_tensor.shape.last_mut() {
                *last_dimension += 1;
            }
        }
    }

    pub(super) fn set_index_total_size_delta(&mut self, delta: i64) {
        self.index_total_size_delta = delta;
    }

    pub(super) fn set_index_shard_override(&mut self, tensor_name: &str) {
        self.index_shard_override = Some(tensor_name.to_owned());
    }
    pub(super) fn delete_vae_weights(&mut self) {
        self.delete_vae_weights = true;
    }

    pub(super) fn write(&self, model_directory: &Path) {
        for nested_directory in [
            "processor",
            "scheduler",
            "text_encoder",
            "transformer",
            "vae",
        ] {
            fs::create_dir_all(model_directory.join(nested_directory))
                .expect("the nested fixture directory should be created");
        }
        write(
            model_directory,
            "model_index.json",
            &model_index_json(self.pipeline_class.as_str()),
        );
        write(
            model_directory,
            "scheduler/scheduler_config.json",
            &scheduler_config_json(),
        );
        write(
            model_directory,
            "text_encoder/config.json",
            &self.text_encoder.config_bytes,
        );
        write(
            model_directory,
            "transformer/config.json",
            &self.transformer.config_bytes,
        );
        write(model_directory, "vae/config.json", &self.vae.config_bytes);
        for sidecar_name in PROCESSOR_SIDECARS {
            write(
                model_directory,
                &format!("processor/{sidecar_name}"),
                b"{}\n",
            );
        }
        self.write_component(model_directory, "text_encoder", &self.text_encoder);
        self.write_component(model_directory, "transformer", &self.transformer);
        if self.delete_vae_weights {
            write_index(
                model_directory,
                "vae",
                &self.vae,
                self.index_total_size_delta,
                self.index_shard_override.as_deref(),
            );
        } else {
            self.write_component(model_directory, "vae", &self.vae);
        }
    }

    fn write_component(
        &self,
        model_directory: &Path,
        component: &str,
        weights: &SyntheticComponent,
    ) {
        write_index(
            model_directory,
            component,
            weights,
            self.index_total_size_delta,
            self.index_shard_override.as_deref(),
        );
        write_sparse_safetensors(
            model_directory,
            &format!("{component}/model.safetensors"),
            &weights.tensors,
        );
    }
}

fn write_index(
    model_directory: &Path,
    component: &str,
    weights: &SyntheticComponent,
    total_size_delta: i64,
    shard_override: Option<&str>,
) {
    let mut weight_map = BTreeMap::new();
    for tensor in &weights.tensors {
        // Point exactly one tensor at a second, nonexistent shard to model a multi-shard package.
        let shard_name = if Some(tensor.tensor_name.as_str()) == shard_override {
            "model-00001-of-00002.safetensors"
        } else {
            "model.safetensors"
        };
        weight_map.insert(tensor.tensor_name.clone(), shard_name.to_owned());
    }
    let total_size = weights.payload_bytes() as i64 + total_size_delta;
    let index = json!({
        "metadata": { "total_size": total_size },
        "weight_map": weight_map,
    });
    write(
        model_directory,
        &format!("{component}/model.safetensors.index.json"),
        &serde_json::to_vec(&index).expect("the shard index should serialize"),
    );
}

fn write_sparse_safetensors(
    model_directory: &Path,
    relative_name: &str,
    tensors: &[QwenImage21TensorProfile],
) {
    let mut payload_offset = 0_u64;
    let mut header = serde_json::Map::new();
    for tensor in tensors {
        let tensor_payload_bytes =
            tensor.shape.iter().product::<usize>() as u64 * dtype_bytes(tensor.dtype);
        let payload_end = payload_offset + tensor_payload_bytes;
        header.insert(
            tensor.tensor_name.clone(),
            json!({
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [payload_offset, payload_end],
            }),
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

fn dtype_bytes(dtype: &str) -> u64 {
    match dtype {
        "U32" | "F32" => 4,
        "BF16" | "F16" => 2,
        other => panic!("the fixture only supports reviewed dtypes, got {other}"),
    }
}

fn write(model_directory: &Path, relative_name: &str, bytes: &[u8]) {
    fs::write(model_directory.join(relative_name), bytes)
        .expect("the synthetic artifact file should be written");
}

// ---------------------------------------------------------------------------
// Reviewed wire documents
// ---------------------------------------------------------------------------

fn model_index_json(class_name: &str) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "_class_name": class_name,
        "_diffusers_version": "0.37.0.dev0",
        "processor": ["transformers", "Qwen3VLProcessor"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
        "transformer": ["diffusers", "QwenImage21Transformer2DModel"],
        "vae": ["diffusers", "AutoencoderKLQwenImage21"]
    }))
    .expect("the model index should serialize")
}

fn scheduler_config_json() -> Vec<u8> {
    serde_json::to_vec(&json!({
        "_class_name": "FlowMatchEulerDiscreteScheduler",
        "_diffusers_version": "0.37.0.dev0",
        "base_image_seq_len": 256,
        "base_shift": 0.5,
        "invert_sigmas": false,
        "max_image_seq_len": 8192,
        "max_shift": 0.9,
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": 0.02,
        "stochastic_sampling": false,
        "time_shift_type": "exponential",
        "use_beta_sigmas": false,
        "use_dynamic_shifting": true,
        "use_exponential_sigmas": false,
        "use_karras_sigmas": false
    }))
    .expect("the scheduler config should serialize")
}

fn transformer_config_with_bits(bits: u32) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "_class_name": "QwenImage21Transformer2DModel",
        "_diffusers_version": "0.37.0.dev0",
        "attention_head_dim": 128,
        "axes_dims_rope": [16, 56, 56],
        "context_in_dim": 4096,
        "eps": 0.000001,
        "in_channels": 64,
        "mlp_ratio": 3,
        "num_attention_heads": 32,
        "num_layers": 32,
        "out_channels": 64,
        "patch_size": 1,
        "causal_condition": true,
        "quantization": { "bits": bits, "group_size": 64, "mode": "affine" },
        "mlx_format": true
    }))
    .expect("the transformer config should serialize")
}

fn transformer_config_json() -> Vec<u8> {
    transformer_config_with_bits(4)
}

fn text_encoder_config_json() -> Vec<u8> {
    serde_json::to_vec(&json!({
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "dtype": "bfloat16",
        "image_token_id": 151655,
        "model_type": "qwen3_vl",
        "text_config": {
            "attention_bias": false,
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "dtype": "bfloat16",
            "eos_token_id": 151645,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 12288,
            "max_position_embeddings": 262144,
            "model_type": "qwen3_vl_text",
            "num_attention_heads": 32,
            "num_hidden_layers": 36,
            "num_key_value_heads": 8,
            "rms_norm_eps": 0.000001,
            "rope_scaling": {
                "mrope_interleaved": true,
                "mrope_section": [24, 20, 20],
                "rope_type": "default"
            },
            "rope_theta": 5000000,
            "use_cache": true,
            "vocab_size": 151936
        },
        "tie_word_embeddings": false,
        "transformers_version": "4.57.1",
        "video_token_id": 151656,
        "vision_config": {
            "deepstack_visual_indexes": [8, 16, 24],
            "depth": 27,
            "dtype": "bfloat16",
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 1152,
            "in_channels": 3,
            "initializer_range": 0.02,
            "intermediate_size": 4304,
            "model_type": "qwen3_vl",
            "num_heads": 16,
            "num_position_embeddings": 2304,
            "out_hidden_size": 4096,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2
        },
        "vision_end_token_id": 151653,
        "vision_start_token_id": 151652,
        "quantization": { "bits": 4, "group_size": 64, "mode": "affine" },
        "mlx_format": true
    }))
    .expect("the text encoder config should serialize")
}

fn vae_config_json() -> Vec<u8> {
    // Deterministic stand-ins for the 64 per-channel latent constants: the validator checks
    // count and positivity; the engine's denormalization is covered against the real package.
    let latents_mean: Vec<f64> = (0..64)
        .map(|channel| channel as f64 * 0.01 - 0.32)
        .collect();
    let latents_std: Vec<f64> = (0..64)
        .map(|channel| 1.0 + channel as f64 * 0.001)
        .collect();
    serde_json::to_vec(&json!({
        "_class_name": "AutoencoderKLQwenImage21",
        "_diffusers_version": "0.37.0.dev0",
        "attn_scales": [],
        "base_dim": 96,
        "decoder_base_dim": 144,
        "dim_mult": [1, 2, 4, 8, 8],
        "dropout": 0.0,
        "in_channels": 4,
        "is_residual": true,
        "latents_mean": latents_mean,
        "latents_std": latents_std,
        "num_res_blocks": 2,
        "out_channels": 4,
        "patch_size": null,
        "scale_factor_spatial": 16,
        "scale_factor_temporal": 8,
        "temperal_downsample": [false, true, true, true],
        "z_dim": 64,
        "mlx_format": true
    }))
    .expect("the vae config should serialize")
}
