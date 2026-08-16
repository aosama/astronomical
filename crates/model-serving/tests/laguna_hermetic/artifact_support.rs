use std::collections::BTreeMap;
use std::fs;

use serde_json::{Map, Value, json};

use super::{artifact_text_validation::synthetic_text_sidecars, support::config_value};

pub(super) const FIRST_SHARD_FILE_NAME: &str = "model-00001-of-00002.safetensors";
pub(super) const SECOND_SHARD_FILE_NAME: &str = "model-00002-of-00002.safetensors";
pub(super) const SYNTHETIC_BOS_TOKEN_ID: u32 = 1;
pub(super) const SYNTHETIC_PAD_TOKEN_ID: u32 = 2;
pub(super) const SYNTHETIC_EOS_TOKEN_ID: u32 = 3;

/// One synthetic tensor declaration whose payload contains deterministic zero bytes.
#[derive(Clone)]
pub(super) struct SyntheticTensor {
    pub(super) name: String,
    pub(super) dtype: &'static str,
    pub(super) shape: Vec<usize>,
}

impl SyntheticTensor {
    pub(super) fn payload_bytes(&self) -> usize {
        let bytes_per_element = match self.dtype {
            "F32" | "I32" => 4,
            "I64" => 8,
            "F16" | "BF16" => 2,
            "U32" => 4,
            "U8" | "F8_E4M3" => 1,
            _ => panic!("the synthetic fixture dtype must have an explicit byte width"),
        };
        self.shape.iter().product::<usize>() * bytes_per_element
    }
}

/// Mutable directory fixture used to express physical/index disagreement precisely.
pub(super) struct SyntheticLagunaArtifact {
    pub(super) config: Value,
    pub(super) tensors_by_shard: BTreeMap<String, Vec<SyntheticTensor>>,
    pub(super) indexed_shard_by_tensor: BTreeMap<String, String>,
    pub(super) declared_shard_file_size_override: Option<u64>,
}

impl SyntheticLagunaArtifact {
    pub(super) fn dense(namespace_prefix: &str) -> Self {
        let mut config = config_value(1);
        config["vocab_size"] = json!(8);
        config["hidden_size"] = json!(4);
        config["intermediate_size"] = json!(6);
        config["num_attention_heads"] = json!(2);
        config["num_key_value_heads"] = json!(1);
        config["head_dim"] = json!(2);
        config["max_position_embeddings"] = json!(32);
        config["bos_token_id"] = json!(SYNTHETIC_BOS_TOKEN_ID);
        config["pad_token_id"] = json!(SYNTHETIC_PAD_TOKEN_ID);
        config["eos_token_id"] = json!(SYNTHETIC_EOS_TOKEN_ID);
        config["torch_dtype"] = json!("float32");
        config["rope_parameters"]["partial_rotary_factor"] = json!(1.0);

        let tensor_shapes = [
            ("model.embed_tokens.weight", vec![8, 4]),
            ("model.norm.weight", vec![4]),
            ("lm_head.weight", vec![8, 4]),
            ("model.layers.0.input_layernorm.weight", vec![4]),
            ("model.layers.0.post_attention_layernorm.weight", vec![4]),
            ("model.layers.0.self_attn.q_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.k_proj.weight", vec![2, 4]),
            ("model.layers.0.self_attn.v_proj.weight", vec![2, 4]),
            ("model.layers.0.self_attn.o_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.q_norm.weight", vec![2]),
            ("model.layers.0.self_attn.k_norm.weight", vec![2]),
            ("model.layers.0.mlp.gate_proj.weight", vec![6, 4]),
            ("model.layers.0.mlp.up_proj.weight", vec![6, 4]),
            ("model.layers.0.mlp.down_proj.weight", vec![4, 6]),
        ];
        Self::from_tensor_shapes(config, namespace_prefix, &tensor_shapes)
    }

    pub(super) fn sparse_stacked() -> Self {
        let mut config = Self::dense("").config;
        config["mlp_layer_types"] = json!(["sparse"]);
        config["num_experts"] = json!(2);
        config["num_experts_per_tok"] = json!(1);
        config["moe_intermediate_size"] = json!(3);
        config["shared_expert_intermediate_size"] = json!(0);
        let tensor_shapes = [
            ("model.embed_tokens.weight", vec![8, 4]),
            ("model.norm.weight", vec![4]),
            ("lm_head.weight", vec![8, 4]),
            ("model.layers.0.input_layernorm.weight", vec![4]),
            ("model.layers.0.post_attention_layernorm.weight", vec![4]),
            ("model.layers.0.self_attn.q_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.k_proj.weight", vec![2, 4]),
            ("model.layers.0.self_attn.v_proj.weight", vec![2, 4]),
            ("model.layers.0.self_attn.o_proj.weight", vec![4, 4]),
            ("model.layers.0.self_attn.q_norm.weight", vec![2]),
            ("model.layers.0.self_attn.k_norm.weight", vec![2]),
            ("model.layers.0.mlp.gate.weight", vec![2, 4]),
            ("model.layers.0.mlp.e_score_correction_bias", vec![2]),
            ("model.layers.0.mlp.experts.gate_proj.weight", vec![2, 3, 4]),
            ("model.layers.0.mlp.experts.up_proj.weight", vec![2, 3, 4]),
            ("model.layers.0.mlp.experts.down_proj.weight", vec![2, 4, 3]),
        ];
        Self::from_tensor_shapes(config, "", &tensor_shapes)
    }

    /// Builds a dimensionally divisible dense artifact in the exact direct-MLX affine form.
    pub(super) fn direct_affine_dense(
        namespace_prefix: &str,
        bit_width: u32,
        group_size: u32,
        module_overrides: &[(&str, u32, u32)],
    ) -> Self {
        let mut fixture = Self::dense(namespace_prefix);
        fixture.config["vocab_size"] = json!(256);
        fixture.config["hidden_size"] = json!(128);
        fixture.config["intermediate_size"] = json!(128);
        fixture.config["num_attention_heads"] = json!(4);
        fixture.config["num_key_value_heads"] = json!(2);
        fixture.config["head_dim"] = json!(32);
        fixture.replace_dense_logical_shapes(namespace_prefix);
        fixture.apply_direct_affine(bit_width, group_size, module_overrides);
        fixture
    }

    /// Builds split stacked routed experts plus an ordinary shared SwiGLU expert.
    pub(super) fn direct_affine_sparse_stacked(bit_width: u32, group_size: u32) -> Self {
        let mut fixture = Self::direct_affine_sparse_fixture(false, false);
        fixture.apply_direct_affine(bit_width, group_size, &[]);
        fixture
    }

    /// Builds split per-expert routed sources so every affine component must stack.
    pub(super) fn direct_affine_sparse_per_expert(bit_width: u32, group_size: u32) -> Self {
        let mut fixture = Self::direct_affine_sparse_fixture(true, false);
        fixture.apply_direct_affine(bit_width, group_size, &[]);
        fixture
    }

    /// Builds one fused stacked gate/up source for every affine component.
    pub(super) fn direct_affine_sparse_fused_stacked(bit_width: u32, group_size: u32) -> Self {
        let mut fixture = Self::direct_affine_sparse_fixture(false, true);
        fixture.apply_direct_affine(bit_width, group_size, &[]);
        fixture
    }

    /// Builds fused per-expert gate/up sources for component-wise split validation.
    pub(super) fn direct_affine_sparse_fused_per_expert(bit_width: u32, group_size: u32) -> Self {
        let mut fixture = Self::direct_affine_sparse_fixture(true, true);
        fixture.apply_direct_affine(bit_width, group_size, &[]);
        fixture
    }

    pub(super) fn direct_affine_sparse_fixture(
        is_per_expert: bool,
        is_fused_gate_up: bool,
    ) -> Self {
        let mut config = Self::direct_affine_dense("", 2, 32, &[]).config;
        config
            .as_object_mut()
            .expect("the config must be an object")
            .remove("quantization");
        config["mlp_layer_types"] = json!(["sparse"]);
        config["num_experts"] = json!(2);
        config["num_experts_per_tok"] = json!(1);
        config["moe_intermediate_size"] = json!(128);
        config["shared_expert_intermediate_size"] = json!(128);
        let mut tensor_shapes = vec![
            ("model.embed_tokens.weight".to_owned(), vec![256, 128]),
            ("model.norm.weight".to_owned(), vec![128]),
            ("lm_head.weight".to_owned(), vec![256, 128]),
            (
                "model.layers.0.input_layernorm.weight".to_owned(),
                vec![128],
            ),
            (
                "model.layers.0.post_attention_layernorm.weight".to_owned(),
                vec![128],
            ),
            (
                "model.layers.0.self_attn.q_proj.weight".to_owned(),
                vec![128, 128],
            ),
            (
                "model.layers.0.self_attn.k_proj.weight".to_owned(),
                vec![64, 128],
            ),
            (
                "model.layers.0.self_attn.v_proj.weight".to_owned(),
                vec![64, 128],
            ),
            (
                "model.layers.0.self_attn.o_proj.weight".to_owned(),
                vec![128, 128],
            ),
            (
                "model.layers.0.self_attn.q_norm.weight".to_owned(),
                vec![32],
            ),
            (
                "model.layers.0.self_attn.k_norm.weight".to_owned(),
                vec![32],
            ),
            ("model.layers.0.mlp.gate.weight".to_owned(), vec![2, 128]),
            (
                "model.layers.0.mlp.e_score_correction_bias".to_owned(),
                vec![2],
            ),
            (
                "model.layers.0.mlp.shared_expert.gate_proj.weight".to_owned(),
                vec![128, 128],
            ),
            (
                "model.layers.0.mlp.shared_expert.up_proj.weight".to_owned(),
                vec![128, 128],
            ),
            (
                "model.layers.0.mlp.shared_expert.down_proj.weight".to_owned(),
                vec![128, 128],
            ),
        ];
        if is_per_expert {
            for expert_index in 0..2 {
                if is_fused_gate_up {
                    tensor_shapes.push((
                        format!("model.layers.0.mlp.experts.{expert_index}.gate_up_proj.weight"),
                        vec![256, 128],
                    ));
                } else {
                    for projection_name in ["gate_proj", "up_proj"] {
                        tensor_shapes.push((
                            format!(
                                "model.layers.0.mlp.experts.{expert_index}.{projection_name}.weight"
                            ),
                            vec![128, 128],
                        ));
                    }
                }
                tensor_shapes.push((
                    format!("model.layers.0.mlp.experts.{expert_index}.down_proj.weight"),
                    vec![128, 128],
                ));
            }
        } else {
            let projection_names: &[&str] = if is_fused_gate_up {
                &["gate_up_proj", "down_proj"]
            } else {
                &["gate_proj", "up_proj", "down_proj"]
            };
            for projection_name in projection_names {
                let output_width = if *projection_name == "gate_up_proj" {
                    256
                } else {
                    128
                };
                tensor_shapes.push((
                    format!("model.layers.0.mlp.switch_mlp.{projection_name}.weight"),
                    vec![2, output_width, 128],
                ));
            }
        }
        let tensor_shape_refs = tensor_shapes
            .iter()
            .map(|(tensor_name, shape)| (tensor_name.as_str(), shape.clone()))
            .collect::<Vec<_>>();
        Self::from_tensor_shapes(config, "", &tensor_shape_refs)
    }

    pub(super) fn replace_dense_logical_shapes(&mut self, namespace_prefix: &str) {
        for (bare_name, shape) in [
            ("model.embed_tokens.weight", vec![256, 128]),
            ("model.norm.weight", vec![128]),
            ("lm_head.weight", vec![256, 128]),
            ("model.layers.0.input_layernorm.weight", vec![128]),
            ("model.layers.0.post_attention_layernorm.weight", vec![128]),
            ("model.layers.0.self_attn.q_proj.weight", vec![128, 128]),
            ("model.layers.0.self_attn.k_proj.weight", vec![64, 128]),
            ("model.layers.0.self_attn.v_proj.weight", vec![64, 128]),
            ("model.layers.0.self_attn.o_proj.weight", vec![128, 128]),
            ("model.layers.0.self_attn.q_norm.weight", vec![32]),
            ("model.layers.0.self_attn.k_norm.weight", vec![32]),
            ("model.layers.0.mlp.gate_proj.weight", vec![128, 128]),
            ("model.layers.0.mlp.up_proj.weight", vec![128, 128]),
            ("model.layers.0.mlp.down_proj.weight", vec![128, 128]),
        ] {
            self.tensor_mut(&format!("{namespace_prefix}{bare_name}"))
                .shape = shape;
        }
    }

    fn apply_direct_affine(
        &mut self,
        default_bit_width: u32,
        default_group_size: u32,
        module_overrides: &[(&str, u32, u32)],
    ) {
        let mut quantization_fields = Map::new();
        quantization_fields.insert("bits".to_owned(), json!(default_bit_width));
        quantization_fields.insert("group_size".to_owned(), json!(default_group_size));
        quantization_fields.insert("mode".to_owned(), json!("affine"));
        for (module_name, bit_width, group_size) in module_overrides {
            quantization_fields.insert(
                (*module_name).to_owned(),
                json!({"bits": bit_width, "group_size": group_size, "mode": "affine"}),
            );
        }
        self.config["quantization"] = Value::Object(quantization_fields);

        // Sidecars remain in the source module namespace while profile lookup uses its wrapper-free ID.
        for (shard_file_name, shard_tensors) in &mut self.tensors_by_shard {
            let mut sidecars = Vec::new();
            for weight_tensor in shard_tensors.iter_mut() {
                let Some(raw_module_name) = weight_tensor.name.strip_suffix(".weight") else {
                    continue;
                };
                let canonical_module_name = raw_module_name
                    .strip_prefix("language_model.")
                    .unwrap_or(raw_module_name);
                if !is_direct_affine_module(canonical_module_name) {
                    continue;
                }
                let (bit_width, group_size) = module_overrides
                    .iter()
                    .find(|(override_name, _, _)| {
                        override_name
                            .strip_prefix("language_model.")
                            .unwrap_or(*override_name)
                            == canonical_module_name
                    })
                    .map(|(_, bit_width, group_size)| (*bit_width, *group_size))
                    .unwrap_or((default_bit_width, default_group_size));
                let logical_input_width = *weight_tensor
                    .shape
                    .last()
                    .expect("an affine test matrix must have an input axis");
                let packed_input_bits = logical_input_width
                    .checked_mul(bit_width as usize)
                    .expect("the affine test input width must not overflow");
                assert_eq!(packed_input_bits % 32, 0);
                assert_eq!(logical_input_width % group_size as usize, 0);
                let mut scale_shape = weight_tensor.shape.clone();
                *weight_tensor
                    .shape
                    .last_mut()
                    .expect("an affine test matrix must have an input axis") =
                    packed_input_bits / 32;
                *scale_shape
                    .last_mut()
                    .expect("an affine scale must have an input-group axis") =
                    logical_input_width / group_size as usize;
                weight_tensor.dtype = "U32";
                for component_name in ["scales", "biases"] {
                    let sidecar_name = format!("{raw_module_name}.{component_name}");
                    sidecars.push(SyntheticTensor {
                        name: sidecar_name.clone(),
                        dtype: "F32",
                        shape: scale_shape.clone(),
                    });
                    self.indexed_shard_by_tensor
                        .insert(sidecar_name, shard_file_name.clone());
                }
            }
            shard_tensors.extend(sidecars);
        }
    }

    pub(super) fn from_tensor_shapes(
        config: Value,
        namespace_prefix: &str,
        tensor_shapes: &[(&str, Vec<usize>)],
    ) -> Self {
        let mut tensors_by_shard = BTreeMap::from([
            (FIRST_SHARD_FILE_NAME.to_owned(), Vec::new()),
            (SECOND_SHARD_FILE_NAME.to_owned(), Vec::new()),
        ]);
        let mut indexed_shard_by_tensor = BTreeMap::new();
        for (tensor_position, (bare_name, shape)) in tensor_shapes.iter().enumerate() {
            let tensor_name = format!("{namespace_prefix}{bare_name}");
            let shard_file_name = if tensor_position.is_multiple_of(2) {
                FIRST_SHARD_FILE_NAME
            } else {
                SECOND_SHARD_FILE_NAME
            };
            let tensor = SyntheticTensor {
                name: tensor_name.clone(),
                dtype: "F32",
                shape: shape.clone(),
            };
            tensors_by_shard
                .get_mut(shard_file_name)
                .expect("the synthetic shard must exist")
                .push(tensor);
            indexed_shard_by_tensor.insert(tensor_name, shard_file_name.to_owned());
        }
        Self {
            config,
            tensors_by_shard,
            indexed_shard_by_tensor,
            declared_shard_file_size_override: None,
        }
    }

    pub(super) fn tensor_mut(&mut self, tensor_name: &str) -> &mut SyntheticTensor {
        self.tensors_by_shard
            .values_mut()
            .flat_map(|shard_tensors| shard_tensors.iter_mut())
            .find(|tensor| tensor.name == tensor_name)
            .expect("the requested synthetic tensor must exist")
    }

    pub(super) fn remove_physical_tensor(&mut self, tensor_name: &str) {
        for shard_tensors in self.tensors_by_shard.values_mut() {
            shard_tensors.retain(|tensor| tensor.name != tensor_name);
        }
    }

    pub(super) fn remove_tensor_completely(&mut self, tensor_name: &str) {
        self.remove_physical_tensor(tensor_name);
        self.indexed_shard_by_tensor.remove(tensor_name);
    }

    pub(super) fn add_tensor(&mut self, shard_file_name: &str, tensor: SyntheticTensor) {
        self.indexed_shard_by_tensor
            .insert(tensor.name.clone(), shard_file_name.to_owned());
        self.tensors_by_shard
            .get_mut(shard_file_name)
            .expect("the requested synthetic shard must exist")
            .push(tensor);
    }

    pub(super) fn tensor_payload_bytes(&self) -> u64 {
        self.tensors_by_shard
            .values()
            .flatten()
            .try_fold(0_u64, |total_payload_bytes, tensor| {
                let tensor_payload_bytes =
                    u64::try_from(tensor.payload_bytes()).expect("fixture bytes fit u64");
                total_payload_bytes.checked_add(tensor_payload_bytes)
            })
            .expect("aggregate fixture tensor payload bytes fit u64")
    }

    pub(super) fn serialized_shard_file_bytes(&self) -> u64 {
        self.tensors_by_shard
            .values()
            .try_fold(0_u64, |total_shard_file_bytes, shard_tensors| {
                let shard_file_bytes = u64::try_from(safetensors_bytes(shard_tensors).len())
                    .expect("serialized fixture shard bytes fit u64");
                total_shard_file_bytes.checked_add(shard_file_bytes)
            })
            .expect("aggregate serialized fixture shard bytes fit u64")
    }

    pub(super) fn write(&self, model_directory: &std::path::Path) {
        fs::write(
            model_directory.join("config.json"),
            serde_json::to_vec(&self.config).expect("the synthetic config must serialize"),
        )
        .expect("the synthetic config must be written");
        // Text sidecars follow the mutable config so every weight fixture is a complete artifact.
        for (sidecar_file_name, sidecar_document) in synthetic_text_sidecars(&self.config) {
            fs::write(
                model_directory.join(sidecar_file_name),
                serde_json::to_vec(&sidecar_document)
                    .expect("the synthetic text sidecar must serialize"),
            )
            .expect("the synthetic text sidecar must be written");
        }
        for (shard_file_name, shard_tensors) in &self.tensors_by_shard {
            fs::write(
                model_directory.join(shard_file_name),
                safetensors_bytes(shard_tensors),
            )
            .expect("the synthetic shard must be written");
        }
        let declared_shard_file_bytes = self
            .declared_shard_file_size_override
            .unwrap_or_else(|| self.serialized_shard_file_bytes());
        let index = json!({
            "metadata": {"total_size": declared_shard_file_bytes},
            "weight_map": self.indexed_shard_by_tensor,
        });
        fs::write(
            model_directory.join("model.safetensors.index.json"),
            serde_json::to_vec(&index).expect("the synthetic index must serialize"),
        )
        .expect("the synthetic index must be written");
    }
}

fn is_direct_affine_module(canonical_module_name: &str) -> bool {
    canonical_module_name == "model.embed_tokens"
        || canonical_module_name == "lm_head"
        || canonical_module_name.contains(".self_attn.") && canonical_module_name.ends_with("_proj")
        || canonical_module_name.contains(".mlp.") && canonical_module_name.ends_with("_proj")
}

fn safetensors_bytes(tensors: &[SyntheticTensor]) -> Vec<u8> {
    let mut payload_bytes = Vec::new();
    let mut tensor_entries = Vec::new();
    for tensor in tensors {
        let data_start_offset = payload_bytes.len();
        payload_bytes.resize(data_start_offset + tensor.payload_bytes(), 0);
        let data_end_offset = payload_bytes.len();
        tensor_entries.push(format!(
            "\"{}\":{{\"dtype\":\"{}\",\"shape\":{},\"data_offsets\":[{},{}]}}",
            tensor.name,
            tensor.dtype,
            serde_json::to_string(&tensor.shape).expect("the fixture shape must serialize"),
            data_start_offset,
            data_end_offset,
        ));
    }
    let header = format!("{{{}}}", tensor_entries.join(","));
    let mut shard_bytes = Vec::new();
    shard_bytes.extend_from_slice(
        &u64::try_from(header.len())
            .expect("the fixture header length must fit u64")
            .to_le_bytes(),
    );
    shard_bytes.extend_from_slice(header.as_bytes());
    shard_bytes.extend_from_slice(&payload_bytes);
    shard_bytes
}
