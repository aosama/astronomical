//! Builds a tiny structurally exact folded pack on disk for validation tests.
//!
//! Dimensions are chosen so every packed input width (hidden, intermediate,
//! attention value width, grouped GDN value width) divides by 128, matching
//! the ternary group contract of the shipped 27B pack.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Value, json};

pub(crate) const FIXTURE_HIDDEN_SIZE: u32 = 512;
pub(crate) const FIXTURE_INTERMEDIATE_SIZE: u32 = 1024;
pub(crate) const FIXTURE_VOCAB_SIZE: u32 = 512;
const FIXTURE_NUM_LAYERS: u32 = 4;
const FIXTURE_FULL_ATTENTION_LAYER_INDEX: u32 = 3;
const FIXTURE_ATTENTION_HEADS: u32 = 8;
const FIXTURE_KV_HEADS: u32 = 2;
const FIXTURE_HEAD_DIM: u32 = 128;
// The gated-delta kernel pins the key head dimension to 128, so the fixture
// must adopt it to stay executable in a forward pass, not merely validatable.
const FIXTURE_GDN_VALUE_HEADS: u32 = 8;
const FIXTURE_GDN_KEY_HEADS: u32 = 2;
const FIXTURE_GDN_KEY_HEAD_DIM: u32 = 128;
const FIXTURE_GDN_VALUE_HEAD_DIM: u32 = 128;
pub(crate) const FIXTURE_HADAMARD_BLOCK: u32 = 512;

/// Every packed module path in the fixture, in catalog order.
fn fixture_packed_module_paths() -> Vec<(String, bool)> {
    let mut module_paths = vec![
        ("lm_head".to_owned(), false),
        ("model.embed_tokens".to_owned(), true),
    ];
    for layer_index in 0..FIXTURE_NUM_LAYERS {
        if layer_index == FIXTURE_FULL_ATTENTION_LAYER_INDEX {
            for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                module_paths.push((
                    format!("model.layers.{layer_index}.self_attn.{projection}"),
                    false,
                ));
            }
        } else {
            for projection in [
                "linear_attn.in_proj_qkv",
                "linear_attn.in_proj_z",
                "linear_attn.out_proj",
            ] {
                module_paths.push((format!("model.layers.{layer_index}.{projection}"), false));
            }
        }
        // Every decoder layer carries the same dense MLP regardless of its
        // attention kind, matching the shipped dense pack.
        for projection in ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"] {
            module_paths.push((format!("model.layers.{layer_index}.{projection}"), false));
        }
    }
    module_paths
}

fn fixture_layer_types() -> Vec<String> {
    (0..FIXTURE_NUM_LAYERS)
        .map(|layer_index| {
            if layer_index == FIXTURE_FULL_ATTENTION_LAYER_INDEX {
                "full_attention"
            } else {
                "linear_attention"
            }
            .to_owned()
        })
        .collect()
}

/// The logical `[rows, input_width]` of each fixture packed module.
fn fixture_module_dimensions(module_path: &str) -> (u32, u32) {
    let gdn_value_width = FIXTURE_GDN_VALUE_HEADS * FIXTURE_GDN_VALUE_HEAD_DIM;
    let attention_value_width = FIXTURE_ATTENTION_HEADS * FIXTURE_HEAD_DIM;
    match module_path {
        "lm_head" | "model.embed_tokens" => (FIXTURE_VOCAB_SIZE, FIXTURE_HIDDEN_SIZE),
        path if path.ends_with("self_attn.q_proj") => (
            2 * FIXTURE_ATTENTION_HEADS * FIXTURE_HEAD_DIM,
            FIXTURE_HIDDEN_SIZE,
        ),
        path if path.ends_with("self_attn.k_proj") => {
            (FIXTURE_KV_HEADS * FIXTURE_HEAD_DIM, FIXTURE_HIDDEN_SIZE)
        }
        path if path.ends_with("self_attn.v_proj") => {
            (FIXTURE_KV_HEADS * FIXTURE_HEAD_DIM, FIXTURE_HIDDEN_SIZE)
        }
        path if path.ends_with("self_attn.o_proj") => (FIXTURE_HIDDEN_SIZE, attention_value_width),
        path if path.ends_with("linear_attn.in_proj_qkv") => (
            2 * FIXTURE_GDN_KEY_HEADS * FIXTURE_GDN_KEY_HEAD_DIM + gdn_value_width,
            FIXTURE_HIDDEN_SIZE,
        ),
        path if path.ends_with("linear_attn.in_proj_z") => (gdn_value_width, FIXTURE_HIDDEN_SIZE),
        path if path.ends_with("linear_attn.out_proj") => (FIXTURE_HIDDEN_SIZE, gdn_value_width),
        path if path.ends_with("mlp.gate_proj") || path.ends_with("mlp.up_proj") => {
            (FIXTURE_INTERMEDIATE_SIZE, FIXTURE_HIDDEN_SIZE)
        }
        path if path.ends_with("mlp.down_proj") => (FIXTURE_HIDDEN_SIZE, FIXTURE_INTERMEDIATE_SIZE),
        _ => panic!("unmapped fixture module path {module_path}"),
    }
}

/// The auxiliary language tensors required by the fixture shape contract.
fn fixture_auxiliary_tensors() -> Vec<(String, String, Vec<usize>)> {
    let mut auxiliary_tensors = vec![(
        "language_model.model.norm.weight".to_owned(),
        "F32".to_owned(),
        vec![FIXTURE_HIDDEN_SIZE as usize],
    )];
    for layer_index in 0..FIXTURE_NUM_LAYERS {
        let layer_prefix = format!("language_model.model.layers.{layer_index}.");
        auxiliary_tensors.push((
            format!("{layer_prefix}input_layernorm.weight"),
            "F32".to_owned(),
            vec![FIXTURE_HIDDEN_SIZE as usize],
        ));
        auxiliary_tensors.push((
            format!("{layer_prefix}post_attention_layernorm.weight"),
            "F32".to_owned(),
            vec![FIXTURE_HIDDEN_SIZE as usize],
        ));
        if layer_index == FIXTURE_FULL_ATTENTION_LAYER_INDEX {
            auxiliary_tensors.push((
                format!("{layer_prefix}self_attn.q_norm.weight"),
                "F32".to_owned(),
                vec![FIXTURE_HEAD_DIM as usize],
            ));
            auxiliary_tensors.push((
                format!("{layer_prefix}self_attn.k_norm.weight"),
                "F32".to_owned(),
                vec![FIXTURE_HEAD_DIM as usize],
            ));
        } else {
            auxiliary_tensors.push((
                format!("{layer_prefix}linear_attn.A_log"),
                "F32".to_owned(),
                vec![FIXTURE_GDN_VALUE_HEADS as usize],
            ));
            auxiliary_tensors.push((
                format!("{layer_prefix}linear_attn.dt_bias"),
                "F32".to_owned(),
                vec![FIXTURE_GDN_VALUE_HEADS as usize],
            ));
            auxiliary_tensors.push((
                format!("{layer_prefix}linear_attn.conv1d.weight"),
                "F32".to_owned(),
                vec![
                    (2 * FIXTURE_GDN_KEY_HEADS * FIXTURE_GDN_KEY_HEAD_DIM
                        + FIXTURE_GDN_VALUE_HEADS * FIXTURE_GDN_VALUE_HEAD_DIM)
                        as usize,
                    4,
                    1,
                ],
            ));
            auxiliary_tensors.push((
                format!("{layer_prefix}linear_attn.in_proj_a.weight"),
                "F32".to_owned(),
                vec![
                    FIXTURE_GDN_VALUE_HEADS as usize,
                    FIXTURE_HIDDEN_SIZE as usize,
                ],
            ));
            auxiliary_tensors.push((
                format!("{layer_prefix}linear_attn.in_proj_b.weight"),
                "F32".to_owned(),
                vec![
                    FIXTURE_GDN_VALUE_HEADS as usize,
                    FIXTURE_HIDDEN_SIZE as usize,
                ],
            ));
            auxiliary_tensors.push((
                format!("{layer_prefix}linear_attn.norm.weight"),
                "F32".to_owned(),
                vec![FIXTURE_GDN_KEY_HEAD_DIM as usize],
            ));
        }
    }
    auxiliary_tensors
}

/// The fixture config.json document.
pub(crate) fn fixture_config_json() -> Value {
    let modules = fixture_packed_module_paths()
        .into_iter()
        .map(|(path, embedding)| {
            json!({"path": path, "block": FIXTURE_HADAMARD_BLOCK, "embedding": embedding, "dtype": "float16"})
        })
        .collect::<Vec<_>>();
    json!({
        "schema_version": 2,
        "model_type": "prism_hadamard_qwen35",
        "base_model_type": "qwen3_5",
        "requires_runtime": "runtime/artifact.py",
        "hadamard_config": "hadamard.json",
        "tensor_namespace": "mlx-vlm-qwen3_5",
        "gdn_activation_layout": "grouped",
        "components": {"text": true, "vision": false, "mtp": false},
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_act": "silu",
            "hidden_size": FIXTURE_HIDDEN_SIZE,
            "intermediate_size": FIXTURE_INTERMEDIATE_SIZE,
            "num_hidden_layers": FIXTURE_NUM_LAYERS,
            "num_attention_heads": FIXTURE_ATTENTION_HEADS,
            "num_key_value_heads": FIXTURE_KV_HEADS,
            "head_dim": FIXTURE_HEAD_DIM,
            "vocab_size": FIXTURE_VOCAB_SIZE,
            "max_position_embeddings": 4096,
            "full_attention_interval": 4,
            "layer_types": fixture_layer_types(),
            "linear_num_value_heads": FIXTURE_GDN_VALUE_HEADS,
            "linear_num_key_heads": FIXTURE_GDN_KEY_HEADS,
            "linear_key_head_dim": FIXTURE_GDN_KEY_HEAD_DIM,
            "linear_value_head_dim": FIXTURE_GDN_VALUE_HEAD_DIM,
            "linear_conv_kernel_dim": 4,
            "rms_norm_eps": 0.000001,
            "rope_parameters": {
                "type": "default",
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25,
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10],
            },
            "eos_token_id": 248044,
            "attention_bias": false,
            "tie_word_embeddings": false,
            "attn_output_gate": true,
            "output_gate_type": "swish",
            "mamba_ssm_dtype": "float32",
        },
        "quantization": {"bits": 2, "group_size": 128, "mode": "affine"},
        "modules": modules,
        "tie_word_embeddings": false,
    })
}

/// The fixture hadamard.json manifest matching the fixture modules exactly.
pub(crate) fn fixture_hadamard_json() -> Value {
    let mut rotated_names = Vec::new();
    let mut inverse_names = Vec::new();
    for (path, embedding) in fixture_packed_module_paths() {
        let weight_name = format!("language_model.{path}.weight");
        if embedding {
            inverse_names.push(weight_name);
        } else {
            rotated_names.push(weight_name);
        }
    }
    // Distinct sign widths only: the grouped GDN value width equals the
    // intermediate width in this fixture, so both share one manifest entry.
    let sign_values = (0..FIXTURE_HIDDEN_SIZE)
        .map(|index| if index % 2 == 0 { 1 } else { -1 })
        .chain((0..FIXTURE_INTERMEDIATE_SIZE).map(|index| if index % 3 == 0 { -1 } else { 1 }))
        .collect::<Vec<_>>();
    json!({
        "prism.hadamard.version": 1,
        "prism.hadamard.block_size": FIXTURE_HADAMARD_BLOCK,
        "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
        "prism.hadamard.axis": "input-last-dimension",
        "prism.hadamard.sign_mode": "explicit",
        "prism.hadamard.weight_names": rotated_names,
        "prism.hadamard.inverse_weight_names": inverse_names,
        "prism.hadamard.sign_widths": [
            FIXTURE_HIDDEN_SIZE,
            FIXTURE_INTERMEDIATE_SIZE
        ],
        "prism.hadamard.sign_values": sign_values,
        "prism.hadamard.gdn_v_grouped": true,
    })
}

/// Byte length of one safetensors tensor payload from its dtype and shape.
fn payload_byte_length(dtype: &str, shape: &[usize]) -> usize {
    let element_sizes: BTreeMap<&str, usize> = BTreeMap::from([("U32", 4), ("F16", 2), ("F32", 4)]);
    shape.iter().product::<usize>() * element_sizes[dtype]
}

/// Builds the safetensors file bytes for the fixture tensor set.
pub(crate) fn fixture_safetensors_bytes(
    tensor_declarations: &[(String, String, Vec<usize>)],
) -> Vec<u8> {
    let mut header_entries = serde_json::Map::new();
    let mut payload_offset = 0_usize;
    let mut payloads = Vec::new();
    for (tensor_name, dtype, shape) in tensor_declarations {
        let payload_bytes = payload_byte_length(dtype, shape);
        header_entries.insert(
            tensor_name.clone(),
            json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [payload_offset, payload_offset + payload_bytes],
            }),
        );
        payloads.extend(std::iter::repeat_n(0_u8, payload_bytes));
        payload_offset += payload_bytes;
    }
    let header_bytes = Value::Object(header_entries).to_string().into_bytes();
    let mut safetensors_bytes = (header_bytes.len() as u64).to_le_bytes().to_vec();
    safetensors_bytes.extend(header_bytes);
    safetensors_bytes.extend(payloads);
    safetensors_bytes
}

/// The complete fixture tensor declaration list (packed modules + auxiliaries).
pub(crate) fn fixture_tensor_declarations() -> Vec<(String, String, Vec<usize>)> {
    let mut tensor_declarations = Vec::new();
    for (path, _) in fixture_packed_module_paths() {
        let (rows, input_width) = fixture_module_dimensions(&path);
        let stem = format!("language_model.{path}");
        tensor_declarations.push((
            format!("{stem}.weight"),
            "U32".to_owned(),
            vec![rows as usize, (input_width / 16) as usize],
        ));
        tensor_declarations.push((
            format!("{stem}.scales"),
            "F16".to_owned(),
            vec![rows as usize, (input_width / 128) as usize],
        ));
        tensor_declarations.push((
            format!("{stem}.biases"),
            "F16".to_owned(),
            vec![rows as usize, (input_width / 128) as usize],
        ));
        tensor_declarations.push((
            format!("{stem}.signs"),
            "F32".to_owned(),
            vec![input_width as usize],
        ));
    }
    for (tensor_name, dtype, shape) in fixture_auxiliary_tensors() {
        tensor_declarations.push((tensor_name, dtype, shape));
    }
    tensor_declarations
}

/// Distinguishes concurrently created fixture directories. Wall-clock time
/// alone is unsafe here: the test harness runs tests in parallel on one clock
/// whose granularity can exceed one nanosecond, so two timestamps can agree
/// and the second `create_dir_all` would silently reuse the first test's
/// live directory and clobber its files mid-validation.
static FIXTURE_DIRECTORY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Writes the complete fixture pack into a fresh temporary directory and
/// returns its path.
pub(crate) fn write_fixture_pack(mutation: impl FnOnce(&mut FixturePackMutation)) -> PathBuf {
    let fixture_directory = std::env::temp_dir().join(format!(
        "prism-hadamard-fixture-{}-{}-{}",
        std::process::id(),
        FIXTURE_DIRECTORY_SEQUENCE.fetch_add(1, Ordering::Relaxed),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock should advance")
            .as_nanos()
    ));
    // Exclusive creation turns any residual name collision into a loud
    // failure instead of silent cross-test interference.
    fs::create_dir(&fixture_directory).expect("the fixture directory should be created");

    let mut mutation_state = FixturePackMutation {
        config: fixture_config_json(),
        hadamard: fixture_hadamard_json(),
        tensor_declarations: fixture_tensor_declarations(),
    };
    mutation(&mut mutation_state);

    fs::write(
        fixture_directory.join("config.json"),
        mutation_state.config.to_string(),
    )
    .expect("the fixture config should be written");
    fs::write(
        fixture_directory.join("hadamard.json"),
        mutation_state.hadamard.to_string(),
    )
    .expect("the fixture manifest should be written");
    fs::write(fixture_directory.join("tokenizer.json"), "{}")
        .expect("the fixture tokenizer stub should be written");
    fs::write(
        fixture_directory.join("model.safetensors"),
        fixture_safetensors_bytes(&mutation_state.tensor_declarations),
    )
    .expect("the fixture weights should be written");
    fixture_directory
}

/// Convenience wrapper validating against the default unmutated fixture.
pub(crate) fn fixture_pack_directory() -> PathBuf {
    write_fixture_pack(|_| {})
}

/// The mutation surface tests use to produce invalid packs.
pub(crate) struct FixturePackMutation {
    pub(crate) config: Value,
    pub(crate) hadamard: Value,
    pub(crate) tensor_declarations: Vec<(String, String, Vec<usize>)>,
}

impl FixturePackMutation {
    /// Drops one tensor declaration from the weights file.
    pub(crate) fn drop_tensor(&mut self, tensor_name: &str) {
        self.tensor_declarations
            .retain(|(name, _, _)| name != tensor_name);
    }

    /// Rewrites one tensor declaration's dtype.
    pub(crate) fn retype_tensor(&mut self, tensor_name: &str, dtype: &str) {
        for (name, declared_dtype, _) in &mut self.tensor_declarations {
            if name == tensor_name {
                *declared_dtype = dtype.to_owned();
            }
        }
    }

    /// Rewrites one tensor declaration's shape (payload length follows).
    pub(crate) fn reshape_tensor(&mut self, tensor_name: &str, shape: Vec<usize>) {
        for (name, _, declared_shape) in &mut self.tensor_declarations {
            if name == tensor_name {
                *declared_shape = shape.clone();
            }
        }
    }
}

/// Removes a directory tree (test fixture cleanup).
pub(crate) fn remove_fixture_directory(directory: &Path) {
    let _ = fs::remove_dir_all(directory);
}
