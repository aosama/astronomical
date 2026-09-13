//! Materializes a `qwen4_exp` shape specification into a compliant artifact
//! directory.
//!
//! The writer emits the configuration document, tokenizer and template
//! sidecars, safetensors shards at the variant's size ceiling, the weight
//! map, and a lookup manifest when the variant specifies one. Tensors carry
//! deterministic filler payloads — the geometry, names, and headers are what
//! the family's validators and forward paths consume, not the values, so a
//! few bytes per tensor keep generated fixtures kilobytes rather than
//! gigabytes.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::Path;

use super::shape_spec::{NgramStorageForm, Qwen4ExpShapeSpec, VariantQuantization};

/// One tensor written into a shard: name, wire dtype, shape, and filler bytes.
struct ShardTensor {
    tensor_name: String,
    dtype: &'static str,
    shape: Vec<usize>,
    payload_bytes: Vec<u8>,
}

/// What the writer produced, for tests that need the manifest or row count.
pub struct GeneratedVariant {
    pub model_directory: std::path::PathBuf,
    pub shard_count: usize,
    pub ngram_row_count: u64,
}

/// Writes one safetensors shard: header length, header, payload.
fn write_safetensors_file(file_path: &Path, tensors: &[ShardTensor]) {
    let mut payload_bytes = Vec::new();
    let mut header = serde_json::Map::new();
    for tensor in tensors {
        let payload_start = payload_bytes.len();
        payload_bytes.extend_from_slice(&tensor.payload_bytes);
        let payload_end = payload_bytes.len();
        header.insert(
            tensor.tensor_name.clone(),
            serde_json::json!({
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [payload_start, payload_end],
            }),
        );
    }
    let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header))
        .expect("the variant shard header should serialize");
    let mut shard_file = fs::File::create(file_path).expect("the variant shard should be created");
    shard_file
        .write_all(&(header_bytes.len() as u64).to_le_bytes())
        .expect("the variant shard should write its header length");
    shard_file
        .write_all(&header_bytes)
        .expect("the variant shard should write its header");
    shard_file
        .write_all(&payload_bytes)
        .expect("the variant shard should write its payload");
}

fn affine_dtype_and_payload(shape: &[usize], bits: u32) -> (&'static str, Vec<u8>) {
    let elements: usize = shape.iter().product();
    let packed_values = elements * bits as usize;
    let u32_columns = packed_values.div_ceil(32);
    let payload = vec![0xA5_u8; u32_columns * 4];
    ("U32", payload)
}

fn native_payload(shape: &[usize]) -> (&'static str, Vec<u8>) {
    let elements: usize = shape.iter().product();
    ("BF16", vec![0x3C_u8; elements * 2])
}

fn tensor_payload(shape: &[usize], quantization: VariantQuantization) -> (&'static str, Vec<u8>) {
    match quantization {
        VariantQuantization::None => native_payload(shape),
        VariantQuantization::Affine { bits, .. } => affine_dtype_and_payload(shape, bits),
        VariantQuantization::Mxfp4 => affine_dtype_and_payload(shape, 4),
    }
}

/// Builds the configuration document for one variant.
fn configuration_document(spec: &Qwen4ExpShapeSpec) -> serde_json::Value {
    let layer_types: Vec<&str> = spec
        .layer_schedule()
        .into_iter()
        .map(|full_attention| {
            if full_attention {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect();
    let mut text_config = serde_json::json!({
        "model_type": "qwen4_exp_text",
        "hidden_size": spec.hidden_size,
        "num_hidden_layers": spec.decoder_layers,
        "layer_types": layer_types,
        "full_attention_interval": spec.full_attention_interval,
        "head_dim": spec.head_dim,
        "num_attention_heads": spec.attention_heads,
        "num_key_value_heads": spec.key_value_heads,
        "vocab_size": spec.vocabulary_size,
        "max_position_embeddings": spec.context_window_tokens,
        "output_gate_type": "sigmoid",
        "rms_norm_eps": 1e-06,
        "dtype": "bfloat16",
        "eos_token_id": spec.eos_token_id,
        "num_experts": spec.routed_experts,
        "num_experts_per_tok": spec.expert_fan_out,
        "moe_intermediate_size": spec.moe_intermediate_size,
        "shared_expert_intermediate_size": spec.moe_intermediate_size,
        "ngram_size": 3,
        "heads_per_ngram": 2,
        "ngram_vocab_size_base": 1000,
        "split_ngram_parts": 4,
        "ple_embed_dim": spec.hidden_size,
        "ple_layer_ids": [2],
        "ple_conv_kernel_size": 4,
        "make_ngram_vocab_size_divisible_by": 8,
    });
    let text_object = text_config
        .as_object_mut()
        .expect("text config is an object");
    if spec.subsystems.linear_attention {
        text_object.insert("linear_num_key_heads".into(), serde_json::json!(2));
        text_object.insert("linear_num_value_heads".into(), serde_json::json!(4));
        text_object.insert("linear_key_head_dim".into(), serde_json::json!(4));
        text_object.insert("linear_value_head_dim".into(), serde_json::json!(4));
        text_object.insert("linear_conv_kernel_dim".into(), serde_json::json!(4));
        text_object.insert("mamba_ssm_dtype".into(), serde_json::json!("float32"));
    }
    if spec.subsystems.sparse_attention {
        text_object.insert("indexer_budget".into(), serde_json::json!(64));
        text_object.insert("indexer_compress_ratio".into(), serde_json::json!(4));
        text_object.insert("indexer_head_dim".into(), serde_json::json!(4));
        text_object.insert("indexer_kv_heads".into(), serde_json::json!(1));
        text_object.insert("indexer_n_heads".into(), serde_json::json!(2));
    }
    if spec.subsystems.hyper_connections {
        text_object.insert("hc_count".into(), serde_json::json!(2));
        text_object.insert("hc_lowrank".into(), serde_json::json!(8));
    }
    if spec.subsystems.multi_token_prediction_declared {
        text_object.insert("mtp_num_hidden_layers".into(), serde_json::json!(1));
    }
    let mut document = serde_json::json!({
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "text_config": text_config,
    });
    match spec.quantization {
        VariantQuantization::None => {}
        VariantQuantization::Affine { bits, group_size } => {
            document["quantization"] = serde_json::json!({
                "bits": bits, "group_size": group_size, "mode": "affine"
            });
        }
        VariantQuantization::Mxfp4 => {
            document["quantization"] = serde_json::json!({
                "bits": 4, "group_size": 32, "mode": "mxfp4"
            });
        }
    }
    document
}

/// The tensor inventory for one variant: every tensor name, its shard
/// assignment, and its shape, derived from the spec alone.
fn build_tensor_inventory(
    spec: &Qwen4ExpShapeSpec,
) -> (Vec<ShardTensor>, BTreeMap<String, String>, u64) {
    let mut tensors = Vec::new();
    let mut weight_map = BTreeMap::new();
    let mut shard_index = 0_usize;
    let mut current_shard_bytes = 0_usize;
    let mut shard_tensor_count = 0_usize;
    let mut shard_files: Vec<String> = Vec::new();
    let begin_shard = |shard_files: &mut Vec<String>, shard_index: &mut usize| {
        let name = format!("model-{shard_index:06}.safetensors");
        shard_files.push(name.clone());
        *shard_index += 1;
        name
    };
    let push_tensor = |tensors: &mut Vec<ShardTensor>,
                       weight_map: &mut BTreeMap<String, String>,
                       current_shard_bytes: &mut usize,
                       shard_tensor_count: &mut usize,
                       shard_files: &mut Vec<String>,
                       shard_index: &mut usize,
                       tensor_name: String,
                       dtype: &'static str,
                       shape: Vec<usize>,
                       payload_bytes: Vec<u8>| {
        if *current_shard_bytes + payload_bytes.len() > spec.maximum_shard_bytes
            || *shard_tensor_count == 0 && false
        {
            *current_shard_bytes = 0;
            *shard_tensor_count = 0;
            let name = begin_shard(shard_files, shard_index);
            weight_map.insert(tensor_name.clone(), name);
        } else if shard_files.is_empty() {
            let name = begin_shard(shard_files, shard_index);
            weight_map.insert(tensor_name.clone(), name);
        }
        *current_shard_bytes += payload_bytes.len();
        *shard_tensor_count += 1;
        tensors.push(ShardTensor {
            tensor_name,
            dtype,
            shape,
            payload_bytes,
        });
    };
    let quantization = spec.quantization;
    // Embeddings and output head.
    let (dtype, payload) = tensor_payload(
        &[
            spec.vocabulary_size as usize,
            spec.packed_columns(spec.hidden_size) as usize,
        ],
        quantization,
    );
    push_tensor(
        &mut tensors,
        &mut weight_map,
        &mut current_shard_bytes,
        &mut shard_tensor_count,
        &mut shard_files,
        &mut shard_index,
        "language_model.model.embed_tokens.weight".into(),
        dtype,
        vec![
            spec.vocabulary_size as usize,
            spec.packed_columns(spec.hidden_size) as usize,
        ],
        payload,
    );
    // Per-layer tensors.
    for layer_index in 0..spec.decoder_layers {
        let prefix = format!("language_model.model.layers.{layer_index}");
        let full_attention = spec.layer_schedule()[layer_index as usize];
        if full_attention {
            let packed = spec.packed_columns(spec.hidden_size) as usize;
            for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                let rows = match projection {
                    "q_proj" => spec.attention_heads * spec.head_dim * 2,
                    "k_proj" | "v_proj" => spec.key_value_heads * spec.head_dim,
                    _ => spec.hidden_size,
                };
                let (dtype, payload) = tensor_payload(&[rows as usize, packed], quantization);
                push_tensor(
                    &mut tensors,
                    &mut weight_map,
                    &mut current_shard_bytes,
                    &mut shard_tensor_count,
                    &mut shard_files,
                    &mut shard_index,
                    format!("{prefix}.self_attn.{projection}.weight"),
                    dtype,
                    vec![rows as usize, packed],
                    payload,
                );
            }
            if spec.subsystems.sparse_attention {
                let (dtype, payload) = if spec.indexer_projection_quantized {
                    tensor_payload(
                        &[640, spec.packed_columns(spec.hidden_size) as usize],
                        quantization,
                    )
                } else {
                    native_payload(&[640, spec.hidden_size as usize])
                };
                push_tensor(
                    &mut tensors,
                    &mut weight_map,
                    &mut current_shard_bytes,
                    &mut shard_tensor_count,
                    &mut shard_files,
                    &mut shard_index,
                    format!("{prefix}.self_attn.indexer.index_qk_proj.weight"),
                    dtype,
                    vec![640, spec.packed_columns(spec.hidden_size) as usize],
                    payload,
                );
            }
        } else if spec.subsystems.linear_attention {
            let packed = spec.packed_columns(spec.hidden_size) as usize;
            for projection in [
                "in_proj_qkv",
                "in_proj_z",
                "in_proj_a",
                "in_proj_b",
                "out_proj",
            ] {
                let (dtype, payload) = tensor_payload(&[64, packed], quantization);
                push_tensor(
                    &mut tensors,
                    &mut weight_map,
                    &mut current_shard_bytes,
                    &mut shard_tensor_count,
                    &mut shard_files,
                    &mut shard_index,
                    format!("{prefix}.linear_attn.{projection}.weight"),
                    dtype,
                    vec![64, packed],
                    payload,
                );
            }
        }
        if spec.subsystems.hyper_connections {
            let packed = spec.packed_columns(spec.hidden_size * 2) as usize;
            for site in ["attn_hyper_connection", "mlp_hyper_connection"] {
                let (dtype, payload) = tensor_payload(&[8, packed], quantization);
                push_tensor(
                    &mut tensors,
                    &mut weight_map,
                    &mut current_shard_bytes,
                    &mut shard_tensor_count,
                    &mut shard_files,
                    &mut shard_index,
                    format!("{prefix}.{site}.input_mix_weight_down.weight"),
                    dtype,
                    vec![8, packed],
                    payload,
                );
            }
        }
        // Expert bank: stacked switch_mlp projections.
        let packed_intermediate = spec.packed_columns(spec.moe_intermediate_size) as usize;
        for projection in ["gate_proj", "up_proj"] {
            let (dtype, payload) = tensor_payload(
                &[
                    spec.routed_experts as usize,
                    spec.moe_intermediate_size as usize,
                    spec.packed_columns(spec.hidden_size) as usize,
                ],
                quantization,
            );
            push_tensor(
                &mut tensors,
                &mut weight_map,
                &mut current_shard_bytes,
                &mut shard_tensor_count,
                &mut shard_files,
                &mut shard_index,
                format!("{prefix}.mlp.switch_mlp.{projection}.weight"),
                dtype,
                vec![
                    spec.routed_experts as usize,
                    spec.moe_intermediate_size as usize,
                    spec.packed_columns(spec.hidden_size) as usize,
                ],
                payload,
            );
        }
        let (dtype, payload) = tensor_payload(
            &[
                spec.routed_experts as usize,
                spec.hidden_size as usize,
                packed_intermediate,
            ],
            quantization,
        );
        push_tensor(
            &mut tensors,
            &mut weight_map,
            &mut current_shard_bytes,
            &mut shard_tensor_count,
            &mut shard_files,
            &mut shard_index,
            format!("{prefix}.mlp.switch_mlp.down_proj.weight"),
            dtype,
            vec![
                spec.routed_experts as usize,
                spec.hidden_size as usize,
                packed_intermediate,
            ],
            payload,
        );
        let (gate_dtype, gate_payload) =
            native_payload(&[spec.routed_experts as usize, spec.hidden_size as usize]);
        push_tensor(
            &mut tensors,
            &mut weight_map,
            &mut current_shard_bytes,
            &mut shard_tensor_count,
            &mut shard_files,
            &mut shard_index,
            format!("{prefix}.mlp.gate.weight"),
            gate_dtype,
            vec![spec.routed_experts as usize, spec.hidden_size as usize],
            gate_payload,
        );
    }
    // Lookup-table shards, in the variant's storage form.
    let (_, _, padded_rows) = spec
        .ngram_layout()
        .expect("variant lookup geometry derives");
    let rows_per_shard = padded_rows / u64::from(spec.ngram_shard_count);
    let without_biases = matches!(
        spec.ngram_storage,
        NgramStorageForm::InlineDotNamingWithoutBiases
    );
    for shard in 0..spec.ngram_shard_count {
        let prefix = match spec.ngram_storage {
            NgramStorageForm::InlineUnderscoreNaming => {
                format!(
                    "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}"
                )
            }
            _ => format!(
                "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.{shard}"
            ),
        };
        let (weight_dtype, weight_payload) =
            affine_dtype_and_payload(&[rows_per_shard as usize, 160], 4);
        tensors.push(ShardTensor {
            tensor_name: format!("{prefix}.weight"),
            dtype: weight_dtype,
            shape: vec![rows_per_shard as usize, 160],
            payload_bytes: weight_payload,
        });
        weight_map.insert(
            format!("{prefix}.weight"),
            format!("model-{shard_index:06}.safetensors"),
        );
        tensors.push(ShardTensor {
            tensor_name: format!("{prefix}.scales"),
            dtype: "BF16",
            shape: vec![rows_per_shard as usize, 5],
            payload_bytes: vec![0x3C_u8; rows_per_shard as usize * 10],
        });
        weight_map.insert(
            format!("{prefix}.scales"),
            format!("model-{shard_index:06}.safetensors"),
        );
        if !without_biases {
            tensors.push(ShardTensor {
                tensor_name: format!("{prefix}.biases"),
                dtype: "BF16",
                shape: vec![rows_per_shard as usize, 5],
                payload_bytes: vec![0x3C_u8; rows_per_shard as usize * 10],
            });
            weight_map.insert(
                format!("{prefix}.biases"),
                format!("model-{shard_index:06}.safetensors"),
            );
        }
        // One lookup shard per file keeps the manifest form addressable.
        shard_index += 1;
    }
    let _ = &mut shard_files;
    (tensors, weight_map, padded_rows)
}

/// Materializes one variant into `parent_directory/model_id`.
///
/// # Errors
/// When any file write fails.
pub fn generate_variant(
    parent_directory: &Path,
    spec: &Qwen4ExpShapeSpec,
) -> std::io::Result<GeneratedVariant> {
    let model_directory = parent_directory.join(spec.model_id);
    fs::create_dir_all(&model_directory)?;
    fs::write(
        model_directory.join("config.json"),
        serde_json::to_vec(&configuration_document(spec)).expect("variant config serializes"),
    )?;
    fs::write(
        model_directory.join("tokenizer.json"),
        br#"{"version":1,"model":{"type":"BPE"}}"#,
    )?;
    fs::write(
        model_directory.join("chat_template.jinja"),
        b"{{ bos_token }}{% for message in messages %}{{ message['content'] }}{% endfor %}",
    )?;
    let (tensors, weight_map, padded_rows) = build_tensor_inventory(spec);
    let total_size: u64 = tensors.iter().map(|t| t.payload_bytes.len() as u64).sum();
    // Group tensors by their assigned shard file and write each group.
    let per_shard: BTreeMap<String, Vec<ShardTensor>> =
        tensors
            .into_iter()
            .fold(BTreeMap::new(), |mut groups, tensor| {
                let shard_name = weight_map
                    .get(&tensor.tensor_name)
                    .cloned()
                    .unwrap_or_else(|| format!("model-{:06}.safetensors", 0));
                groups.entry(shard_name).or_default().push(tensor);
                groups
            });
    for (shard_name, shard_tensors) in &per_shard {
        write_safetensors_file(&model_directory.join(shard_name), shard_tensors.as_slice());
    }
    let weight_map_json = serde_json::json!({
        "metadata": { "total_size": total_size },
        "weight_map": weight_map,
    });
    fs::write(
        model_directory.join("model.safetensors.index.json"),
        serde_json::to_vec(&weight_map_json).expect("weight map serializes"),
    )?;
    if matches!(
        spec.ngram_storage,
        NgramStorageForm::InlineDotNamingWithManifest
    ) {
        fs::write(model_directory.join("ple-store.json"), br#"{"version":2}"#)?;
    }
    Ok(GeneratedVariant {
        shard_count: per_shard.len(),
        ngram_row_count: padded_rows,
        model_directory,
    })
}
