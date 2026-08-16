//! Shared synthetic Laguna page artifact used by pager and model journey tests.

use std::collections::BTreeMap;
use std::fs;

use astronomical_model_serving::{
    LagunaArtifactValidator, LagunaExpertPagingPlan, PerformanceAttribution,
};
use astronomical_runtime_integration::{MlxArray, MlxMemoryLimits, MlxRuntime};
use serde_json::{Map, Value, json};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

const POOLSIDE_TEMPLATE: &str = r#"
{{- bos_token -}}
{%- set enable_thinking = enable_thinking | default(false) -%}
{%- set preserve_thinking = preserve_thinking | default(false) -%}
{%- set add_generation_prompt = add_generation_prompt | default(false) -%}
{%- set system_message = "Use the supplied play as the only source for literary analysis." -%}
{%- if messages and messages[0].role == "system" -%}
  {%- set system_message = messages[0].content -%}
  {%- set messages = messages[1:] -%}
{%- endif -%}
{%- if (system_message and system_message.strip()) or tools -%}
  {{- "<system>" -}}
  {{- system_message.rstrip() if system_message and system_message.strip() else "" -}}
  {%- if tools -%}
    {{- "\n\n<available_tools>\n" -}}
    {%- for tool in tools -%}
      {{- (tool | tojson) + "\n" -}}
    {%- endfor -%}
    {{- "</available_tools>" -}}
  {%- endif -%}
  {{- "</system>\n" -}}
{%- endif -%}
{%- for message in messages -%}
  {%- if message.role == "user" -%}
    {{- "<user>" + message.content + "</user>\n" -}}
  {%- elif message.role == "assistant" -%}
    {{- "<assistant>" -}}
    {%- if enable_thinking or preserve_thinking -%}
      {{- "<think>" + message.reasoning_content + "</think>" -}}
    {%- else -%}
      {{- "</think>" -}}
    {%- endif -%}
    {{- message.content if message.content else "" -}}
    {%- for tool_call in message.tool_calls -%}
      {{- "<tool_call>" + tool_call.function.name -}}
      {%- for argument_name, argument_value in tool_call.function.arguments.items() -%}
        {{- "<arg_key>" + argument_name + "</arg_key>" -}}
        {{- "<arg_value>" + argument_value + "</arg_value>" -}}
      {%- endfor -%}
      {{- "</tool_call>" -}}
    {%- endfor -%}
    {{- "</assistant>\n" -}}
  {%- elif message.role == "tool" -%}
    {{- "<tool_response>" + message.content + "</tool_response>\n" -}}
  {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
  {{- "<assistant>" -}}
  {{- "<think>" if enable_thinking else "</think>" -}}
{%- endif -%}
"#;

pub(super) fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("Laguna page test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

pub(super) fn filled(runtime: &MlxRuntime, shape: &[i32], fill: f32) -> MlxArray {
    let element_count = shape.iter().product::<i32>() as usize;
    runtime
        .array_from_f32(&vec![fill; element_count], shape)
        .expect("a filled tensor should be valid")
}

struct SyntheticTensor {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
    fill: f32,
}

impl SyntheticTensor {
    fn payload_bytes(&self) -> Vec<u8> {
        let element_count = self.shape.iter().product::<usize>();
        let mut payload_bytes = Vec::with_capacity(element_count * 4);
        for _ in 0..element_count {
            payload_bytes.extend_from_slice(&self.fill.to_le_bytes());
        }
        payload_bytes
    }
}

pub(super) fn write_sparse_artifact(model_directory: &std::path::Path, is_per_expert: bool) {
    let config = json!({
        "architectures": ["LagunaForCausalLM"],
        "model_type": "laguna",
        "vocab_size": 8,
        "hidden_size": 4,
        "intermediate_size": 6,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "max_position_embeddings": 32,
        "rms_norm_eps": 0.00001,
        "tie_word_embeddings": false,
        "torch_dtype": "float32",
        "bos_token_id": 1,
        "pad_token_id": 2,
        "eos_token_id": 3,
        "mlp_layer_types": ["sparse"],
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 4,
        "shared_expert_intermediate_size": 0,
        "rope_parameters": { "rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0 }
    });
    let mut tensors = vec![
        tensor("model.embed_tokens.weight", vec![8, 4], 0.01),
        tensor("model.norm.weight", vec![4], 1.0),
        tensor("lm_head.weight", vec![8, 4], 0.01),
        tensor("model.layers.0.input_layernorm.weight", vec![4], 1.0),
        tensor(
            "model.layers.0.post_attention_layernorm.weight",
            vec![4],
            1.0,
        ),
        tensor("model.layers.0.self_attn.q_proj.weight", vec![4, 4], 0.02),
        tensor("model.layers.0.self_attn.k_proj.weight", vec![2, 4], 0.02),
        tensor("model.layers.0.self_attn.v_proj.weight", vec![2, 4], 0.02),
        tensor("model.layers.0.self_attn.o_proj.weight", vec![4, 4], 0.02),
        tensor("model.layers.0.self_attn.q_norm.weight", vec![2], 1.0),
        tensor("model.layers.0.self_attn.k_norm.weight", vec![2], 1.0),
        tensor("model.layers.0.mlp.gate.weight", vec![2, 4], 0.03),
    ];
    if is_per_expert {
        for expert_index in 0..2 {
            let fill = 0.04 + expert_index as f32 * 0.01;
            tensors.push(tensor(
                &format!("model.layers.0.mlp.experts.{expert_index}.gate_proj.weight"),
                vec![4, 4],
                fill,
            ));
            tensors.push(tensor(
                &format!("model.layers.0.mlp.experts.{expert_index}.up_proj.weight"),
                vec![4, 4],
                fill,
            ));
            tensors.push(tensor(
                &format!("model.layers.0.mlp.experts.{expert_index}.down_proj.weight"),
                vec![4, 4],
                fill,
            ));
        }
    } else {
        tensors.push(tensor(
            "model.layers.0.mlp.experts.gate_proj.weight",
            vec![2, 4, 4],
            0.04,
        ));
        tensors.push(tensor(
            "model.layers.0.mlp.experts.up_proj.weight",
            vec![2, 4, 4],
            0.05,
        ));
        tensors.push(tensor(
            "model.layers.0.mlp.experts.down_proj.weight",
            vec![2, 4, 4],
            0.06,
        ));
    }
    write_text_sidecars(model_directory, &config);
    let mut tensors_by_shard: BTreeMap<String, Vec<SyntheticTensor>> = BTreeMap::from([
        ("model-00001-of-00002.safetensors".to_owned(), Vec::new()),
        ("model-00002-of-00002.safetensors".to_owned(), Vec::new()),
    ]);
    let mut weight_map = BTreeMap::new();
    for (tensor_position, tensor) in tensors.into_iter().enumerate() {
        let shard_file_name = if tensor_position.is_multiple_of(2) {
            "model-00001-of-00002.safetensors"
        } else {
            "model-00002-of-00002.safetensors"
        };
        weight_map.insert(tensor.name.clone(), shard_file_name.to_owned());
        tensors_by_shard
            .get_mut(shard_file_name)
            .expect("shard map")
            .push(tensor);
    }
    let mut total_shard_file_bytes = 0_u64;
    for (shard_file_name, shard_tensors) in &tensors_by_shard {
        let shard_bytes = safetensors_bytes(shard_tensors);
        total_shard_file_bytes += shard_bytes.len() as u64;
        fs::write(model_directory.join(shard_file_name), shard_bytes)
            .expect("the synthetic shard should be written");
    }
    let index = json!({
        "metadata": { "total_size": total_shard_file_bytes },
        "weight_map": weight_map,
    });
    fs::write(
        model_directory.join("model.safetensors.index.json"),
        serde_json::to_vec(&index).expect("index"),
    )
    .expect("index should be written");
    fs::write(
        model_directory.join("config.json"),
        serde_json::to_vec(&config).expect("config"),
    )
    .expect("config should be written");
}

fn tensor(name: &str, shape: Vec<usize>, fill: f32) -> SyntheticTensor {
    SyntheticTensor {
        name: name.to_owned(),
        dtype: "F32",
        shape,
        fill,
    }
}

fn safetensors_bytes(tensors: &[SyntheticTensor]) -> Vec<u8> {
    let mut payload_bytes = Vec::new();
    let mut tensor_entries = Vec::new();
    for tensor in tensors {
        let data_start_offset = payload_bytes.len();
        payload_bytes.extend_from_slice(&tensor.payload_bytes());
        tensor_entries.push(format!(
            "\"{}\":{{\"dtype\":\"{}\",\"shape\":{},\"data_offsets\":[{},{}]}}",
            tensor.name,
            tensor.dtype,
            serde_json::to_string(&tensor.shape).expect("shape"),
            data_start_offset,
            payload_bytes.len(),
        ));
    }
    let header = format!("{{{}}}", tensor_entries.join(","));
    let mut shard_bytes = Vec::new();
    shard_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    shard_bytes.extend_from_slice(header.as_bytes());
    shard_bytes.extend_from_slice(&payload_bytes);
    shard_bytes
}

fn write_text_sidecars(model_directory: &std::path::Path, config: &Value) {
    let vocabulary_size = 8_u32;
    let mut vocabulary = Map::new();
    for token_id in 0..vocabulary_size {
        vocabulary.insert(format!("token_{token_id}"), json!(token_id));
    }
    let token_descriptor = |token_id: u32, token_content: &str| {
        json!({
            "id": token_id,
            "content": token_content,
            "single_word": false,
            "lstrip": false,
            "rstrip": false,
            "normalized": false,
            "special": true
        })
    };
    let mut added_tokens_decoder = Map::new();
    let added_tokens = [
        (1_u32, "<synthetic_bos>"),
        (2, "<synthetic_pad>"),
        (3, "<synthetic_eos>"),
    ]
    .map(|(token_id, token_content)| {
        vocabulary.remove(&format!("token_{token_id}"));
        vocabulary.insert(token_content.to_owned(), json!(token_id));
        let descriptor = token_descriptor(token_id, token_content);
        added_tokens_decoder.insert(token_id.to_string(), descriptor.clone());
        descriptor
    });
    let tokenizer = json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": added_tokens,
        "normalizer": null,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocabulary, "unk_token": "token_0"}
    });
    let tokenizer_config = json!({
        "added_tokens_decoder": added_tokens_decoder,
        "bos_token": "<synthetic_bos>",
        "pad_token": "<synthetic_pad>",
        "eos_token": "<synthetic_eos>",
        "model_max_length": config["max_position_embeddings"],
        "chat_template": POOLSIDE_TEMPLATE
    });
    let generation_config = json!({
        "do_sample": true,
        "bos_token_id": 1,
        "pad_token_id": 2,
        "eos_token_id": [3],
        "temperature": 1.0,
        "top_p": 1.0,
        "reasoning_parser": "poolside_v1",
        "tool_call_parser": "poolside_v1",
        "default_chat_template_kwargs": {"enable_thinking": true}
    });
    for (file_name, document) in [
        ("tokenizer.json", tokenizer),
        ("tokenizer_config.json", tokenizer_config),
        ("generation_config.json", generation_config),
    ] {
        fs::write(
            model_directory.join(file_name),
            serde_json::to_vec(&document).expect("sidecar"),
        )
        .expect("sidecar should be written");
    }
}

pub(super) fn paging_plan(
    model_directory: &std::path::Path,
) -> (
    astronomical_model_serving::ValidatedLagunaArtifact,
    LagunaExpertPagingPlan,
) {
    let artifact = LagunaArtifactValidator::new()
        .validate(model_directory)
        .expect("the synthetic Laguna page artifact should validate");
    let mut performance_attribution = PerformanceAttribution::enabled();
    let plan = LagunaExpertPagingPlan::from_validated_artifact(
        &artifact,
        model_directory,
        &mut performance_attribution,
    )
    .expect("the validated artifact should build a paging plan");
    (artifact, plan)
}
