use serde::Deserialize;

/// Sampler tuning parameters specific to a Qwen3.5-MoE model instance.
///
/// These parameters control the top-k sampling strategy used during text
/// generation. Different model instances may require different top-k values
/// for optimal output quality.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3_5MoESamplerConfig {
    /// The certified top-k value for sampling. Must match the value used
    /// during model training and evaluation.
    pub certified_top_k: i32,
}

/// Discovers the sampler configuration from a model's `generation_config.json` bytes.
///
/// The Qwen3.5-MoE family typically uses `top_k: 20` in generation_config.json.
/// If the file is absent or does not contain a `top_k` field, the default of 20
/// is used.
pub fn discover_sampler_config(generation_config_bytes: Option<&[u8]>) -> Qwen3_5MoESamplerConfig {
    let certified_top_k = generation_config_bytes
        .and_then(|bytes| serde_json::from_slice::<GenerationConfig>(bytes).ok())
        .and_then(|gen_config| gen_config.top_k)
        .unwrap_or(20);
    Qwen3_5MoESamplerConfig { certified_top_k }
}

#[derive(Deserialize)]
struct GenerationConfig {
    #[serde(default)]
    top_k: Option<i32>,
}
