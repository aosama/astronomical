use serde::Deserialize;

/// Sampler tuning parameters specific to a Qwen3.5 model instance.
///
/// These parameters control the top-k sampling strategy used during text
/// generation. Different model instances may require different top-k values
/// for optimal output quality.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3_5SamplerConfig {
    /// The model's default sampling temperature expressed in thousandths.
    pub temperature_thousandths: u16,
    /// The model's default nucleus-sampling probability expressed in thousandths.
    pub top_p_thousandths: u16,
    /// The model top-k value for sampling. Must match the value used
    /// during model training and evaluation.
    pub model_top_k: i32,
}

/// Discovers the sampler configuration from a model's `generation_config.json` bytes.
///
/// If a model has no valid generation configuration, product defaults are used.
pub fn discover_sampler_config(generation_config_bytes: Option<&[u8]>) -> Qwen3_5SamplerConfig {
    let model_generation_configuration = generation_config_bytes
        .and_then(|bytes| serde_json::from_slice::<GenerationConfig>(bytes).ok())
        .unwrap_or_default();
    Qwen3_5SamplerConfig {
        // Temperature zero is a valid model-provided highest-logit policy. Keep it distinct from an
        // absent or malformed field, which alone falls back to Astronomical's product default.
        temperature_thousandths: decimal_thousandths(
            model_generation_configuration.temperature,
            1_000,
            0,
            u16::MAX,
        ),
        top_p_thousandths: decimal_thousandths(model_generation_configuration.top_p, 950, 1, 1_000),
        model_top_k: model_generation_configuration
            .top_k
            .filter(|configured_top_k| *configured_top_k > 0)
            .unwrap_or(20),
    }
}

fn decimal_thousandths(
    configured_probability_or_temperature: Option<f64>,
    fallback_thousandths: u16,
    minimum_thousandths: u16,
    maximum_thousandths: u16,
) -> u16 {
    configured_probability_or_temperature
        .filter(|probability_or_temperature| {
            probability_or_temperature.is_finite() && *probability_or_temperature >= 0.0
        })
        .and_then(|probability_or_temperature| {
            let scaled_thousandths = (probability_or_temperature * 1_000.0).round();
            (scaled_thousandths >= f64::from(minimum_thousandths)
                && scaled_thousandths <= f64::from(maximum_thousandths))
            .then_some(scaled_thousandths as u16)
        })
        .unwrap_or(fallback_thousandths)
}

#[derive(Default, Deserialize)]
struct GenerationConfig {
    #[serde(default)]
    top_k: Option<i32>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
}
