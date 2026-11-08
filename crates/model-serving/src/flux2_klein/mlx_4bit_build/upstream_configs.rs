//! The configuration the 4-bit build does not ship.
//!
//! mflux quantized the BF16 4B pipeline without copying its Diffusers
//! configuration documents. These are the reviewed BF16 documents, embedded and
//! parsed by the same strict parsers the BF16 build uses; the tensor profile
//! check then proves the 4-bit weights have exactly the shapes they describe.

use crate::flux2_klein::artifact::Flux2KleinComponentConfigs;
use crate::flux2_klein::{
    Flux2KleinConfigError, Flux2KleinPipelineConfig, Flux2KleinSchedulerConfig,
    Flux2KleinTextEncoderConfig, Flux2KleinTransformerConfig, Flux2KleinVaeConfig,
};

const PIPELINE_INDEX_JSON: &str = include_str!("upstream_configs/model_index.json");
const SCHEDULER_CONFIG_JSON: &str = include_str!("upstream_configs/scheduler_config.json");
const TEXT_ENCODER_CONFIG_JSON: &str = include_str!("upstream_configs/text_encoder_config.json");
const TRANSFORMER_CONFIG_JSON: &str = include_str!("upstream_configs/transformer_config.json");
const VAE_CONFIG_JSON: &str = include_str!("upstream_configs/vae_config.json");

/// The upstream configuration of FLUX.2 Klein 4B, parsed.
pub(super) fn parse_upstream_configs() -> Result<Flux2KleinComponentConfigs, Flux2KleinConfigError>
{
    Ok(Flux2KleinComponentConfigs {
        pipeline: Flux2KleinPipelineConfig::parse(PIPELINE_INDEX_JSON.as_bytes())?,
        scheduler: Flux2KleinSchedulerConfig::parse(SCHEDULER_CONFIG_JSON.as_bytes())?,
        text_encoder: Flux2KleinTextEncoderConfig::parse(TEXT_ENCODER_CONFIG_JSON.as_bytes())?,
        transformer: Flux2KleinTransformerConfig::parse(TRANSFORMER_CONFIG_JSON.as_bytes())?,
        vae: Flux2KleinVaeConfig::parse(VAE_CONFIG_JSON.as_bytes())?,
    })
}

#[cfg(test)]
mod tests {
    use super::parse_upstream_configs;

    #[test]
    fn should_parse_every_embedded_upstream_config_as_klein_4b() {
        let configs = parse_upstream_configs().expect("the embedded configs should parse");
        assert_eq!(configs.transformer.double_stream_block_count(), 5);
        assert_eq!(configs.transformer.single_stream_block_count(), 20);
        assert_eq!(configs.text_encoder.conditioning_width(), 7_680);
        assert!(configs.pipeline.is_distilled());
        assert_eq!(configs.scheduler.inference_step_count(), 4);
        assert_eq!(configs.vae.latent_channel_count(), 32);
    }
}
