use astronomical_model_serving::{
    Flux2KleinConfigError, Flux2KleinOfficialProfile, Flux2KleinPipelineConfig,
    Flux2KleinSchedulerConfig, Flux2KleinTextEncoderConfig, Flux2KleinTransformerConfig,
    Flux2KleinVaeConfig,
};

use super::support::{
    model_index_json, scheduler_config_json, text_encoder_config_json, transformer_config_json,
    vae_config_json,
};

#[test]
fn should_parse_the_strict_official_distilled_4b_profile() {
    let pipeline = Flux2KleinPipelineConfig::parse(&model_index_json())
        .expect("the official pipeline contract should parse");
    let text_encoder = Flux2KleinTextEncoderConfig::parse(&text_encoder_config_json())
        .expect("the official Qwen3 4B contract should parse");
    let transformer = Flux2KleinTransformerConfig::parse(&transformer_config_json())
        .expect("the official transformer contract should parse");
    let vae = Flux2KleinVaeConfig::parse(&vae_config_json())
        .expect("the official VAE contract should parse");
    let scheduler = Flux2KleinSchedulerConfig::parse(&scheduler_config_json())
        .expect("the official scheduler contract should parse");

    assert!(pipeline.is_distilled());
    assert_eq!(text_encoder.hidden_state_taps(), &[9, 18, 27]);
    assert_eq!(text_encoder.conditioning_width(), 7_680);
    assert_eq!(transformer.double_stream_block_count(), 5);
    assert_eq!(transformer.single_stream_block_count(), 20);
    assert_eq!(transformer.hidden_width(), 3_072);
    assert_eq!(transformer.feed_forward_width(), 9_216);
    assert_eq!(vae.latent_channel_count(), 32);
    assert_eq!(scheduler.inference_step_count(), 4);
    assert_eq!(Flux2KleinOfficialProfile::guidance_thousandths(), 1_000);
}

#[test]
fn should_reject_unknown_or_changed_official_configuration_fields() {
    let mut unknown_field = serde_json::from_slice::<serde_json::Value>(&transformer_config_json())
        .expect("the fixture should be JSON");
    unknown_field["future_field"] = serde_json::json!(true);
    let unknown_bytes = serde_json::to_vec(&unknown_field).expect("the fixture should serialize");
    assert!(matches!(
        Flux2KleinTransformerConfig::parse(&unknown_bytes),
        Err(Flux2KleinConfigError::Malformed { .. })
    ));

    let mut wrong_geometry =
        serde_json::from_slice::<serde_json::Value>(&transformer_config_json())
            .expect("the fixture should be JSON");
    wrong_geometry["num_layers"] = serde_json::json!(6);
    let wrong_bytes = serde_json::to_vec(&wrong_geometry).expect("the fixture should serialize");
    assert!(matches!(
        Flux2KleinTransformerConfig::parse(&wrong_bytes),
        Err(Flux2KleinConfigError::UnsupportedProfile { .. })
    ));

    let mut wrong_dtype = serde_json::from_slice::<serde_json::Value>(&text_encoder_config_json())
        .expect("the text encoder fixture should be JSON");
    wrong_dtype["dtype"] = serde_json::json!("float16");
    let wrong_dtype_bytes =
        serde_json::to_vec(&wrong_dtype).expect("the wrong dtype fixture should serialize");
    assert!(matches!(
        Flux2KleinTextEncoderConfig::parse(&wrong_dtype_bytes),
        Err(Flux2KleinConfigError::UnsupportedProfile {
            document: "text_encoder/config.json",
            field: "dtype",
        })
    ));
}
