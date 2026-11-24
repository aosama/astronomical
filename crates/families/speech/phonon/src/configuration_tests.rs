use super::PhononConfiguration;

fn phonon_configuration_json(time_norm: &str) -> String {
    format!(
        r#"{{"flow_lm": {{"d_model": 512, "num_heads": 8, "num_layers": 6, "ldim": 32, "flow_depth": 6,
              "max_period": 10000.0, "time_rms_norm": "{time_norm}", "gelu": "erf"}},
            "mimi": {{"sample_rate": 24000, "frame_rate": 12.5, "n_filters": 64, "n_residual_layers": 1,
              "ratios": [6, 5, 4], "compress": 2, "transformer_num_heads": 8, "transformer_num_layers": 2,
              "transformer_context": 250, "transformer_max_period": 10000.0, "transformer_layer_scale": 0.01}},
            "lsd_decode_steps": 1, "eos_threshold": 4.0,
            "fuser": {{"sum": ["num_speakers", "duration_delta"], "prepend": ["transcript_in_segment", "speaker_wavs"], "cross": []}},
            "conditioners": [{{"type": "lut", "name": "num_speakers",
              "lut": {{"possible_values": ["0", "1", "2"]}}}},
              {{"type": "continuous", "name": "duration_delta",
              "continuous": {{"scale_factor": 1000.0, "dim": 128, "max_period": 10000.0}}}}]}}"#
    )
}

#[test]
fn should_read_the_published_phonon_shape() {
    let configuration =
        PhononConfiguration::parse(&phonon_configuration_json("rms")).expect("valid configuration");
    assert_eq!(configuration.flow.layer_count, 6);
    assert_eq!(configuration.mimi.upsampling_ratios, [6, 5, 4]);
    assert_eq!(configuration.speaker_count_values, ["0", "1", "2"]);
    assert_eq!(configuration.end_of_speech_threshold, 4.0);
}

// Boundary: 24 kHz at 12.5 frames a second is 1920 samples a frame.
#[test]
fn should_give_1920_samples_per_frame() {
    let configuration =
        PhononConfiguration::parse(&phonon_configuration_json("rms")).expect("valid configuration");
    assert_eq!(configuration.samples_per_frame(), 1920);
    assert_eq!(configuration.sample_rate_hertz(), 24_000);
}

#[test]
fn should_refuse_a_pocket_tts_time_norm() {
    let refusal = PhononConfiguration::parse(&phonon_configuration_json("rms_var"))
        .expect_err("rms_var is not ported");
    assert!(refusal.to_string().contains("time_rms_norm"), "{refusal}");
}

#[test]
fn should_refuse_text_that_is_not_json() {
    assert!(PhononConfiguration::parse("O Romeo, Romeo").is_err());
}

#[test]
fn should_read_the_duration_conditioner_and_layer_scale() {
    let configuration =
        PhononConfiguration::parse(&phonon_configuration_json("rms")).expect("valid configuration");
    assert_eq!(configuration.duration_conditioner.width, 128);
    assert_eq!(configuration.duration_conditioner.scale_factor, 1000.0);
    assert!(configuration.mimi.has_layer_scale);
}

// Negative: a fusion the port does not implement.
#[test]
fn should_refuse_a_checkpoint_that_cross_attends_to_its_conditions() {
    let cross_attending =
        phonon_configuration_json("rms").replace(r#""cross": []"#, r#""cross": ["speaker_wavs"]"#);
    assert!(PhononConfiguration::parse(&cross_attending).is_err());
}

// Negative: a zero window would never advance the Mimi decoder's blocks.
#[test]
fn should_refuse_a_zero_or_negative_count() {
    let zero_window = phonon_configuration_json("rms").replace(
        r#""transformer_context": 250"#,
        r#""transformer_context": 0"#,
    );
    let negative_heads =
        phonon_configuration_json("rms").replace(r#""num_heads": 8"#, r#""num_heads": -8"#);
    assert!(PhononConfiguration::parse(&zero_window).is_err());
    assert!(PhononConfiguration::parse(&negative_heads).is_err());
}

// Negative: a zero frame rate or a zero upsampling ratio would give no audio.
#[test]
fn should_refuse_a_zero_frame_rate_or_ratio() {
    let zero_frame_rate =
        phonon_configuration_json("rms").replace(r#""frame_rate": 12.5"#, r#""frame_rate": 0.0"#);
    let zero_ratio = phonon_configuration_json("rms")
        .replace(r#""ratios": [6, 5, 4]"#, r#""ratios": [6, 0, 4]"#);
    assert!(PhononConfiguration::parse(&zero_frame_rate).is_err());
    assert!(PhononConfiguration::parse(&zero_ratio).is_err());
}
