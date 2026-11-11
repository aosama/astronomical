use super::KokoroConfiguration;

fn kokoro_configuration_json(hidden_width: u64) -> String {
    format!(
        r#"{{"hidden_dim": {hidden_width}, "style_dim": 128, "n_layer": 3, "max_dur": 50,
            "istftnet": {{"gen_istft_n_fft": 20, "gen_istft_hop_size": 5, "upsample_initial_channel": 512,
              "upsample_rates": [10, 6], "upsample_kernel_sizes": [20, 12], "resblock_kernel_sizes": [3, 7, 11],
              "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]]}},
            "plbert": {{"hidden_size": 768, "num_attention_heads": 12, "num_hidden_layers": 12}},
            "vocab": {{"ˈ": 156, "h": 50, "ə": 83, ",": 3}}}}"#
    )
}

#[test]
fn should_read_the_vocabulary_and_albert_shape() {
    let configuration =
        KokoroConfiguration::parse(&kokoro_configuration_json(512)).expect("valid configuration");
    assert_eq!(configuration.albert_attention_head_count, 12);
    assert_eq!(configuration.albert_layer_count, 12);
}

#[test]
fn should_wrap_phonemes_in_start_and_end_markers_and_skip_unknown_ones() {
    let configuration =
        KokoroConfiguration::parse(&kokoro_configuration_json(512)).expect("valid configuration");
    assert_eq!(configuration.token_ids("hˈə,❓"), [0, 50, 156, 83, 3, 0]);
}

// Boundary: an empty utterance is still the two markers.
#[test]
fn should_give_only_the_markers_for_no_phonemes() {
    let configuration =
        KokoroConfiguration::parse(&kokoro_configuration_json(512)).expect("valid configuration");
    assert_eq!(configuration.token_ids(""), [0, 0]);
}

#[test]
fn should_refuse_a_checkpoint_of_another_shape() {
    let refusal =
        KokoroConfiguration::parse(&kokoro_configuration_json(256)).expect_err("wrong width");
    assert!(refusal.to_string().contains("hidden_dim"), "{refusal}");
}

#[test]
fn should_refuse_text_that_is_not_json() {
    assert!(KokoroConfiguration::parse("Romeo, Romeo").is_err());
}

// Negative: zero attention heads would divide ALBERT's width by zero.
#[test]
fn should_refuse_zero_attention_heads() {
    let zero_heads = kokoro_configuration_json(512).replace(
        r#""num_attention_heads": 12"#,
        r#""num_attention_heads": 0"#,
    );
    assert!(KokoroConfiguration::parse(&zero_heads).is_err());
}
