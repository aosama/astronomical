use super::register_mtp_sidecar;
use crate::requantize::MtpSidecarSummary;

fn registered(config_text: &str) -> serde_json::Value {
    let directory = tempfile::tempdir().expect("a temporary directory should be creatable");
    let config_path = directory.path().join("config.json");
    std::fs::write(&config_path, config_text).expect("the config should be writable");
    register_mtp_sidecar(
        &config_path,
        &MtpSidecarSummary {
            tensor_count: 29,
            bits: 4,
            group_size: 64,
            layer_count: 1,
        },
    )
    .expect("the head should register");
    let registered_text =
        std::fs::read_to_string(&config_path).expect("the config should be readable");
    serde_json::from_str(&registered_text).expect("the config should parse")
}

#[test]
fn should_record_where_the_head_is_and_how_it_was_quantized() {
    let config = registered(r#"{"model_type": "qwen3_5"}"#);

    assert_eq!(config["mtp_file"], "optiq/mtp.safetensors");
    assert_eq!(config["mtp_tensor_count"], 29);
    assert_eq!(config["mtp_policy"], "optiq-int4-prequantized-gs64");
    assert_eq!(
        config["mtplx_mtp_quantization"],
        serde_json::json!({
            "bits": 4, "group_size": 64, "mode": "affine",
            "policy": "cyankiwi", "prequantized": true
        })
    );
    assert_eq!(
        config["mlx_lm_extra_tensors"]["mtp_file"],
        "optiq/mtp.safetensors"
    );
    assert_eq!(config["mtp_num_hidden_layers"], 1);
    assert_eq!(config["model_type"], "qwen3_5");
}

#[test]
fn should_keep_the_layer_count_and_extra_tensors_the_source_declared() {
    let config = registered(
        r#"{"mtp_num_hidden_layers": 3, "mlx_lm_extra_tensors": {"other_file": "x.safetensors"}}"#,
    );

    assert_eq!(config["mtp_num_hidden_layers"], 3);
    assert_eq!(
        config["mlx_lm_extra_tensors"]["other_file"],
        "x.safetensors"
    );
    assert_eq!(
        config["mlx_lm_extra_tensors"]["mtp_file"],
        "optiq/mtp.safetensors"
    );
}

#[test]
fn should_replace_an_extra_tensors_entry_that_is_not_an_object() {
    let config = registered(r#"{"mlx_lm_extra_tensors": "stale"}"#);

    assert_eq!(
        config["mlx_lm_extra_tensors"],
        serde_json::json!({"mtp_file": "optiq/mtp.safetensors"})
    );
}

#[test]
fn should_reject_a_config_that_is_not_json() {
    let directory = tempfile::tempdir().expect("a temporary directory should be creatable");
    let config_path = directory.path().join("config.json");
    std::fs::write(&config_path, "not json").expect("the config should be writable");

    let registration = register_mtp_sidecar(
        &config_path,
        &MtpSidecarSummary {
            tensor_count: 1,
            bits: 4,
            group_size: 64,
            layer_count: 1,
        },
    );

    assert!(matches!(
        registration,
        Err(crate::error::QuantizationError::ModelConfigUnreadable { .. })
    ));
}
