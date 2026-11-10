//! Unit contracts for the converted model's config: the fields the engine
//! reads are carried over, and nothing that names a local path survives.

use serde_json::json;

use super::{mlx_laya_config, without_local_paths};

fn checkpoint_settings() -> serde_json::Value {
    json!({
        "encoder": "example-org/example-encoder",
        "head_layers": 2,
        "max_len": 512,
        "head_max_len": 192,
        "max_prefixes": 6,
        "act_costs": {"escalate": 0.5},
        "temperature": [2.5, 1.0, 1.0],
        "model_name": "example-agent",
        "training": {
            "eval_data": "/fictional-home/fictional-person/datasets/eval.jsonl",
            "updates": 7
        }
    })
}

fn encoder_config() -> serde_json::Value {
    json!({
        "model_type": "modernbert",
        "hidden_size": 1024,
        "_name_or_path": "/fictional-home/fictional-person/models/encoder"
    })
}

#[test]
fn should_carry_the_settings_the_engine_reads_and_nest_the_encoder_config() {
    let model_config = mlx_laya_config(&checkpoint_settings(), &encoder_config())
        .expect("a complete checkpoint should produce a config");

    assert_eq!(model_config["model_type"], "laya");
    assert_eq!(model_config["encoder"]["hidden_size"], 1024);
    assert_eq!(model_config["head_layers"], 2);
    assert_eq!(model_config["max_len"], 512);
    assert_eq!(model_config["head_max_len"], 192);
    assert_eq!(model_config["temperature"], json!([2.5, 1.0, 1.0]));
    assert_eq!(model_config["model_name"], "example-agent");
    assert_eq!(
        model_config["quantization"],
        json!({"group_size": 32, "bits": 8})
    );
}

#[test]
fn should_carry_the_per_option_temperatures_when_the_checkpoint_has_them() {
    let mut settings_with_table = checkpoint_settings();
    settings_with_table["temperature_by_options"] = json!({"choice:2": 1.9, "noul:2": 1.98});

    let model_config = mlx_laya_config(&settings_with_table, &encoder_config())
        .expect("a checkpoint with a temperature table should produce a config");

    assert_eq!(model_config["temperature_by_options"]["choice:2"], 1.9);
}

#[test]
fn should_never_copy_the_training_record_or_any_local_path() {
    let model_config = mlx_laya_config(&checkpoint_settings(), &encoder_config())
        .expect("a complete checkpoint should produce a config");

    let config_text = model_config.to_string();
    assert!(model_config.get("training").is_none());
    assert!(model_config["encoder"].get("_name_or_path").is_none());
    assert!(
        !config_text.contains("fictional-home"),
        "no local path may reach the output: {config_text}"
    );
}

#[test]
fn should_refuse_a_checkpoint_without_three_numeric_temperatures() {
    let mut settings_without_temperatures = checkpoint_settings();
    settings_without_temperatures
        .as_object_mut()
        .expect("the settings are an object")
        .remove("temperature");
    let mut settings_with_two_temperatures = checkpoint_settings();
    settings_with_two_temperatures["temperature"] = json!([1.0, 1.0]);
    let mut settings_with_text_temperatures = checkpoint_settings();
    settings_with_text_temperatures["temperature"] = json!(["1", "1", "1"]);

    assert!(mlx_laya_config(&settings_without_temperatures, &encoder_config()).is_err());
    assert!(mlx_laya_config(&settings_with_two_temperatures, &encoder_config()).is_err());
    assert!(mlx_laya_config(&settings_with_text_temperatures, &encoder_config()).is_err());
}

#[test]
fn should_refuse_an_encoder_config_that_is_not_an_object() {
    assert!(mlx_laya_config(&checkpoint_settings(), &json!("modernbert")).is_err());
}

#[test]
fn should_remove_path_like_text_at_any_depth_and_keep_everything_else() {
    let scrubbed_document = without_local_paths(json!({
        "cls_token": "[CLS]",
        "model_max_length": 8192,
        "nested": {
            "cache_dir": "~/fictional-cache",
            "kept": "plain text",
            "windows": "C:\\fictional\\models"
        },
        "files": ["./vocabulary.txt", "../other.txt", "kept.txt", "file:///fictional/x"]
    }));

    assert_eq!(
        scrubbed_document,
        json!({
            "cls_token": "[CLS]",
            "model_max_length": 8192,
            "nested": {"kept": "plain text"},
            "files": ["kept.txt"]
        })
    );
}

#[test]
fn should_keep_text_that_only_contains_a_slash() {
    let scrubbed_document = without_local_paths(json!({"encoder": "example-org/example-encoder"}));

    assert_eq!(
        scrubbed_document,
        json!({"encoder": "example-org/example-encoder"})
    );
}
