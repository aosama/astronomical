//! Unit contracts for reading a LoRA adapter's settings: the plain form is
//! read, and every variant the merge formula does not describe is refused.

use serde_json::json;

use super::plain_lora_rank_and_alpha;

fn plain_adapter_config() -> serde_json::Value {
    json!({
        "peft_type": "LORA",
        "r": 16,
        "lora_alpha": 32,
        "use_rslora": false,
        "use_dora": false,
        "fan_in_fan_out": false,
        "bias": "none",
        "rank_pattern": {},
        "alpha_pattern": {}
    })
}

#[test]
fn should_read_the_rank_and_alpha_of_a_plain_adapter() {
    assert_eq!(
        plain_lora_rank_and_alpha(&plain_adapter_config()),
        Ok((16, 32.0))
    );
}

#[test]
fn should_refuse_every_variant_that_changes_the_merge() {
    for (setting_name, setting_value) in [
        ("use_rslora", json!(true)),
        ("use_dora", json!(true)),
        ("fan_in_fan_out", json!(true)),
        ("bias", json!("lora_only")),
        ("rank_pattern", json!({"q_proj": 8})),
        ("alpha_pattern", json!({"q_proj": 4})),
        ("peft_type", json!("IA3")),
    ] {
        let mut adapter_config = plain_adapter_config();
        adapter_config[setting_name] = setting_value;

        assert!(
            plain_lora_rank_and_alpha(&adapter_config).is_err(),
            "{setting_name} must be refused"
        );
    }
}

#[test]
fn should_refuse_a_missing_or_zero_rank_or_alpha() {
    for (setting_name, setting_value) in [
        ("r", json!(0)),
        ("lora_alpha", json!(0)),
        ("r", json!(null)),
    ] {
        let mut adapter_config = plain_adapter_config();
        adapter_config[setting_name] = setting_value;

        assert!(plain_lora_rank_and_alpha(&adapter_config).is_err());
    }
}
