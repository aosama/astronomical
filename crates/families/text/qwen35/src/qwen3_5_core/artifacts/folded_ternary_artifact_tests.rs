//! The folded pack's configuration as the shared core reads it: the keys only
//! the pack uses are gone, and the base model's type and architecture stand.

use super::shared_core_config_bytes;

fn translated(config_document: &serde_json::Value) -> serde_json::Value {
    let config_bytes = shared_core_config_bytes(config_document.to_string().as_bytes())
        .expect("a folded pack extending qwen3_5 should translate");
    serde_json::from_slice(&config_bytes).expect("the translated config is JSON")
}

#[test]
fn should_hand_the_core_a_tied_pack_without_prisms_tied_output_marker() {
    let shared_config = translated(&serde_json::json!({
        "model_type": "prism_hadamard_qwen35",
        "base_model_type": "qwen3_5",
        "modules": [],
        "hadamard_config": "hadamard.json",
        "tensor_namespace": "mlx-vlm-qwen3_5",
        "components": {"text": true, "vision": true, "mtp": false},
        "tie_word_embeddings": true,
        "tied_output": true,
    }));
    assert!(shared_config.get("tied_output").is_none());
    assert!(shared_config.get("modules").is_none());
    assert_eq!(
        shared_config["tie_word_embeddings"],
        serde_json::json!(true)
    );
    assert_eq!(shared_config["model_type"], serde_json::json!("qwen3_5"));
}

#[test]
fn should_refuse_a_folded_pack_that_extends_another_base() {
    let refusal = shared_core_config_bytes(
        serde_json::json!({"model_type": "prism_hadamard_qwen35", "base_model_type": "llama"})
            .to_string()
            .as_bytes(),
    );
    assert!(refusal.is_err());
}

#[test]
fn should_refuse_a_config_that_is_not_an_object() {
    assert!(shared_core_config_bytes(b"[1, 2]").is_err());
}
