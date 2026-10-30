//! Beside-source tests for the decomposed `Qwen3_5TextConfig` validation units.
//!
//! Text configs are built from JSON fixtures in the house style: one compact
//! valid baseline per feed-forward family, with per-test field overrides. The
//! precedence tests pin the original single-pass check order, which the
//! extracted units must reproduce exactly.

use serde_json::{Value, json};

use super::{
    Qwen3_5ConfigError, Qwen3_5FeedForwardArchitecture, Qwen3_5TextConfig,
    validate_attention_and_expert_head_counts, validate_boolean_flags_and_mamba_dtype,
    validate_exact_text_values, validate_layer_types, validate_linear_attention_shape_bounds,
    validate_numeric_fields_are_positive,
};

const BASELINE_DENSE_TEXT_CONFIG_JSON: &str = r#"
{
    "attention_bias": false,
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "intermediate_size": 512,
    "layer_types": ["linear_attention", "full_attention"],
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 32,
    "linear_num_key_heads": 8,
    "linear_num_value_heads": 16,
    "linear_value_head_dim": 32,
    "max_position_embeddings": 4096,
    "mlp_bias": false,
    "model_type": "qwen3_5_text",
    "norm_topk_prob": true,
    "num_attention_heads": 16,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "rms_norm_eps": 0.000001,
    "vocab_size": 1000
}
"#;

const BASELINE_MOE_TEXT_CONFIG_JSON: &str = r#"
{
    "attention_bias": false,
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "layer_types": ["linear_attention", "full_attention"],
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 32,
    "linear_num_key_heads": 8,
    "linear_num_value_heads": 16,
    "linear_value_head_dim": 32,
    "max_position_embeddings": 4096,
    "mlp_bias": false,
    "model_type": "qwen3_5_moe_text",
    "moe_intermediate_size": 512,
    "norm_topk_prob": true,
    "num_attention_heads": 16,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "rms_norm_eps": 0.000001,
    "shared_expert_intermediate_size": 512,
    "vocab_size": 1000
}
"#;

fn baseline_dense_text_config_value() -> Value {
    serde_json::from_str(BASELINE_DENSE_TEXT_CONFIG_JSON)
        .expect("the dense baseline text config should decode as JSON")
}

fn baseline_moe_text_config_value() -> Value {
    serde_json::from_str(BASELINE_MOE_TEXT_CONFIG_JSON)
        .expect("the MoE baseline text config should decode as JSON")
}

fn text_config_from_value(text_config_value: Value) -> Qwen3_5TextConfig {
    serde_json::from_value(text_config_value).expect("the text config should deserialize")
}

fn dense_text_config_with(field_overrides: &[(&str, Value)]) -> Qwen3_5TextConfig {
    let mut text_config_value = baseline_dense_text_config_value();
    for (field_name, field_value) in field_overrides {
        text_config_value[field_name] = field_value.clone();
    }
    text_config_from_value(text_config_value)
}

fn moe_text_config_with(field_overrides: &[(&str, Value)]) -> Qwen3_5TextConfig {
    let mut text_config_value = baseline_moe_text_config_value();
    for (field_name, field_value) in field_overrides {
        text_config_value[field_name] = field_value.clone();
    }
    text_config_from_value(text_config_value)
}

const MAXIMUM_MLX_SHAPE_DIMENSION: u64 = i32::MAX as u64;

#[test]
fn should_accept_the_pinned_model_type_and_hidden_act_for_both_architectures() {
    let dense_text_config = dense_text_config_with(&[]);
    let moe_text_config = moe_text_config_with(&[]);

    let dense_outcome = validate_exact_text_values(&dense_text_config, "qwen3_5_text");
    let moe_outcome = validate_exact_text_values(&moe_text_config, "qwen3_5_moe_text");

    assert!(dense_outcome.is_ok());
    assert!(moe_outcome.is_ok());
}

#[test]
fn should_reject_a_foreign_text_model_type_or_hidden_activation() {
    let foreign_model_type = dense_text_config_with(&[("model_type", json!("qwen3_text"))]);
    let foreign_hidden_act = dense_text_config_with(&[("hidden_act", json!("gelu"))]);

    let model_type_outcome = validate_exact_text_values(&foreign_model_type, "qwen3_5_text");
    let hidden_act_outcome = validate_exact_text_values(&foreign_hidden_act, "qwen3_5_text");

    assert!(matches!(
        model_type_outcome,
        Err(Qwen3_5ConfigError::UnexpectedStringValue {
            field_name: "text_config.model_type",
            ..
        })
    ));
    assert!(matches!(
        hidden_act_outcome,
        Err(Qwen3_5ConfigError::UnexpectedStringValue {
            field_name: "text_config.hidden_act",
            ..
        })
    ));
}

#[test]
fn should_report_the_model_type_mismatch_before_a_hidden_act_mismatch() {
    let doubly_foreign = dense_text_config_with(&[
        ("model_type", json!("qwen3_text")),
        ("hidden_act", json!("gelu")),
    ]);

    let outcome = validate_exact_text_values(&doubly_foreign, "qwen3_5_text");

    assert!(matches!(
        outcome,
        Err(Qwen3_5ConfigError::UnexpectedStringValue {
            field_name: "text_config.model_type",
            ..
        })
    ));
}

#[test]
fn should_accept_the_pinned_boolean_flags_and_allowed_mamba_dtypes() {
    let baseline = dense_text_config_with(&[]);
    let bfloat16_mamba = dense_text_config_with(&[("mamba_ssm_dtype", json!("bfloat16"))]);
    let float32_mamba = dense_text_config_with(&[("mamba_ssm_dtype", json!("float32"))]);

    assert!(validate_boolean_flags_and_mamba_dtype(&baseline).is_ok());
    assert!(validate_boolean_flags_and_mamba_dtype(&bfloat16_mamba).is_ok());
    assert!(validate_boolean_flags_and_mamba_dtype(&float32_mamba).is_ok());
}

#[test]
fn should_reject_unpinned_boolean_flags_and_unknown_mamba_dtypes() {
    let attention_bias_on = dense_text_config_with(&[("attention_bias", json!(true))]);
    let mlp_bias_on = dense_text_config_with(&[("mlp_bias", json!(true))]);
    let norm_topk_prob_off = dense_text_config_with(&[("norm_topk_prob", json!(false))]);
    let float16_mamba = dense_text_config_with(&[("mamba_ssm_dtype", json!("float16"))]);

    assert!(matches!(
        validate_boolean_flags_and_mamba_dtype(&attention_bias_on),
        Err(Qwen3_5ConfigError::UnexpectedBooleanValue {
            field_name: "text_config.attention_bias",
            ..
        })
    ));
    assert!(matches!(
        validate_boolean_flags_and_mamba_dtype(&mlp_bias_on),
        Err(Qwen3_5ConfigError::UnexpectedBooleanValue {
            field_name: "text_config.mlp_bias",
            ..
        })
    ));
    assert!(matches!(
        validate_boolean_flags_and_mamba_dtype(&norm_topk_prob_off),
        Err(Qwen3_5ConfigError::UnexpectedBooleanValue {
            field_name: "text_config.norm_topk_prob",
            ..
        })
    ));
    assert!(matches!(
        validate_boolean_flags_and_mamba_dtype(&float16_mamba),
        Err(Qwen3_5ConfigError::InvalidConfigValueDynamic { description })
            if description.contains("text_config.mamba_ssm_dtype")
    ));
}

#[test]
fn should_report_a_boolean_flag_failure_before_a_mamba_dtype_failure() {
    let doubly_invalid = dense_text_config_with(&[
        ("attention_bias", json!(true)),
        ("mamba_ssm_dtype", json!("float16")),
    ]);

    let outcome = validate_boolean_flags_and_mamba_dtype(&doubly_invalid);

    assert!(matches!(
        outcome,
        Err(Qwen3_5ConfigError::UnexpectedBooleanValue {
            field_name: "text_config.attention_bias",
            ..
        })
    ));
}

#[test]
fn should_accept_positive_dimensions_for_both_feed_forward_families() {
    let dense_text_config = dense_text_config_with(&[]);
    let moe_text_config = moe_text_config_with(&[]);

    let dense_outcome = validate_numeric_fields_are_positive(
        &dense_text_config,
        Qwen3_5FeedForwardArchitecture::Dense,
    );
    let moe_outcome = validate_numeric_fields_are_positive(
        &moe_text_config,
        Qwen3_5FeedForwardArchitecture::MixtureOfExperts,
    );

    assert!(dense_outcome.is_ok());
    assert!(moe_outcome.is_ok());
}

#[test]
fn should_reject_zero_dimensions_declared_by_the_text_config() {
    let dense_field_overrides = [
        ("hidden_size", json!(0)),
        ("num_hidden_layers", json!(0)),
        ("num_attention_heads", json!(0)),
        ("num_key_value_heads", json!(0)),
        ("head_dim", json!(0)),
        ("vocab_size", json!(0)),
        ("max_position_embeddings", json!(0)),
        ("linear_conv_kernel_dim", json!(0)),
        ("linear_num_key_heads", json!(0)),
        ("linear_num_value_heads", json!(0)),
        ("linear_key_head_dim", json!(0)),
        ("linear_value_head_dim", json!(0)),
        ("intermediate_size", json!(0)),
    ];
    for (field_name, field_value) in dense_field_overrides {
        let zeroed_dense = dense_text_config_with(&[(field_name, field_value)]);
        assert!(
            validate_numeric_fields_are_positive(
                &zeroed_dense,
                Qwen3_5FeedForwardArchitecture::Dense,
            )
            .is_err(),
            "a zeroed '{field_name}' must fail the positivity contract"
        );
    }

    let moe_field_overrides = [
        ("num_experts", json!(0)),
        ("num_experts_per_tok", json!(0)),
        ("moe_intermediate_size", json!(0)),
        ("shared_expert_intermediate_size", json!(0)),
    ];
    for (field_name, field_value) in moe_field_overrides {
        let zeroed_moe = moe_text_config_with(&[(field_name, field_value)]);
        assert!(
            validate_numeric_fields_are_positive(
                &zeroed_moe,
                Qwen3_5FeedForwardArchitecture::MixtureOfExperts,
            )
            .is_err(),
            "a zeroed '{field_name}' must fail the MoE positivity contract"
        );
    }
}

#[test]
fn should_not_require_the_dense_intermediate_size_for_mixture_of_experts() {
    let moe_without_intermediate_size = moe_text_config_with(&[("intermediate_size", json!(0))]);
    let dense_single_token_intermediate =
        dense_text_config_with(&[("intermediate_size", json!(1))]);

    let moe_outcome = validate_numeric_fields_are_positive(
        &moe_without_intermediate_size,
        Qwen3_5FeedForwardArchitecture::MixtureOfExperts,
    );
    let dense_outcome = validate_numeric_fields_are_positive(
        &dense_single_token_intermediate,
        Qwen3_5FeedForwardArchitecture::Dense,
    );

    assert!(moe_outcome.is_ok());
    assert!(dense_outcome.is_ok());
}

#[test]
fn should_accept_divisible_attention_heads_and_consistent_expert_dimensions() {
    let dense_text_config = dense_text_config_with(&[]);
    let moe_at_expert_boundary = moe_text_config_with(&[("num_experts_per_tok", json!(8))]);

    let dense_outcome = validate_attention_and_expert_head_counts(
        &dense_text_config,
        Qwen3_5FeedForwardArchitecture::Dense,
    );
    let moe_outcome = validate_attention_and_expert_head_counts(
        &moe_at_expert_boundary,
        Qwen3_5FeedForwardArchitecture::MixtureOfExperts,
    );

    assert!(dense_outcome.is_ok());
    assert!(moe_outcome.is_ok());
}

#[test]
fn should_reject_undivisible_attention_heads_and_inconsistent_expert_dimensions() {
    let undivisible_heads = dense_text_config_with(&[("num_attention_heads", json!(15))]);
    assert!(matches!(
        validate_attention_and_expert_head_counts(
            &undivisible_heads,
            Qwen3_5FeedForwardArchitecture::Dense,
        ),
        Err(Qwen3_5ConfigError::InvalidConfigValue { description })
            if description.contains("divide evenly")
    ));

    let routing_beyond_experts = moe_text_config_with(&[("num_experts_per_tok", json!(257))]);
    assert!(matches!(
        validate_attention_and_expert_head_counts(
            &routing_beyond_experts,
            Qwen3_5FeedForwardArchitecture::MixtureOfExperts,
        ),
        Err(Qwen3_5ConfigError::InvalidConfigValue { description })
            if description.contains("must not exceed num_experts")
    ));

    let dense_field_overrides = [
        ("num_experts", json!(4)),
        ("num_experts_per_tok", json!(2)),
        ("moe_intermediate_size", json!(8)),
        ("shared_expert_intermediate_size", json!(8)),
    ];
    for (field_name, field_value) in dense_field_overrides {
        let dense_with_sparse_dimension = dense_text_config_with(&[(field_name, field_value)]);
        assert!(
            matches!(
                validate_attention_and_expert_head_counts(
                    &dense_with_sparse_dimension,
                    Qwen3_5FeedForwardArchitecture::Dense,
                ),
                Err(Qwen3_5ConfigError::InvalidConfigValue { description })
                    if description.contains("must not declare sparse-expert dimensions")
            ),
            "a dense config with '{field_name}' declared must fail closed"
        );
    }
}

#[test]
fn should_report_undivisible_attention_heads_before_expert_routing_failures() {
    let doubly_invalid = moe_text_config_with(&[
        ("num_attention_heads", json!(15)),
        ("num_experts_per_tok", json!(257)),
    ]);

    let outcome = validate_attention_and_expert_head_counts(
        &doubly_invalid,
        Qwen3_5FeedForwardArchitecture::MixtureOfExperts,
    );

    assert!(matches!(
        outcome,
        Err(Qwen3_5ConfigError::InvalidConfigValue { description })
            if description.contains("divide evenly")
    ));
}

#[test]
fn should_accept_linear_attention_dimensions_within_the_mlx_shape_range() {
    let baseline = dense_text_config_with(&[]);

    let outcome = validate_linear_attention_shape_bounds(&baseline);

    assert!(outcome.is_ok());
}

#[test]
fn should_reject_linear_attention_dimensions_beyond_the_mlx_shape_range() {
    let beyond_boundary_field_overrides = [
        (
            "linear_conv_kernel_dim",
            json!(MAXIMUM_MLX_SHAPE_DIMENSION + 1),
        ),
        (
            "linear_num_value_heads",
            json!(MAXIMUM_MLX_SHAPE_DIMENSION + 1),
        ),
        (
            "linear_value_head_dim",
            json!(MAXIMUM_MLX_SHAPE_DIMENSION + 1),
        ),
        (
            "linear_key_head_dim",
            json!(MAXIMUM_MLX_SHAPE_DIMENSION + 1),
        ),
    ];
    for (field_name, field_value) in beyond_boundary_field_overrides {
        let oversized = dense_text_config_with(&[(field_name, field_value)]);
        assert!(
            matches!(
                validate_linear_attention_shape_bounds(&oversized),
                Err(Qwen3_5ConfigError::InvalidConfigValue { description })
                    if description.contains("MLX signed 32-bit shape range")
            ),
            "a '{field_name}' beyond the MLX shape range must fail closed"
        );
    }

    // Each dimension stays inside the signed 32-bit range while their derived
    // convolution state width (2^30 * 4 * 2 + 16 * 32) overflows it.
    let overflowing_state_width = dense_text_config_with(&[
        ("linear_num_key_heads", json!(1_073_741_824_u64)),
        ("linear_key_head_dim", json!(4)),
    ]);
    assert!(matches!(
        validate_linear_attention_shape_bounds(&overflowing_state_width),
        Err(Qwen3_5ConfigError::InvalidConfigValue { description })
            if description.contains("MLX signed 32-bit shape range")
    ));
}

#[test]
fn should_accept_linear_attention_dimensions_at_the_exact_mlx_shape_boundary() {
    // A single value head at exactly i32::MAX with a zero-width key side keeps
    // the derived state width at exactly i32::MAX, the inclusive boundary.
    let at_boundary = dense_text_config_with(&[
        ("linear_num_value_heads", json!(1)),
        ("linear_value_head_dim", json!(MAXIMUM_MLX_SHAPE_DIMENSION)),
        ("linear_key_head_dim", json!(0)),
    ]);
    let just_beyond_boundary = dense_text_config_with(&[
        ("linear_num_value_heads", json!(1)),
        ("linear_value_head_dim", json!(MAXIMUM_MLX_SHAPE_DIMENSION)),
        ("linear_key_head_dim", json!(1)),
    ]);

    let boundary_outcome = validate_linear_attention_shape_bounds(&at_boundary);
    let beyond_outcome = validate_linear_attention_shape_bounds(&just_beyond_boundary);

    assert!(boundary_outcome.is_ok());
    assert!(beyond_outcome.is_err());
}

#[test]
fn should_accept_supported_attention_types_matching_the_layer_count() {
    let baseline = dense_text_config_with(&[]);
    let empty_decoder =
        dense_text_config_with(&[("layer_types", json!([])), ("num_hidden_layers", json!(0))]);

    assert!(validate_layer_types(&baseline).is_ok());
    assert!(validate_layer_types(&empty_decoder).is_ok());
}

#[test]
fn should_reject_layer_type_count_mismatches_and_unsupported_attention_types() {
    let count_mismatch = dense_text_config_with(&[("layer_types", json!(["linear_attention"]))]);
    assert!(matches!(
        validate_layer_types(&count_mismatch),
        Err(Qwen3_5ConfigError::LayerTypeCountMismatch {
            actual_layer_type_count: 1,
            expected_layer_type_count: 2,
        })
    ));

    let unsupported_type = dense_text_config_with(&[(
        "layer_types",
        json!(["sliding_attention", "full_attention"]),
    )]);
    assert!(matches!(
        validate_layer_types(&unsupported_type),
        Err(Qwen3_5ConfigError::InvalidConfigValueDynamic { description })
            if description.contains("text_config.layer_types")
                && description.contains("sliding_attention")
    ));
}

#[test]
fn should_report_the_layer_type_count_mismatch_before_an_unsupported_entry() {
    let count_mismatch_with_unsupported_entry =
        dense_text_config_with(&[("layer_types", json!(["sliding_attention"]))]);

    let outcome = validate_layer_types(&count_mismatch_with_unsupported_entry);

    assert!(matches!(
        outcome,
        Err(Qwen3_5ConfigError::LayerTypeCountMismatch { .. })
    ));
}

#[test]
fn should_validate_the_full_text_config_for_both_feed_forward_families() {
    let dense_text_config = dense_text_config_with(&[]);
    let moe_text_config = moe_text_config_with(&[]);
    let moe_with_rope_parameters = {
        let mut text_config_value = baseline_moe_text_config_value();
        text_config_value["rope_parameters"] = json!({
            "mrope_interleaved": true,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000.0,
            "type": "default"
        });
        text_config_from_value(text_config_value)
    };

    assert!(
        dense_text_config
            .validate(Qwen3_5FeedForwardArchitecture::Dense)
            .is_ok()
    );
    assert!(
        moe_text_config
            .validate(Qwen3_5FeedForwardArchitecture::MixtureOfExperts)
            .is_ok()
    );
    assert!(
        moe_with_rope_parameters
            .validate(Qwen3_5FeedForwardArchitecture::MixtureOfExperts)
            .is_ok()
    );

    // The expected text model type follows the declared architecture argument.
    assert!(matches!(
        dense_text_config.validate(Qwen3_5FeedForwardArchitecture::MixtureOfExperts),
        Err(Qwen3_5ConfigError::UnexpectedStringValue {
            field_name: "text_config.model_type",
            ..
        })
    ));
}
