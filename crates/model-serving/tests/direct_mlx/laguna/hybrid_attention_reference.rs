use astronomical_runtime_integration::MlxDtype;

use crate::direct_mlx::attention::grouped_query_reference::{
    AttentionGeometry, AttentionVisibility, assert_attention_matches_operations_reference,
    test_runtime as attention_test_runtime,
};
use crate::direct_mlx::attention::rotary_reference::{
    FrequencyKind, RotaryGeometry, assert_rotary_matches_operations_reference,
    test_runtime as rotary_test_runtime,
};

/// Named XS and S rows prove Laguna integration without turning observed
/// artifact geometry into a neutral attention default or allowed-domain check.
#[tokio::test]
async fn should_match_grouped_attention_reference_for_xs_and_s_descriptors() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = attention_test_runtime();
    let rows = [
        attention_row("xs_full", 48, AttentionVisibility::Full, MlxDtype::BFloat16),
        attention_row(
            "xs_sliding",
            64,
            AttentionVisibility::Sliding { window_size: 512 },
            MlxDtype::BFloat16,
        ),
        attention_row("s_full", 48, AttentionVisibility::Full, MlxDtype::Float16),
        attention_row(
            "s_sliding",
            72,
            AttentionVisibility::Sliding { window_size: 512 },
            MlxDtype::Float16,
        ),
    ];
    for geometry in rows {
        assert_attention_matches_operations_reference(&runtime, geometry);
    }
}

#[tokio::test]
async fn should_match_rotary_reference_for_xs_and_s_descriptors() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = rotary_test_runtime();
    let rows = [
        yarn_row("xs_full", MlxDtype::BFloat16, 32.0, 64.0, 1.346_573_6),
        default_row("xs_sliding", MlxDtype::BFloat16),
        yarn_row("s_full", MlxDtype::Float16, 128.0, 32.0, 1.485_203),
        default_row("s_sliding", MlxDtype::Float16),
    ];
    for geometry in rows {
        assert_rotary_matches_operations_reference(&runtime, geometry);
    }
}

const fn attention_row(
    row_name: &'static str,
    query_head_count: i32,
    visibility: AttentionVisibility,
    activation_dtype: MlxDtype,
) -> AttentionGeometry {
    AttentionGeometry {
        row_name,
        query_head_count,
        key_value_head_count: 8,
        query_token_count: 3,
        prefix_token_count: 6,
        head_width: 128,
        activation_dtype,
        visibility,
    }
}

const fn yarn_row(
    row_name: &'static str,
    activation_dtype: MlxDtype,
    factor: f64,
    beta_fast: f64,
    attention_factor: f32,
) -> RotaryGeometry {
    RotaryGeometry {
        row_name,
        head_width: 128,
        rotary_dimension: 64,
        activation_dtype,
        token_positions: &[5, 21, 34],
        attention_factor,
        frequency_kind: FrequencyKind::Yarn {
            theta: 500_000.0,
            original_maximum_position_count: 8_192,
            factor,
            beta_fast,
            beta_slow: 1.0,
        },
    }
}

const fn default_row(row_name: &'static str, activation_dtype: MlxDtype) -> RotaryGeometry {
    RotaryGeometry {
        row_name,
        head_width: 128,
        rotary_dimension: 128,
        activation_dtype,
        token_positions: &[6, 22, 35],
        attention_factor: 1.0,
        frequency_kind: FrequencyKind::Default { theta: 10_000.0 },
    }
}
