//! The fused gated-delta prework over whole prompt spans: the same kernel the
//! decode contract proves, at a 2,048-row span and an uneven 37-row tail,
//! against the composed prompt path, bit for bit.

use mlx_bridge::MlxDtype;
use std::time::Duration;
use tokio::time::timeout;

use super::gdn_decode_prework_numerics::compare_fused_prework_with_composed_path;

const TEST_TIMEOUT: Duration = Duration::from_secs(115);

#[tokio::test]
async fn should_match_the_composed_prompt_path_bit_for_bit_over_a_2048_token_span() {
    timeout(
        TEST_TIMEOUT,
        compare_fused_prework_with_composed_path(2_048, MlxDtype::BFloat16),
    )
    .await
    .expect("the 2,048-token prework numerics test must finish within 115 seconds");
}

#[tokio::test]
async fn should_match_the_composed_prompt_path_bit_for_bit_over_an_uneven_tail() {
    timeout(
        TEST_TIMEOUT,
        compare_fused_prework_with_composed_path(37, MlxDtype::BFloat16),
    )
    .await
    .expect("the 37-token prework numerics test must finish within 115 seconds");
}
