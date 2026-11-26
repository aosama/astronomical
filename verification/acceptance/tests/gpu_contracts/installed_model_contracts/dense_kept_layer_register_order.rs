//! A dense layer the engine keeps may be held in register order
//! (issues #2568 and #2585): the layout a draft tree's rows multiply fastest
//! through, which the engine already converts to when a drafter
//! binds.
//!
//! One load of the dense_vision model (4-bit, the packing that converts),
//! half its layers kept, answers the
//! Shakespeare prompt greedily before and after the kept layers convert. The
//! two layouts hold the same codes, and the kernels differ only in the order
//! they add a row's products, so logits move by rounding: the converted
//! layers must begin the reply as the stock ones do, and the replies may
//! part only where two tokens are nearly tied.
//!
//! Run with `cargo test-dense-kept-layer-register-order`.

use std::time::Duration;

use model_serving::Qwen3_5TextModel;
use serial_test::serial;
use tokio::time::timeout;

use super::greedy_answer::{
    greedy_answer, keep_dense_layers, load_materialized_streaming_model,
    shakespeare_prompt_token_ids,
};

const CONVERSION_BUDGET: Duration = Duration::from_secs(120);
/// Half of the 9B model's 32 layers, so converted and streamed layers run in
/// one forward.
const KEPT_LAYER_COUNT: usize = 16;
/// The largest logit movement the two layouts may show. They add the same
/// products in a different order, in bf16, whose neighbouring values are an
/// eighth apart at a logit of 16: the largest movement measured over the
/// vocabulary after eight greedy steps was 0.53. A wrong code or scale moves
/// logits by tens.
const LARGEST_LOGIT_DIFFERENCE: f32 = 2.0;

#[tokio::test]
#[ignore = "loads the dense_vision model and converts its kept layers to register order"]
#[serial]
async fn should_begin_the_same_reply_with_kept_layers_in_register_order() {
    timeout(CONVERSION_BUDGET, assert_kept_layers_convert())
        .await
        .expect("the kept layer conversion journey must finish within its budget");
}

async fn assert_kept_layers_convert() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = crate::common::configured_dense_vision_model_directory();
    let prompt_token_ids = shakespeare_prompt_token_ids(&model_directory);
    let (mut streaming_model, qwen3_5_config) =
        load_materialized_streaming_model(&model_directory).await;

    // Edge: with no kernels declared nothing converts, kept layers or not.
    keep_dense_layers(&streaming_model, KEPT_LAYER_COUNT);
    assert_eq!(converted_now(&streaming_model), 0);
    let stock_answer = greedy_answer(&streaming_model, &qwen3_5_config, &prompt_token_ids);

    streaming_model
        .declare_register_ordered_kernels()
        .expect("the register-ordered kernels should declare");
    // Negative: a layer a forward still holds is left in MLX's layout.
    let held_first_layer = streaming_model
        .forward_layer_weights(0)
        .expect("the first kept layer should be found");
    let converted_around_the_held_layer = converted_now(&streaming_model);
    drop(held_first_layer);
    let converted_in_the_released_layer = converted_now(&streaming_model);
    assert!(
        converted_around_the_held_layer > 0,
        "the kept layers should convert"
    );
    assert!(
        converted_in_the_released_layer > 0
            && converted_in_the_released_layer < converted_around_the_held_layer,
        "the held layer converts once released: {converted_in_the_released_layer} then, \
         {converted_around_the_held_layer} around it"
    );
    // Edge: converting again finds nothing left, and a layer read to be kept
    // after the kernels are declared arrives converted.
    assert_eq!(converted_now(&streaming_model), 0);
    keep_dense_layers(&streaming_model, KEPT_LAYER_COUNT + 1);
    assert_eq!(converted_now(&streaming_model), 0);
    keep_dense_layers(&streaming_model, KEPT_LAYER_COUNT);

    // Positive: the converted layers begin the reply as the stock layers do.
    // The two layouts add a row's products in a different order, so logits
    // move by rounding and a later pick between two nearly tied tokens may
    // part; from there the replies are different text and their logits are
    // not comparable, so the logits are held to rounding only when every
    // token agreed.
    let converted_answer = greedy_answer(&streaming_model, &qwen3_5_config, &prompt_token_ids);
    let mut agreeing_token_count = 0;
    while agreeing_token_count < stock_answer.generated_token_ids.len()
        && converted_answer.generated_token_ids[agreeing_token_count]
            == stock_answer.generated_token_ids[agreeing_token_count]
    {
        agreeing_token_count += 1;
    }
    assert!(
        agreeing_token_count > 0,
        "kept layers in register order must begin the reply as the stock layers do"
    );
    let every_token_agrees = agreeing_token_count == stock_answer.generated_token_ids.len();
    let largest_logit_difference =
        largest_difference(&converted_answer.final_logits, &stock_answer.final_logits);
    assert!(
        !every_token_agrees || largest_logit_difference <= LARGEST_LOGIT_DIFFERENCE,
        "logits moved by {largest_logit_difference}, more than rounding explains"
    );
    eprintln!(
        "[kept-register-order] status=success converted_projections={} agreeing_tokens={agreeing_token_count} largest_logit_difference={largest_logit_difference}",
        converted_around_the_held_layer + converted_in_the_released_layer
    );
}

fn converted_now(streaming_model: &Qwen3_5TextModel) -> usize {
    streaming_model
        .convert_kept_dense_layers_to_register_order()
        .expect("the kept layers should convert")
}

fn largest_difference(converted_logits: &[f32], stock_logits: &[f32]) -> f32 {
    assert_eq!(converted_logits.len(), stock_logits.len());
    let mut largest_logit_difference = 0.0_f32;
    for (converted_logit, stock_logit) in converted_logits.iter().zip(stock_logits) {
        largest_logit_difference =
            largest_logit_difference.max((converted_logit - stock_logit).abs());
    }
    largest_logit_difference
}
