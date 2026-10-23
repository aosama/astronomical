//! Router-numerics parity probe for K2 Horizon MoVA (issue #1154 item 8).
//!
//! The upstream reference contract keeps the routers unquantized: a dense
//! product of bfloat16 operands accumulating in float32, enforced by its
//! quantization predicate. The served artifact declares quantized routers and
//! the port honors the artifact. Because sigmoid-plus-bias top-k selection is
//! threshold sensitive, this probe measures whether the two maths pick the
//! same experts on real Romeo and Juliet activations: it walks one prefill
//! forward layer by layer and compares, per sparse layer and per token, the
//! selected top-k expert set under the served quantized-router product
//! against the same selection chain fed by a dequantized dense-router
//! product.
//!
//! Measured verdict on the installed 8-bit-router artifact: the probe FAILS
//! its pre-registered bounds — roughly 90 percent token-set agreement overall
//! with the worst early sparse layer diverging for about half its tokens —
//! so the artifact's quantized routers are a material fidelity deviation
//! from the reference contract, concentrated in the early layers. The probe
//! keeps asserting those bounds so the deviation stays visible until the
//! artifact pipeline changes or the bounds are consciously relaxed.

use std::time::Duration;

use astronomical_runtime_integration::MlxRuntime;

use crate::PerformanceAttribution;

use super::K2HorizonMoVAKvState;
use super::affine::K2HorizonMoVAAffineLinear;
use super::decoder;
use super::model::K2HorizonMoVAModel;
use super::prefill_causality_probe::{probe_runtime, romeo_and_juliet_prompt_token_ids};
use super::weights::K2HorizonMoVALayerWeights;
use mlx_c_rust::{MlxArray, MlxDtype};

/// The probe deadline. The synchronous MLX work cannot be interrupted, so the
/// timeout fails the journey when the blocking task overruns and the bounded
/// runner's process-tree kill remains the hard backstop.
const PROBE_DEADLINE: Duration = Duration::from_secs(115);
/// Pre-registered parity bounds: at least 95 percent of tokens must select
/// the identical expert set per router, and no single router may mismatch
/// for more than 20 percent of its tokens. A failure means the artifact's
/// quantized routers are a measurable fidelity deviation from the reference
/// contract, not a noise artifact.
const MINIMUM_ROUTER_SET_MATCH_FRACTION: f64 = 0.95;
const MAXIMUM_ROUTER_MISMATCH_FRACTION: f64 = 0.20;

#[tokio::test]
#[ignore = "requires model_directories to discover a stacked affine K2 Horizon MoVA artifact"]
async fn should_select_the_same_experts_under_served_and_reference_router_math() {
    let probe_journey = tokio::task::spawn_blocking(probe_body);
    tokio::time::timeout(PROBE_DEADLINE, probe_journey)
        .await
        .expect("the router parity probe should finish within 115 seconds")
        .expect("the router parity probe blocking task should join");
}

fn probe_body() {
    let runtime = probe_runtime();
    let model = load_probe_model(runtime);
    let prompt_token_ids = romeo_and_juliet_prompt_token_ids(&model);
    let token_count = prompt_token_ids.len() as i32;
    let feed_forward_top_k = model.config.num_experts_per_tok();
    let value_top_k = model.config.mova_num_experts_per_tok();

    let mut caches = (0..model.config.num_hidden_layers())
        .map(|_| K2HorizonMoVAKvState::build(false, 256).expect("probe KV state should build"))
        .collect::<Vec<_>>();
    let mut performance_attribution = PerformanceAttribution::disabled();

    let token_array = model
        .runtime
        .array_from_u32(&prompt_token_ids, &[prompt_token_ids.len() as i32])
        .expect("probe prompt tokens should build");
    let embeddings = model.embed(&token_array).expect("probe embeddings");
    let mut hidden_states = model
        .runtime
        .reshape(
            &embeddings,
            &[1, token_count, model.config.hidden_size() as i32],
        )
        .expect("probe hidden reshape");
    let rope_offset = 0;

    let mut router_mismatch_counts = Vec::new();
    for (layer_index, layer_weights) in model.weights.layers.iter().enumerate() {
        model
            .runtime
            .evaluate_arrays(&[&hidden_states])
            .expect("probe layer input should realize");
        if let K2HorizonMoVALayerWeights::SparseMixtureOfValues { attention, mlp, .. } =
            layer_weights
        {
            let value_mismatches = compare_one_router(
                &model.runtime,
                &hidden_states,
                &attention.v_router,
                value_top_k,
            );
            let feed_forward_mismatches = compare_one_router(
                &model.runtime,
                &hidden_states,
                &mlp.router,
                feed_forward_top_k,
            );
            eprintln!(
                "[k2-router-parity-probe] layer={layer_index} tokens={token_count} value_router_mismatches={value_mismatches} feed_forward_router_mismatches={feed_forward_mismatches}"
            );
            router_mismatch_counts.push((layer_index, value_mismatches, feed_forward_mismatches));
        }
        hidden_states = decoder::forward_layer(
            &model.runtime,
            &model.config,
            &hidden_states,
            layer_weights,
            caches.get_mut(layer_index).expect("layer cache exists"),
            rope_offset,
            true,
            &model.compiled_swiglu,
            &model.compiled_elementwise_graphs,
            model.sorted_expert_reduction_kernel.as_ref(),
            model.fused_expert_decode_kernels.as_ref(),
            &mut performance_attribution,
            None,
        )
        .expect("probe layer forward");
    }

    let router_count = router_mismatch_counts.len() as f64 * 2.0;
    let token_total = router_mismatch_counts.len() as f64 * f64::from(token_count);
    let mismatch_total: usize = router_mismatch_counts
        .iter()
        .map(|(_, value, feed_forward)| value + feed_forward)
        .sum();
    let worst_router = router_mismatch_counts
        .iter()
        .flat_map(|(layer_index, value, feed_forward)| {
            [
                (*layer_index, "value", *value),
                (*layer_index, "feed_forward", *feed_forward),
            ]
        })
        .max_by_key(|(_, _, mismatches)| *mismatches)
        .expect("the walk visits at least one sparse layer");
    let match_fraction = 1.0 - mismatch_total as f64 / token_total;
    let worst_mismatch_fraction = worst_router.2 as f64 / f64::from(token_count);
    eprintln!(
        "[k2-router-parity-probe] routers={router_count} set_match_fraction={match_fraction:.6} worst=layer({}, {}) worst_router_mismatch_fraction={worst_mismatch_fraction:.6}",
        worst_router.0, worst_router.1,
    );
    assert!(
        match_fraction >= MINIMUM_ROUTER_SET_MATCH_FRACTION,
        "served and reference router maths must select the same expert sets for at least \
         {MINIMUM_ROUTER_SET_MATCH_FRACTION} of tokens: set_match_fraction={match_fraction:.6}"
    );
    assert!(
        worst_mismatch_fraction <= MAXIMUM_ROUTER_MISMATCH_FRACTION,
        "no router may mismatch for more than {MAXIMUM_ROUTER_MISMATCH_FRACTION} of tokens: \
         worst=layer({}, {}) worst_router_mismatch_fraction={worst_mismatch_fraction:.6}",
        worst_router.0,
        worst_router.1,
    );
}

/// Compares the served quantized-router selection against the dequantized
/// dense-router selection for every token of one router's captured input.
fn compare_one_router(
    runtime: &MlxRuntime,
    hidden_states: &MlxArray,
    router: &K2HorizonMoVAAffineLinear,
    top_k: usize,
) -> usize {
    let shape = hidden_states.shape();
    let token_count = shape[..shape.len() - 1].iter().product::<i32>();
    let hidden_dimension = *shape
        .last()
        .expect("hidden states have a trailing dimension");
    let flat = runtime
        .reshape(hidden_states, &[token_count, hidden_dimension])
        .expect("probe router flat reshape");
    let served_indices = served_selection(runtime, &flat, router, top_k);
    let reference_indices = reference_selection(runtime, &flat, router, top_k);
    let token_count = token_count as usize;
    let mut mismatched_tokens = 0_usize;
    for token_index in 0..token_count {
        let mut served_row =
            served_indices[token_index * top_k..(token_index + 1) * top_k].to_vec();
        let mut reference_row =
            reference_indices[token_index * top_k..(token_index + 1) * top_k].to_vec();
        served_row.sort_unstable();
        reference_row.sort_unstable();
        if served_row != reference_row {
            mismatched_tokens += 1;
        }
    }
    mismatched_tokens
}

/// The served chain: quantized-router product, then the shared selection math.
fn served_selection(
    runtime: &MlxRuntime,
    flat: &MlxArray,
    router: &K2HorizonMoVAAffineLinear,
    top_k: usize,
) -> Vec<u32> {
    let logits = router
        .project_without_dense_bias(runtime, flat)
        .expect("served router product");
    selection_indices(runtime, &logits, router, top_k)
}

/// The reference chain: the same router weights dequantized to a dense
/// bfloat16 product (float32-accumulating inside the Metal kernel), then the
/// identical selection math.
fn reference_selection(
    runtime: &MlxRuntime,
    flat: &MlxArray,
    router: &K2HorizonMoVAAffineLinear,
    top_k: usize,
) -> Vec<u32> {
    let dense_weight = runtime
        .dequantize_affine(
            router.packed_weight(),
            router.scales(),
            router.biases(),
            router.group_size(),
            router.bits(),
        )
        .expect("reference router dequantization");
    let logits = runtime
        .matmul(
            flat,
            &runtime.transpose(&dense_weight).expect("router transpose"),
        )
        .expect("reference router product");
    selection_indices(runtime, &logits, router, top_k)
}

/// The selection math shared by both variants, mirroring `route_k2_experts`:
/// float32 sigmoid, optional dense bias, partition top-k.
fn selection_indices(
    runtime: &MlxRuntime,
    logits: &MlxArray,
    router: &K2HorizonMoVAAffineLinear,
    top_k: usize,
) -> Vec<u32> {
    let scores = runtime
        .sigmoid(
            &runtime
                .astype(logits, MlxDtype::Float32)
                .expect("probe logits cast"),
        )
        .expect("probe sigmoid");
    let scores = match router.dense_bias() {
        Some(router_bias) => runtime
            .add(
                &scores,
                &runtime
                    .astype(router_bias, MlxDtype::Float32)
                    .expect("bias cast"),
            )
            .expect("probe bias add"),
        None => scores,
    };
    let expert_count = *scores.shape().last().expect("scores have an expert axis");
    let top_k = i32::try_from(top_k).expect("probe top-k fits i32");
    let partitioned = runtime
        .argpartition_axis(&scores, expert_count - top_k, -1)
        .expect("probe argpartition");
    let token_count = *partitioned.shape().first().expect("partitioned has tokens");
    let top_k_indices = runtime
        .slice(
            &partitioned,
            &[0, expert_count - top_k],
            &[token_count, expert_count],
            &[1, 1],
        )
        .expect("probe top-k slice");
    runtime
        .astype(&top_k_indices, MlxDtype::UInt32)
        .and_then(|indices| indices.to_vec_u32())
        .expect("probe indices read")
}

fn load_probe_model(runtime: MlxRuntime) -> K2HorizonMoVAModel {
    super::prefill_causality_probe::load_probe_model(runtime)
}
