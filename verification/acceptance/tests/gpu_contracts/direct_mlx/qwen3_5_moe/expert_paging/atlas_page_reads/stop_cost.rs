//! What stopping a forward to bring a route to the host costs: a chain of
//! small layers evaluated once, against the same chain stopped after every
//! layer to copy eight indices out.

use std::time::{Duration, Instant};

use mlx_bridge::{MlxArray, MlxDtype};
use runtime_integration::MlxRuntime;

use super::fetching_layer_stages::{median, milliseconds};

/// Layers in the stop measurement: the anchor's reply stops about this often
/// in one token.
pub(super) const STOP_CHAIN_LAYER_COUNT: usize = 28;
const STOP_CHAIN_WIDTH: i32 = 2048;
const STOP_CHAIN_PASS_COUNT: usize = 40;
const STOP_CHAIN_WARM_UP_PASS_COUNT: usize = 5;
const ROUTED_INDEX_COUNT: i32 = 8;

/// How a pass brings each layer's route to the host.
#[derive(Clone, Copy, PartialEq)]
enum RouteStop {
    /// No stop: the chain is evaluated once at the end.
    Never,
    /// The host sleeps inside MLX until the route is ready, as serving does.
    Waiting,
    /// The route is submitted and the host spins on its readiness, which
    /// trades a busy processor core for the time a sleeping thread takes to
    /// be woken.
    Polling,
}

/// One pass down the chain. With a stop, each layer's eight highest outputs
/// are brought to the host before the next layer is built, as a route is.
fn stop_chain_pass(
    runtime: &MlxRuntime,
    layer_weights: &[MlxArray],
    first_row: &MlxArray,
    route_stop: RouteStop,
) -> Duration {
    let started_at = Instant::now();
    let mut hidden_row = first_row.retain().expect("the row should retain");
    let mut routed_index_count = 0_usize;
    for layer_weight in layer_weights {
        let projected = runtime
            .matmul(&hidden_row, layer_weight)
            .expect("the layer should build");
        hidden_row = runtime
            .tanh(&projected)
            .expect("the activation should build");
        if route_stop != RouteStop::Never {
            let routed_indices = runtime
                .argpartition(&hidden_row, -ROUTED_INDEX_COUNT)
                .expect("the route should build");
            if route_stop == RouteStop::Polling {
                runtime
                    .async_eval_arrays(&[&routed_indices])
                    .expect("the route should be submitted");
                while !routed_indices
                    .is_available()
                    .expect("the route's readiness should be readable")
                {
                    std::hint::spin_loop();
                }
            }
            routed_index_count += routed_indices
                .to_vec_u32()
                .expect("the route should reach the host")
                .len();
        }
    }
    runtime
        .evaluate_arrays(&[&hidden_row])
        .expect("the chain should evaluate");
    std::hint::black_box(routed_index_count);
    started_at.elapsed()
}

/// Median milliseconds of the chain evaluated once and stopped at every
/// layer, in that order.
pub(super) fn measure_stop_cost(runtime: &MlxRuntime) -> (f64, f64) {
    let random_key = runtime.random_key(2329).expect("a key builds");
    let mut layer_weights = Vec::with_capacity(STOP_CHAIN_LAYER_COUNT);
    for _ in 0..STOP_CHAIN_LAYER_COUNT {
        layer_weights.push(
            runtime
                .random_normal(
                    &[STOP_CHAIN_WIDTH, STOP_CHAIN_WIDTH],
                    MlxDtype::Float32,
                    0.0,
                    0.02,
                    &random_key,
                )
                .expect("a layer weight builds"),
        );
    }
    let first_row = runtime
        .random_normal(
            &[1, STOP_CHAIN_WIDTH],
            MlxDtype::Float32,
            0.0,
            1.0,
            &random_key,
        )
        .expect("the first row builds");
    let mut prepared_arrays: Vec<&MlxArray> = layer_weights.iter().collect();
    prepared_arrays.push(&first_row);
    runtime
        .evaluate_arrays(&prepared_arrays)
        .expect("the chain's inputs materialize");
    let mut evaluated_once = Vec::with_capacity(STOP_CHAIN_PASS_COUNT);
    let mut stopped_at_every_layer = Vec::with_capacity(STOP_CHAIN_PASS_COUNT);
    let mut polled_at_every_layer = Vec::with_capacity(STOP_CHAIN_PASS_COUNT);
    for pass_index in 0..STOP_CHAIN_WARM_UP_PASS_COUNT + STOP_CHAIN_PASS_COUNT {
        let once = stop_chain_pass(runtime, &layer_weights, &first_row, RouteStop::Never);
        let stopped = stop_chain_pass(runtime, &layer_weights, &first_row, RouteStop::Waiting);
        let polled = stop_chain_pass(runtime, &layer_weights, &first_row, RouteStop::Polling);
        if pass_index >= STOP_CHAIN_WARM_UP_PASS_COUNT {
            evaluated_once.push(milliseconds(once));
            stopped_at_every_layer.push(milliseconds(stopped));
            polled_at_every_layer.push(milliseconds(polled));
        }
    }
    eprintln!(
        "[stop-cost] chain of {STOP_CHAIN_LAYER_COUNT} layers: {:.3} ms evaluated once, {:.3} ms waiting at every layer, {:.3} ms polling at every layer",
        median(&mut evaluated_once),
        median(&mut stopped_at_every_layer),
        median(&mut polled_at_every_layer),
    );
    (
        median(&mut evaluated_once),
        median(&mut stopped_at_every_layer),
    )
}
