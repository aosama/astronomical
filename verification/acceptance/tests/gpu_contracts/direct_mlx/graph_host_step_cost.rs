//! What it costs a streamed reply token to look at each layer's routes on
//! the host: by ending the evaluation at every layer (what the streaming
//! engine does today) against a host step inside the graph
//! (`graph-host-step`), with a pass that never leaves the GPU as the floor.
//!
//! Random float32 tensors in the 35B-A3B's shape: 40 layers a token, each a
//! router over 256 experts, the best 8 looked up, their rows gathered from a
//! table and added to a 2,048-wide projection. The lookup is the identity, so
//! all three passes compute the same token and the GPU-only pass can do it
//! without the host; the test fails if they disagree. No expert is read from
//! storage here: this prices the hand-off between the GPU and the host, which
//! is paid on every layer whether or not an expert is missing.
//!
//! MLX has two ways to order a CPU step among GPU steps, Metal shared events
//! and its fast synch, so the step inside the graph is priced under each.
//!
//! Every pass gathers the looked-up rows three times, as a real expert layer
//! does for its gate, up and down projections. MLX makes each GPU step that
//! reads a host step's output wait for it on its own, so three readers are
//! three waits, and a fourth is the host step's wait for its route. The last
//! pass puts one GPU copy between the step and its readers, which leaves two
//! waits a layer however many kernels read the rows. Each pass prints the
//! waits MLX counted a layer beside its time.
//!
//! Run with `cargo measure-graph-host-step`. Seconds.

use std::sync::Arc;
use std::time::{Duration, Instant};

use graph_host_step::{
    HostStep, HostStepFailed, HostStepInputs, HostStepOutput, append_host_step, set_fast_synch,
};
use mlx_bridge::{MlxArray, MlxDtype};
use runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::serial;

const HIDDEN_WIDTH: i32 = 2_048;
const EXPERT_COUNT: i32 = 256;
const ROUTED_EXPERT_COUNT: i32 = 8;
const LAYERS_PER_TOKEN: usize = 40;
const WARMUP_TOKEN_COUNT: usize = 20;
const TIMED_TOKEN_COUNT: usize = 300;
const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);
const BYTES_PER_EXPERT_ID: usize = std::mem::size_of::<u32>();
/// How a layer turns the experts its router chose into rows of the table.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RouteLookup {
    /// The evaluation ends at the router; the routes are read back, looked
    /// up, and uploaded for the rest of the layer.
    HostStopALayer,
    /// A host step inside the graph looks the routes up; the token is one
    /// evaluation. Ordered by Metal shared events, MLX's default.
    StepInsideTheGraph,
    /// The same step, ordered by MLX's fast synch.
    StepInsideTheGraphFastSynch,
    /// The same step under fast synch, its output copied once on the GPU
    /// so only the copy waits for the host.
    StepInsideTheGraphReadOnce,
    /// The routes index the table directly: no host at all.
    GpuOnly,
}

impl RouteLookup {
    fn description(self) -> &'static str {
        match self {
            Self::HostStopALayer => "host stop a layer (today)",
            Self::StepInsideTheGraph => "step in graph, shared events",
            Self::StepInsideTheGraphFastSynch => "step in graph, fast synch",
            Self::StepInsideTheGraphReadOnce => "step in graph, read once",
            Self::GpuOnly => "GPU only (floor)",
        }
    }
}

/// One synthetic layer's weights, shared by every layer of the token.
struct SyntheticLayer {
    router: MlxArray,
    projection: MlxArray,
    expert_rows: MlxArray,
    routed_row_scale: MlxArray,
    /// Added to looked-up rows to copy them on the GPU, and multiplied into
    /// the second and third gathers so they are computed and change nothing.
    zero_row: MlxArray,
    zero_scale: MlxArray,
    /// The host's side of the lookup: the table row that holds each expert.
    row_of_expert: Arc<[u32]>,
}

fn synthetic_layer(runtime: &MlxRuntime) -> SyntheticLayer {
    let hidden_width = HIDDEN_WIDTH as usize;
    let expert_count = EXPERT_COUNT as usize;
    let router = runtime
        .array_from_f32(
            &super::synthetic_values::pseudo_random_values(hidden_width * expert_count, 3),
            &[HIDDEN_WIDTH, EXPERT_COUNT],
        )
        .expect("the router uploads");
    // Scaled so forty layers of it neither grow nor vanish.
    let projection = runtime
        .multiply_scalar(
            &runtime
                .array_from_f32(
                    &super::synthetic_values::pseudo_random_values(hidden_width * hidden_width, 5),
                    &[HIDDEN_WIDTH, HIDDEN_WIDTH],
                )
                .expect("the projection uploads"),
            1.0 / 32.0,
        )
        .expect("the projection scales");
    let expert_rows = runtime
        .array_from_f32(
            &super::synthetic_values::pseudo_random_values(expert_count * hidden_width, 7),
            &[EXPERT_COUNT, HIDDEN_WIDTH],
        )
        .expect("the expert rows upload");
    let routed_row_scale = runtime
        .array_from_f32(&[1.0 / ROUTED_EXPERT_COUNT as f32], &[1])
        .expect("the scale uploads");
    let zero_row = MlxArray::from_u32(&[0], &[1]).expect("the zero row uploads");
    let zero_scale = runtime
        .array_from_f32(&[0.0], &[1])
        .expect("the zero scale uploads");
    runtime
        .evaluate_arrays(&[
            &router,
            &projection,
            &expert_rows,
            &routed_row_scale,
            &zero_row,
            &zero_scale,
        ])
        .expect("the weights evaluate");
    SyntheticLayer {
        router,
        projection,
        expert_rows,
        routed_row_scale,
        zero_row,
        zero_scale,
        row_of_expert: (0..EXPERT_COUNT as u32).collect(),
    }
}

/// The lookup as a host step: expert ids in, table rows out.
struct ExpertRowLookup {
    row_of_expert: Arc<[u32]>,
}

impl HostStep for ExpertRowLookup {
    fn run(&mut self, inputs: HostStepInputs<'_>, output: &mut [u8]) -> Result<(), HostStepFailed> {
        let expert_id_bytes = inputs.bytes(0).ok_or(HostStepFailed)?;
        let (expert_ids, _) = expert_id_bytes.as_chunks::<BYTES_PER_EXPERT_ID>();
        let (rows, _) = output.as_chunks_mut::<BYTES_PER_EXPERT_ID>();
        for (expert_id, row) in expert_ids.iter().zip(rows) {
            let expert_row = self
                .row_of_expert
                .get(u32::from_ne_bytes(*expert_id) as usize)
                .ok_or(HostStepFailed)?;
            *row = expert_row.to_ne_bytes();
        }
        Ok(())
    }
}

/// The table rows of the experts `routes` names, by `route_lookup`.
fn routed_rows(layer: &SyntheticLayer, routes: &MlxArray, route_lookup: RouteLookup) -> MlxArray {
    match route_lookup {
        RouteLookup::GpuOnly => routes.retain().expect("an array handle is retained"),
        RouteLookup::HostStopALayer => {
            let expert_ids = routes.to_vec_u32().expect("the routes read back");
            let mut rows = Vec::with_capacity(expert_ids.len());
            for expert_id in expert_ids {
                rows.push(layer.row_of_expert[expert_id as usize]);
            }
            MlxArray::from_u32(&rows, &[ROUTED_EXPERT_COUNT]).expect("the rows upload")
        }
        RouteLookup::StepInsideTheGraph
        | RouteLookup::StepInsideTheGraphFastSynch
        | RouteLookup::StepInsideTheGraphReadOnce => append_host_step(
            &[routes],
            HostStepOutput {
                shape: &[ROUTED_EXPERT_COUNT],
                dtype: MlxDtype::UInt32,
            },
            Box::new(ExpertRowLookup {
                row_of_expert: Arc::clone(&layer.row_of_expert),
            }),
        )
        .expect("the step joins the graph"),
    }
}

/// The rows gathered once for each of an expert layer's three projections.
/// The second and third gathers are scaled by zero: they read `rows` and
/// run, and the sum is the first gather exactly.
fn gathered_three_times(runtime: &MlxRuntime, layer: &SyntheticLayer, rows: &MlxArray) -> MlxArray {
    let mut gathered = runtime
        .take_axis(&layer.expert_rows, rows, 0)
        .expect("the first gather builds");
    for _ in 0..2 {
        let further_gather = runtime
            .take_axis(&layer.expert_rows, rows, 0)
            .expect("a further gather builds");
        gathered = runtime
            .add(
                &gathered,
                &runtime
                    .multiply(&further_gather, &layer.zero_scale)
                    .expect("the further gather scales to nothing"),
            )
            .expect("the gathers sum");
    }
    gathered
}

/// One layer: route, look up, gather, project.
fn forward_layer(
    runtime: &MlxRuntime,
    layer: &SyntheticLayer,
    hidden: &MlxArray,
    route_lookup: RouteLookup,
) -> MlxArray {
    let router_logits = runtime
        .matmul(hidden, &layer.router)
        .expect("the router builds");
    let partitioned_experts = runtime
        .argpartition_axis(&router_logits, EXPERT_COUNT - ROUTED_EXPERT_COUNT, -1)
        .expect("the partition builds");
    let routes = runtime
        .reshape(
            &runtime
                .slice(
                    &partitioned_experts,
                    &[0, EXPERT_COUNT - ROUTED_EXPERT_COUNT],
                    &[1, EXPERT_COUNT],
                    &[1, 1],
                )
                .expect("the routes slice"),
            &[ROUTED_EXPERT_COUNT],
        )
        .expect("the routes flatten");
    let rows = routed_rows(layer, &routes, route_lookup);
    let rows = if route_lookup == RouteLookup::StepInsideTheGraphReadOnce {
        runtime
            .add(&rows, &layer.zero_row)
            .expect("the GPU copy of the rows builds")
    } else {
        rows
    };
    let gathered = gathered_three_times(runtime, layer, &rows);
    let routed_sum = runtime
        .sum_axis(&gathered, 0, true)
        .expect("the routed sum builds");
    let projected = runtime
        .matmul(hidden, &layer.projection)
        .expect("the projection builds");
    runtime
        .tanh(
            &runtime
                .add(
                    &projected,
                    &runtime
                        .multiply(&routed_sum, &layer.routed_row_scale)
                        .expect("the routed sum scales"),
                )
                .expect("the layer sums"),
        )
        .expect("the layer bounds its output")
}

/// One token: forty layers from the same first hidden row, evaluated.
fn forward_token(
    runtime: &MlxRuntime,
    layer: &SyntheticLayer,
    first_hidden: &MlxArray,
    route_lookup: RouteLookup,
) -> Vec<f32> {
    let mut hidden = first_hidden.retain().expect("an array handle is retained");
    for _ in 0..LAYERS_PER_TOKEN {
        hidden = forward_layer(runtime, layer, &hidden, route_lookup);
    }
    hidden.to_vec_f32().expect("the token evaluates")
}

/// A pass's median token, and the last token it computed.
struct PassTiming {
    median_token_milliseconds: f64,
    /// Times one stream waited for another's output, a layer, as MLX
    /// counted them over the timed tokens.
    cross_stream_waits_a_layer: f64,
    last_hidden: Vec<f32>,
}

fn time_pass(
    runtime: &MlxRuntime,
    layer: &SyntheticLayer,
    first_hidden: &MlxArray,
    route_lookup: RouteLookup,
) -> PassTiming {
    set_fast_synch(matches!(
        route_lookup,
        RouteLookup::StepInsideTheGraphFastSynch | RouteLookup::StepInsideTheGraphReadOnce
    ));
    let mut last_hidden = Vec::new();
    for _ in 0..WARMUP_TOKEN_COUNT {
        last_hidden = forward_token(runtime, layer, first_hidden, route_lookup);
    }
    let counts_before = graph_host_step::evaluation_counts();
    let mut token_milliseconds = Vec::with_capacity(TIMED_TOKEN_COUNT);
    let pass_started = Instant::now();
    let mut last_report = Instant::now();
    for token_index in 0..TIMED_TOKEN_COUNT {
        let token_started = Instant::now();
        last_hidden = forward_token(runtime, layer, first_hidden, route_lookup);
        token_milliseconds.push(token_started.elapsed().as_secs_f64() * 1000.0);
        if last_report.elapsed() >= PROGRESS_INTERVAL {
            last_report = Instant::now();
            let completed = token_index + 1;
            let elapsed_seconds = pass_started.elapsed().as_secs_f64();
            eprintln!(
                "[graph-host-step-cost] {} {completed}/{TIMED_TOKEN_COUNT} ({:.0}%) elapsed {elapsed_seconds:.1}s, about {:.0}s left",
                route_lookup.description(),
                100.0 * completed as f64 / TIMED_TOKEN_COUNT as f64,
                elapsed_seconds / completed as f64 * (TIMED_TOKEN_COUNT - completed) as f64
            );
        }
    }
    token_milliseconds.sort_by(f64::total_cmp);
    let cross_stream_wait_count = graph_host_step::evaluation_counts()
        .since(counts_before)
        .cross_stream_waits;
    PassTiming {
        cross_stream_waits_a_layer: cross_stream_wait_count as f64
            / (TIMED_TOKEN_COUNT * LAYERS_PER_TOKEN) as f64,
        median_token_milliseconds: token_milliseconds[token_milliseconds.len() / 2],
        last_hidden,
    }
}

fn report_pass(route_lookup: RouteLookup, timing: &PassTiming, floor_milliseconds: f64) {
    let hand_off_microseconds_a_layer =
        (timing.median_token_milliseconds - floor_milliseconds) * 1000.0 / LAYERS_PER_TOKEN as f64;
    eprintln!(
        "[graph-host-step-cost] {:<28} token_milliseconds={:.3} tokens_per_second={:.1} hand_off_microseconds_a_layer={hand_off_microseconds_a_layer:.1} cross_stream_waits_a_layer={:.1}",
        route_lookup.description(),
        timing.median_token_milliseconds,
        1000.0 / timing.median_token_milliseconds,
        timing.cross_stream_waits_a_layer
    );
}

#[test]
#[ignore = "runs MLX on the GPU for a timing; seconds"]
#[serial]
fn should_price_a_host_stop_a_layer_against_a_step_inside_the_graph() {
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(2 * 1024 * 1024 * 1024, 512 * 1024 * 1024)
            .expect("memory limits are valid"),
    )
    .expect("the MLX runtime initializes");
    let layer = synthetic_layer(&runtime);
    let first_hidden = runtime
        .array_from_f32(
            &super::synthetic_values::pseudo_random_values(HIDDEN_WIDTH as usize, 11),
            &[1, HIDDEN_WIDTH],
        )
        .expect("the first hidden row uploads");
    eprintln!(
        "[graph-host-step-cost] layers_per_token={LAYERS_PER_TOKEN} timed_tokens={TIMED_TOKEN_COUNT}"
    );
    let route_lookups = [
        RouteLookup::GpuOnly,
        RouteLookup::HostStopALayer,
        RouteLookup::StepInsideTheGraph,
        RouteLookup::StepInsideTheGraphFastSynch,
        RouteLookup::StepInsideTheGraphReadOnce,
    ];
    let mut timings = Vec::with_capacity(route_lookups.len());
    for route_lookup in route_lookups {
        timings.push(time_pass(&runtime, &layer, &first_hidden, route_lookup));
    }
    set_fast_synch(false);

    let floor_milliseconds = timings[0].median_token_milliseconds;
    for (route_lookup, timing) in route_lookups.into_iter().zip(&timings) {
        report_pass(route_lookup, timing, floor_milliseconds);
        assert_eq!(
            timing.last_hidden,
            timings[0].last_hidden,
            "{} computes the GPU-only token",
            route_lookup.description()
        );
    }
}
