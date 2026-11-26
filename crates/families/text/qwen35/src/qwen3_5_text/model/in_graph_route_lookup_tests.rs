//! A lookup outside its forward's window refuses without reaching the
//! model, which is what makes the model pointer safe to carry.
//!
//! The lookups here carry a null model pointer and run for real, inside an
//! MLX evaluation: any use of the pointer would crash the test, so a passing
//! test shows it was never used.

use std::sync::Arc;

use graph_host_step::{HostStepOutput, append_host_step};
use mlx_bridge::{MlxArray, MlxDtype};
use serial_test::file_serial;

use super::{
    InGraphForwardWindow, InGraphLayerOutcome, InGraphLookupTable, InGraphRouteLookup,
    ModelDuringForward, SharedLayerOutcome,
};
use runtime_integration::{MlxMemoryLimits, MlxRuntime};

pub(crate) fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(512 * 1024 * 1024, 512 * 1024 * 1024)
            .expect("the test memory limits are valid"),
    )
    .expect("the direct MLX runtime initializes")
}

fn lookup_with_no_model(
    window: &Arc<InGraphForwardWindow>,
    outcome: &Arc<SharedLayerOutcome>,
) -> Box<InGraphRouteLookup> {
    Box::new(InGraphRouteLookup {
        model: ModelDuringForward(std::ptr::null()),
        window: Arc::clone(window),
        layer_index: 0,
        lookup_table: InGraphLookupTable {
            warm_slot_count: 8,
            slot_count_when_built: 8,
        },
        outcome: Arc::clone(outcome),
    })
}

/// Evaluates a lookup of a two-expert route and returns the slots the GPU
/// would have been handed.
fn evaluated_table_slots(
    window: &Arc<InGraphForwardWindow>,
    outcome: &Arc<SharedLayerOutcome>,
) -> Vec<u32> {
    let routed_expert_ids = MlxArray::from_u32(&[5, 9], &[2]).expect("the route uploads");
    append_host_step(
        &[&routed_expert_ids],
        HostStepOutput {
            shape: &[2],
            dtype: MlxDtype::UInt32,
        },
        lookup_with_no_model(window, outcome),
    )
    .expect("the lookup joins the graph")
    .to_vec_u32()
    .expect("the lookup evaluates")
}

#[test]
#[file_serial]
fn should_refuse_without_reaching_the_model_when_its_forward_is_over() {
    let _runtime = test_runtime();
    let window = Arc::new(InGraphForwardWindow::default());
    let never_opened = Arc::new(SharedLayerOutcome::default());
    let closed_again = Arc::new(SharedLayerOutcome::default());

    let slots_before_any_forward = evaluated_table_slots(&window, &never_opened);
    window.open();
    window.close();
    let slots_after_a_forward = evaluated_table_slots(&window, &closed_again);

    for (outcome, table_slots) in [
        (never_opened, slots_before_any_forward),
        (closed_again, slots_after_a_forward),
    ] {
        assert!(matches!(
            *outcome.lock(),
            InGraphLayerOutcome::Refused(None)
        ));
        assert_eq!(table_slots, vec![0, 0], "slot zero, which every table has");
    }
}

#[test]
#[file_serial]
fn should_refuse_without_reaching_the_model_after_an_earlier_lookup_refused() {
    let _runtime = test_runtime();
    let window = Arc::new(InGraphForwardWindow::default());
    let outcome = Arc::new(SharedLayerOutcome::default());
    window.open();
    window.mark_refused();

    let table_slots = evaluated_table_slots(&window, &outcome);

    assert!(matches!(
        *outcome.lock(),
        InGraphLayerOutcome::Refused(None)
    ));
    assert_eq!(table_slots, vec![0, 0]);
    assert!(window.was_refused());
}

#[test]
fn should_start_each_forward_unrefused_and_open() {
    let window = InGraphForwardWindow::default();
    assert!(!window.is_open(), "no forward has begun");
    window.open();
    window.mark_refused();
    window.close();

    window.open();

    assert!(window.is_open());
    assert!(
        !window.was_refused(),
        "a refusal belongs to its own forward"
    );
}
