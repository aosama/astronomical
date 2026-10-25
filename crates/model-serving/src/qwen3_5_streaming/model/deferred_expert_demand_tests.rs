//! Unit contracts for the deferred expert-demand collector: routes queue in
//! layer order without evaluation, join completion roots by reference, and a
//! discard or take leaves the collector empty for the next forward.

use super::DeferredExpertDemandCollector;
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use mlx_c_rust::MlxArray;
use serial_test::serial;

const EXPERT_CAPACITY: usize = 256;

fn initialize_test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(512 * 1024 * 1024, 512 * 1024 * 1024)
            .expect("the test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

fn route_of(expert_ids: &[u32]) -> MlxArray {
    MlxArray::from_u32(expert_ids, &[1, 1, expert_ids.len() as i32])
        .expect("a route array should upload")
}

#[test]
#[serial]
fn should_queue_routes_in_layer_order_and_return_their_identifiers_on_take() {
    let _runtime = initialize_test_runtime();
    let mut collector = DeferredExpertDemandCollector::default();
    collector.retain_layer_route(3, EXPERT_CAPACITY, &route_of(&[7, 9]));
    collector.retain_layer_route(4, EXPERT_CAPACITY, &route_of(&[1, 255]));
    assert_eq!(collector.pending_layer_count(), 2);
    assert_eq!(collector.pending_route_array_refs().count(), 2);

    let pending_layer_routes = collector.take_pending_layer_routes();
    assert_eq!(collector.pending_layer_count(), 0);
    let layer_indices: Vec<usize> = pending_layer_routes
        .iter()
        .map(|pending_layer_route| pending_layer_route.layer_index)
        .collect();
    assert_eq!(layer_indices, vec![3, 4]);
    assert_eq!(
        pending_layer_routes[1]
            .selected_indices
            .to_vec_u32()
            .expect("the retained route should read back"),
        vec![1, 255]
    );
    assert!(
        pending_layer_routes
            .iter()
            .all(|pending_layer_route| pending_layer_route.expert_capacity == EXPERT_CAPACITY)
    );
}

#[test]
#[serial]
fn should_discard_every_pending_route() {
    let _runtime = initialize_test_runtime();
    let mut collector = DeferredExpertDemandCollector::default();
    collector.retain_layer_route(0, EXPERT_CAPACITY, &route_of(&[0]));
    collector.discard_pending_layer_routes();
    assert_eq!(collector.pending_layer_count(), 0);
    assert_eq!(collector.pending_route_array_refs().count(), 0);
}

#[test]
#[serial]
fn should_take_nothing_from_an_empty_collector() {
    let _runtime = initialize_test_runtime();
    let mut collector = DeferredExpertDemandCollector::default();
    assert!(collector.take_pending_layer_routes().is_empty());
}

#[test]
#[serial]
fn should_keep_the_retained_route_alive_after_the_caller_drops_its_array() {
    let _runtime = initialize_test_runtime();
    let mut collector = DeferredExpertDemandCollector::default();
    {
        let transient_route = route_of(&[42, 43]);
        collector.retain_layer_route(1, EXPERT_CAPACITY, &transient_route);
    }
    let pending_layer_routes = collector.take_pending_layer_routes();
    assert_eq!(
        pending_layer_routes[0]
            .selected_indices
            .to_vec_u32()
            .expect("the retained route should outlive the caller's array"),
        vec![42, 43]
    );
}
