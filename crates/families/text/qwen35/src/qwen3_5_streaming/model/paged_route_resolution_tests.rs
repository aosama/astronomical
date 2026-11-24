use super::PagedForwardMissingRouteCollector;

#[test]
fn should_not_be_optimistic_until_told_to() {
    let collector = PagedForwardMissingRouteCollector::default();

    assert!(!collector.is_optimistic());
    assert!(!collector.has_abandoned_optimistic_serving.get());
}

#[test]
fn should_remember_that_a_layer_could_not_be_served_optimistically() {
    let collector = PagedForwardMissingRouteCollector::default();
    collector.set_optimistic(true);

    collector.abandon_optimistic_serving();

    assert!(collector.is_optimistic());
    assert!(collector.has_abandoned_optimistic_serving.get());
}

#[test]
fn should_forget_an_abandoned_attempt_when_the_next_forward_starts() {
    let collector = PagedForwardMissingRouteCollector::default();
    collector.abandon_optimistic_serving();

    collector.clear();

    assert!(!collector.has_abandoned_optimistic_serving.get());
    assert!(collector.take_optimistic_layer_servings().is_empty());
}
