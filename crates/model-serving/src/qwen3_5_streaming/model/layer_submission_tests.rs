//! Contracts for configured layer-group submission: never with no interval,
//! at each multiple of the interval, and never after the final layer.

use super::LayerGroupSubmission;

fn submission_after(completed_layer_count: usize, interval: usize) -> LayerGroupSubmission {
    LayerGroupSubmission {
        completed_layer_count,
        decoder_layer_count: 32,
        graph_submission_layer_interval: interval,
    }
}

#[test]
fn should_submit_at_each_multiple_of_the_interval() {
    assert!(submission_after(4, 4).is_due());
    assert!(submission_after(8, 4).is_due());
}

#[test]
fn should_not_submit_between_multiples_of_the_interval() {
    assert!(!submission_after(5, 4).is_due());
}

#[test]
fn should_never_submit_after_the_final_layer() {
    assert!(!submission_after(32, 4).is_due());
    assert!(!submission_after(32, 1).is_due());
}

#[test]
fn should_never_submit_without_an_interval() {
    assert!(!submission_after(4, 0).is_due());
}
