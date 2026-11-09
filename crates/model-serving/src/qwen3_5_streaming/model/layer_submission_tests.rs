//! Contracts for layer submission: configured groups submit never with no
//! interval, at each multiple of the interval, and never after the final
//! layer; a dense layer submits on its own by forward size.

use super::{LayerGroupSubmission, LayerSubmission};

fn submission_after(completed_layer_count: usize, interval: usize) -> LayerGroupSubmission {
    LayerGroupSubmission {
        completed_layer_count,
        decoder_layer_count: 32,
        graph_submission_layer_interval: interval,
        forward_token_count: 561,
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

#[test]
fn should_submit_every_dense_layer_alone_in_a_prompt_forward() {
    assert_eq!(
        LayerSubmission::for_layer(true, true, 561),
        LayerSubmission::DensePromptLayer
    );
    assert_eq!(
        LayerSubmission::for_layer(true, false, 561),
        LayerSubmission::DensePromptLayer
    );
}

#[test]
fn should_treat_a_two_token_forward_as_a_prompt_forward() {
    assert_eq!(
        LayerSubmission::for_layer(true, false, 2),
        LayerSubmission::DensePromptLayer
    );
}

#[test]
fn should_submit_only_streamed_dense_layers_in_a_one_token_forward() {
    assert_eq!(
        LayerSubmission::for_layer(true, true, 1),
        LayerSubmission::StreamedDecodeLayer
    );
    assert_eq!(
        LayerSubmission::for_layer(true, false, 1),
        LayerSubmission::ConfiguredGroup
    );
}

#[test]
fn should_leave_sparse_layers_to_configured_groups() {
    assert_eq!(
        LayerSubmission::for_layer(false, false, 561),
        LayerSubmission::ConfiguredGroup
    );
    assert_eq!(
        LayerSubmission::for_layer(false, false, 1),
        LayerSubmission::ConfiguredGroup
    );
}
