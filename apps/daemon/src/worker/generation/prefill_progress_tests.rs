use ipc_protocol::WorkerPromptLayerProgress;

use super::running_prefill_token_count;

fn layers_done(completed_layer_count: u32, layer_count: u32) -> Option<WorkerPromptLayerProgress> {
    Some(WorkerPromptLayerProgress {
        completed_layer_count,
        layer_count,
    })
}

#[test]
fn should_count_the_computed_tokens_of_a_prompt_read_in_chunks() {
    assert_eq!(running_prefill_token_count(4_096, 10_000, None), 4_096);
}

#[test]
fn should_count_the_layer_share_of_a_prompt_read_one_layer_at_a_time() {
    // 30 of 40 layers have read all 27,468 tokens: three quarters of the work.
    assert_eq!(
        running_prefill_token_count(0, 27_468, layers_done(30, 40)),
        20_601
    );
}

#[test]
fn should_count_nothing_before_the_first_layer_and_everything_after_the_last() {
    assert_eq!(
        running_prefill_token_count(0, 27_468, layers_done(0, 40)),
        0
    );
    assert_eq!(
        running_prefill_token_count(0, 27_468, layers_done(40, 40)),
        27_468
    );
    assert_eq!(
        running_prefill_token_count(0, 27_468, layers_done(41, 40)),
        27_468
    );
}

#[test]
fn should_fall_back_to_the_computed_tokens_for_a_layer_report_without_layers() {
    assert_eq!(
        running_prefill_token_count(12, 27_468, layers_done(3, 0)),
        12
    );
}
