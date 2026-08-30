//! Shared Romeo and Juliet prompt construction for remaining paged-prefill journeys.

pub(crate) use crate::serving_acceptance::support::exact_model_prompt::prepare_reproduced_long_prompt_token_ids_for_model;

pub(crate) fn prepare_reproduced_long_prompt_token_ids(
    prompt_token_count: usize,
    maximum_output_token_count: u16,
) -> Vec<u32> {
    let model_directory = crate::common::configured_large_sparse_moe_model_directory();
    prepare_reproduced_long_prompt_token_ids_for_model(
        &model_directory,
        crate::common::large_sparse_moe_model_id(),
        prompt_token_count,
        maximum_output_token_count,
    )
    .expect("the reproduced prompt should prepare at the exact requested length")
}
