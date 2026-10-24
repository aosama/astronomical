use super::*;

#[test]
fn should_preserve_the_default_paging_experiment_cell() {
    let paging_experiment = PagingExperimentCase::default();
    let configuration_document =
        paging_experiment_configuration(Path::new("selected-model"), &paging_experiment);

    assert_eq!(configuration_document["maximum_mlx_memory_gb"], 32);
    assert_eq!(
        configuration_document["chunking"]["fixed_ssd_streaming_prompt_processing_chunk_size_tokens"],
        2_048
    );
    assert_eq!(
        configuration_document["chunking"]["experimental_ssd_paging_prefill_graph_submission_layer_interval"],
        1
    );
    assert_eq!(
        configuration_document["chunking"]["experimental_ssd_paging_generation_graph_submission_layer_interval"],
        1
    );
    assert_eq!(
        paging_experiment.maximum_mlx_memory_bytes().unwrap(),
        32 * BYTES_PER_GIGABYTE
    );
}

#[test]
fn should_use_each_explicit_paging_experiment_input_without_ambient_overrides() {
    let paging_experiment = PagingExperimentCase {
        maximum_mlx_memory_gigabytes: 28,
        streaming_prompt_processing_chunk_tokens: 1_024,
        prefill_graph_submission_layer_interval: 2,
        generation_graph_submission_layer_interval: 3,
    };
    let configuration_document =
        paging_experiment_configuration(Path::new("selected-model"), &paging_experiment);

    assert_eq!(configuration_document["maximum_mlx_memory_gb"], 28);
    assert_eq!(
        configuration_document["chunking"]["fixed_prompt_processing_chunk_size_tokens"],
        1_024
    );
    assert_eq!(
        configuration_document["chunking"]["fixed_ssd_streaming_prompt_processing_chunk_size_tokens"],
        1_024
    );
    assert_eq!(
        configuration_document["chunking"]["experimental_ssd_paging_prefill_graph_submission_layer_interval"],
        2
    );
    assert_eq!(
        configuration_document["chunking"]["experimental_ssd_paging_generation_graph_submission_layer_interval"],
        3
    );
    assert_eq!(
        configuration_document["persistent_prompt_cache_enabled"],
        false
    );
    assert_eq!(
        configuration_document["performance_attribution_enabled"],
        true
    );
}

#[test]
fn should_reject_a_zero_gigabyte_paging_experiment_ceiling() {
    let paging_experiment = PagingExperimentCase {
        maximum_mlx_memory_gigabytes: 0,
        ..PagingExperimentCase::default()
    };

    assert!(paging_experiment.maximum_mlx_memory_bytes().is_err());
}

#[test]
fn should_reject_a_paging_experiment_ceiling_that_overflows_bytes() {
    let paging_experiment = PagingExperimentCase {
        maximum_mlx_memory_gigabytes: u64::MAX / BYTES_PER_GIGABYTE + 1,
        ..PagingExperimentCase::default()
    };

    assert!(paging_experiment.maximum_mlx_memory_bytes().is_err());
}
