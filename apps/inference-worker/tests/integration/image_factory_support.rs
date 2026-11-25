//! What the image factory contracts share: the factory under test and the
//! policies the daemon would send it.

use astronomical_inference_worker::model_family_factory::ModelFamilyFactory;
use config::AstroSsdPromptCacheConfig;
use ipc_protocol::{
    WorkerAutoregressiveModelConfiguration, WorkerChunkingConfiguration,
    WorkerImageGenerationModelConfiguration, WorkerModelConfiguration,
};

use super::flux2_klein_fixture::CANONICAL_MODEL_ID;

/// The image policy the daemon sends for a folder of `model_family`.
pub(super) fn image_model_configuration(
    model_family: &str,
    model_id: &str,
    artifact_revision: &str,
) -> WorkerModelConfiguration {
    WorkerModelConfiguration::ImageGeneration(WorkerImageGenerationModelConfiguration {
        model_id: model_id.to_owned(),
        model_family: model_family.to_owned(),
        artifact_revision: artifact_revision.to_owned(),
    })
}

pub(super) fn model_family_factory(fixture_directory: &std::path::Path) -> ModelFamilyFactory {
    ModelFamilyFactory::new(
        2_000_000_000,
        200_000_000,
        AstroSsdPromptCacheConfig::new(
            fixture_directory.join("astro-ssd-prompt-cache"),
            1_000_000_000,
        ),
        true,
        ipc_protocol::WorkerDecodeAstroExpertCachePolicy::Baseline,
        fixture_directory.join("performance-attribution.jsonl"),
        true,
    )
}

/// A chat policy, which no image folder may be served under.
pub(super) fn autoregressive_configuration() -> WorkerModelConfiguration {
    WorkerModelConfiguration::Autoregressive(WorkerAutoregressiveModelConfiguration {
        model_id: CANONICAL_MODEL_ID.to_owned(),
        maximum_context_tokens: 4_096,
        maximum_output_tokens: 1_024,
        chunking: WorkerChunkingConfiguration {
            fixed_prompt_processing_chunk_size_tokens: 1_024,
            fixed_ssd_streaming_prompt_processing_chunk_size_tokens: 2_048,
            full_attention_key_value_growth_tokens: 256,
            prefill_graph_submission_layer_interval: 0,
            experimental_ssd_paging_generation_graph_submission_layer_interval: 0,
            ssd_prompt_cache_block_tokens: None,
            ssd_prompt_cache_common_prefix_stride_blocks: 4,
            experimental_decode_stage_attribution_enabled: false,
            experimental_prefill_stage_attribution_enabled: false,
            experimental_in_memory_prefix_adoption_enabled: false,
            experimental_neural_engine_prefill_enabled: false,
            multi_token_prediction_draft_tokens: None,
        },
    })
}
