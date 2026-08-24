#[cfg(feature = "model-artifact-qualification")]
mod dense_qwen3_5_image_e2e;
#[cfg(feature = "model-artifact-qualification")]
mod deployment_litmus_model;
#[cfg(feature = "model-artifact-qualification")]
mod flux2_klein_rest_qualification;
#[cfg(feature = "model-artifact-qualification")]
mod flux2_klein_rest_support;
#[cfg(feature = "model-artifact-qualification")]
mod laguna;
#[cfg(feature = "model-artifact-qualification")]
pub(crate) mod model_artifact_rest_qualification;
#[cfg(feature = "model-artifact-qualification")]
pub(crate) mod model_artifact_rest_transport;
#[cfg(feature = "model-artifact-qualification")]
mod model_fixture_discovery;
#[cfg(feature = "model-artifact-qualification")]
mod persistent_prompt_cache_append_only_rest_journey;
#[cfg(feature = "model-artifact-qualification")]
mod persistent_prompt_cache_memory_rest_journey;
#[cfg(feature = "model-artifact-qualification")]
mod persistent_prompt_cache_rest_support;
#[cfg(feature = "model-artifact-qualification")]
mod smallest_configured_qwen3_5_hard_thinking_budget_openai_rest_e2e;
#[cfg(feature = "model-artifact-qualification")]
mod speculative_prefill_rest_journey;
