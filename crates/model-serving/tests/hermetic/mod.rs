mod adaptive_ram_growth_guard;
mod adaptive_ram_growth_observations;
mod artifact_validator;
mod attention;
mod complete_residency_headroom_boundary;
mod decoder_cache;
mod e2e_test_model_names;
mod engine_backed_worker;
mod expert_memory_admission;
mod expert_residency_policy;
mod kernel_capability;
mod memory_policy;
mod mlx_ram_budget;
#[cfg(feature = "direct-mlx")]
mod paged_route_materialization;
mod performance_attribution;
mod phase_aware_expert_residency;
mod quantized_expert_page_manifest;
#[cfg(feature = "direct-mlx")]
mod qwen3_5_execution_error;
#[cfg(feature = "direct-mlx")]
mod qwen_prompt_processing_chunk_sizer;
mod raw_safetensors_inventory;
mod required_files;
mod retained_expert_page_cache;
mod sparse_experts;
mod tensor_inventory;
