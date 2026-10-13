mod artifact_validator;
mod e2e_test_model_names;
mod engine_backed_worker;
mod macos_process_io;
mod performance_attribution_log;
mod qwen3_5_import_direction;
mod raw_safetensors_inventory;
mod required_files;
mod tensor_inventory;

// The #596 reproduction exercises the feature-gated ModernBERT tokenizer, so it runs in the
// unified (feature-enabled) lanes exactly like the other gated offline modules.
#[cfg(feature = "direct-mlx")]
mod modernbert_tokenizer_gap;
#[cfg(feature = "direct-mlx")]
mod paged_route_materialization;
#[cfg(feature = "direct-mlx")]
mod qwen3_5_execution_error;
#[cfg(feature = "direct-mlx")]
mod qwen_prompt_processing_chunk_sizer;
#[cfg(feature = "direct-mlx")]
mod qwen_resident_prompt_processing_chunk_sizer;
