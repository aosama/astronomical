// Persisted-artifact contracts: what the serving layer writes, reads, and validates
// across the safetensors, persistent prompt-cache, and visual-embedding surfaces.

mod artifact_validator;
mod persistent_prompt_cache_block_format;
mod persistent_prompt_cache_model_contract_fixture;
mod persistent_prompt_cache_partial_tail_lookup;
mod persistent_prompt_cache_prefix_lookup;
mod persistent_prompt_cache_retained_prefix_lookup;
mod raw_safetensors_inventory;
mod required_files;
mod tensor_inventory;
mod visual_embedding_format;
