// Persisted-artifact contracts: what the serving layer writes, reads, and validates
// across the safetensors, persistent prompt-cache, and visual-embedding surfaces.

mod frozen_ornith_1_0_fixtures;
mod optiq_artifact_round_trip;
mod persistent_prompt_cache_block_format;
mod persistent_prompt_cache_retained_prefix_lookup;
mod required_files;
mod safetensors_byte_layout_validation;
mod safetensors_profile_validation;
mod tensor_inventory;
mod visual_embedding_format;
