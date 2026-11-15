// Persisted-artifact contracts: what the serving layer writes, reads, and validates
// across the safetensors, persistent astro SSD prompt cache, and visual-embedding surfaces.

mod astro_ssd_prompt_cache_block_format;
mod astro_ssd_prompt_cache_retained_prefix_lookup;
mod frozen_ornith_1_0_fixtures;
mod optiq_artifact_round_trip;
mod required_files;
mod safetensors_byte_layout_validation;
mod safetensors_profile_validation;
mod tensor_inventory;
mod visual_embedding_format;
