// The model contract helpers are shared with the root acceptance package's
// GPU contracts; a single path declaration keeps the offline lane compiling
// against the one copy without duplicating fixture resolution logic.
#[path = "../../../tests/acceptance/tests/gpu_contracts/common/mod.rs"]
mod common;
mod flux2_klein_offline;
mod k2_horizon_mova_offline;
mod offline;
mod qwen3_5_moe_offline;
mod qwen3_5_offline;
mod qwen_image_21_offline;
