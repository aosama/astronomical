//! Journeys over a real folded ternary pack (Prism ML's Ternary Bonsai) on the
//! shared Qwen 3.5 core, which serves these packs since #2595:
//!
//! - `shared_core_validation`: the core validates the pack.
//! - `shared_core_reply_token_profile`: where one reply token's time goes.
//! - `stacked_sibling_projection_verdict`: stacked sibling matmuls against
//!   one launch each.
//! - `elementwise_fusion_parity`: fused elementwise kernels against MLX.
//!
//! Every journey loads the real pack, touches the graphics processor, and is
//! therefore `#[serial]` and `#[ignore]`d; name the pack with `model_directory`
//! in the explicit test-run configuration.

pub(crate) mod support;

mod elementwise_fusion_parity;
mod shared_core;
mod shared_core_reply_token_profile;
mod shared_core_validation;
mod stacked_sibling_projection_verdict;

pub(crate) use shared_core::{loaded_shared_core, shared_core_greedy_tokens};
