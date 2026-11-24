//! Diffusers-reference oracle fixtures for the Qwen-Image-2.1 unit and acceptance contracts.
//!
//! Every float is stored as exact decimal text rather than as a JSON number so
//! consumers round-trip the reference values bit-for-bit through the shared
//! oracle-parsing helpers in the owning crate.

/// Text-conditioning oracle: encoded prompt token ids to expected hidden states.
pub const QWEN_IMAGE_21_CONDITIONING_ORACLE_JSON: &str =
    include_str!("qwen_image_21_oracles/conditioning_oracle.json");

/// Rotary-embedding oracle for the reference image shapes and pad mask.
pub const QWEN_IMAGE_21_ROPE_ORACLE_JSON: &str =
    include_str!("qwen_image_21_oracles/rope_oracle.json");

/// Timestep oracle for the reference scheduler documents.
pub const QWEN_IMAGE_21_TIME_ORACLE_JSON: &str =
    include_str!("qwen_image_21_oracles/time_oracle.json");
