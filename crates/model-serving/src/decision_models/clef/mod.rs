//! The Clef family: a Qwen3.5 backbone read once over the whole request,
//! and a joint schema head that scores every option of every question
//! together (`Cloudflare/clef-flash`, served from its MLX builds).
//!
//! Text only: the vision tower an MLX build may carry is not loaded.

mod engine;
mod head_layers;
mod head_settings;
mod head_weights;
mod joint_head;
mod prompt;
#[cfg(test)]
mod test_tokenizer;
mod tokenizer;

pub use engine::ClefDecisionEngine;
