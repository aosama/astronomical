//! The Kev family: a Qwen3.5 backbone with a LoRA adapter folded in, and a
//! pointer head that scores each option against its question.
//!
//! Kev reads one state and answers each question as its own causal row
//! continuing that state: `<|fim_middle|> instructions`, then each option as
//! `<|box_start|> option <|box_end|>`, then `<|fim_suffix|>`, the decision
//! token. The head projects the decision token's final hidden state and each
//! option's closing token's, and their scaled dot products, divided by the
//! checkpoint's temperature, are the option logits.
//!
//! The folder is the output of `cargo xtask convert-kev`: the merged
//! backbone in the base model's own layout, `kev_head.safetensors`,
//! `kev_decision.json`, and the tokenizer.

mod engine;
mod pointer_head;
mod sequence;
mod settings;
mod tokenizer;

pub use engine::KevDecisionEngine;
pub use sequence::KevRowLayout;
