//! The Laya family: a ModernBERT encoder with one marker token per option.
//!
//! Each question is its own sequence, `[CLS] <type> question: … [SEP] [MASK]
//! option … [SEP] state [SEP]`. The shared ModernBERT trunk encodes a batch of
//! them, a small decision head reads the hidden state at each option's
//! `[MASK]`, and the contract crate turns the logits into calibrated answers.
//!
//! The artifact layout is the MLX conversion of the `convaiinnovations/laya`
//! checkpoints: one `config.json` holding the encoder config under `encoder`
//! beside the head settings and temperatures, `tokenizer.json`,
//! `tokenizer_config.json`, and one `model.safetensors` with the encoder under
//! `encoder.` and the head under `head.`, `type_emb.` and `scorer.`.

mod artifact;
mod batching;
mod configuration;
mod engine;
mod head;
mod memory_telemetry;
mod tokenizer;

pub use engine::LayaDecisionEngine;
