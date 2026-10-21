//! Real-model quantization journeys. Every journey is `#[ignore]` and
//! `#[serial]`, discovers its artifacts through environment variables, and
//! feeds the Romeo and Juliet fixture through the source's own tokenizer.

#[path = "quantization/mod.rs"]
mod quantization_journeys;
