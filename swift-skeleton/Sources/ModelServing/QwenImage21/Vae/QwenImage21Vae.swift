// QwenImage21Vae.swift — ModelServing.QwenImage21.Vae
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/qwen_image_21/vae/*
// (latents-to-pixels decoding; kv_cache, latent_layout, and mask stay flat
// at the family root, as in Rust).
//
// Carried contracts:
// - VAE (Variational Autoencoder) decode is attributed separately from
//   diffusion steps so image latency has per-stage attribution.
