// ImageEncoding.swift — ModelServing.Flux2Klein.ImageEncoding
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/flux2_klein/
// image_encoding/* (VAE (Variational Autoencoder)-side image encoding for
// image-to-image conditioning).
//
// Carried contracts:
// - Input images are untrusted payloads: decode and resize stay bounded
//   and produce typed errors.
