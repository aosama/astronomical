// Engine.swift — ModelServing.QwenImage21.Engine
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/qwen_image_21/engine/*
// (image generation engine; the family root keeps engine_error,
// render_contract, render_session, scheduler flat beside it, as in Rust).
//
// Carried contracts:
// - Diffusion-step and render timings feed the switchable performance
//   attribution log.
// - Render sessions enforce the render contract explicitly; a violated
//   contract is a typed error, not undefined behavior.
