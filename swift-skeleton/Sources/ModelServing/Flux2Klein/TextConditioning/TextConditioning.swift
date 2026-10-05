// TextConditioning.swift — ModelServing.Flux2Klein.TextConditioning
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/flux2_klein/
// text_conditioning/* (prompt encoding into the diffusion conditioning
// stream).
//
// Carried contracts:
// - Text conditioning timings feed the switchable performance attribution
//   log; they are part of the serving critical path, not an afterthought.
