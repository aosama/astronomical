// ModelServingTests.swift — ModelServingTests
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Test target for Sources/ModelServing (wave 3). Tests mirror the source
// package paths.
//
// Carried contracts:
// - LLM (Large Language Model) tests use the Romeo and Juliet fixture; no
//   random text or tokens.
// - Real-model journeys run strictly serially; hermetic CPU (Central
//   Processing Unit) tests may parallelize.
// - Throughput journeys: at least 10000 input and 1000 output tokens (plus
//   or minus 10 percent), 1000-in/100-out warmup, SSD (Solid State Drive)
//   cache disabled, optimized build only.
// - Memory journey cases allocate RAM at roughly half the model's on-disk
//   size.
// - Every test carries a built-in timeout capped at 120 seconds;
//   endurance and out-of-memory reproduction cases are the only
//   exceptions.
