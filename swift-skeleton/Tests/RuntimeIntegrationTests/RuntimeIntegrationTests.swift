// RuntimeIntegrationTests.swift — RuntimeIntegrationTests
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Test target for Sources/RuntimeIntegration (wave 3). Tests mirror the
// source package paths.
//
// Carried contracts:
// - Real-model journeys run strictly serially: one at a time on the GPU
//   (Graphics Processing Unit), never in parallel.
// - Custom-kernel claims ship with numeric performance comparisons against
//   the MLX stock layer.
// - Every test carries a built-in timeout capped at 120 seconds.
