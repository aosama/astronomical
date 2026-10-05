// InferenceWorkerTests.swift — InferenceWorkerTests
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Test target for Sources/InferenceWorker (wave 3). Tests mirror the source
// package paths.
//
// Carried contracts:
// - Real-model worker journeys run strictly serially on the GPU (Graphics
//   Processing Unit); hermetic tests may parallelize.
// - Performance attribution runs through these end-to-end journeys rather
//   than a manually driven development instance.
// - Every test carries a built-in timeout capped at 120 seconds.
