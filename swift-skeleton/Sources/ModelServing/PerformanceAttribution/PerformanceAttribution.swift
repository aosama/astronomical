// PerformanceAttribution.swift — ModelServing.PerformanceAttribution
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/performance_attribution/*
// (catalog, counter_catalog, measurement_catalog, log, report,
// expert_streaming_source, macos_process_io) plus the flat
// crates/model-serving/src/performance_attribution.rs.
//
// Carried contracts:
// - Logging is switchable through configuration, on and off.
// - Every attributed operation captures start time and end time, so slow
//   downs can be attributed to specific code parts instead of guessed at.
// - Priority order stays: model loading from disk, prompt processing,
//   tokenization, disk cache, expert paging, token generation.
// - Attribution runs through e2e (end-to-end) journeys, not a manually
//   driven development instance.
