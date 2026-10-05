// DownloadJob.swift — Supervisor.Library.DownloadJob
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 2): apps/supervisor/src/library/download_job/* —
// path_validation.rs, transitions.rs (the download state machine).
//
// Carried contracts:
// - Path validation is its own step in the state machine; destination
//   paths are never constructed from unvalidated user input.
// - Job transitions emit progress events so downloads never run silently.
