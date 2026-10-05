// Main.swift — InferenceWorker
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): apps/inference-worker main and worker_startup*
// modules — process entry, IPC (Inter-Process Communication) connection
// handshake, and startup sequencing.
//
// Carried contracts:
// - Startup speaks the unchanged IpcProtocol surface; supervisor and worker
//   swap implementations in coordinated waves, never both ad hoc.
// - One real-model worker journey at a time on the GPU (Graphics Processing
//   Unit); hermetic CPU work may parallelize.
