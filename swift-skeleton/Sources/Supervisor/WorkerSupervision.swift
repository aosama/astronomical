// WorkerSupervision.swift — Supervisor
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 2): apps/supervisor worker* modules — spawning
// inference workers, health checking, restart policy, and IPC command
// dispatch over FrameTransport.
//
// Carried contracts:
// - Worker control flows through the unchanged IpcProtocol surface — this
//   is the wave 2 strangler seam while workers are still Rust.
// - One real-model worker journey at a time; concurrent real-model journeys
//   can starve the GPU (Graphics Processing Unit) watchdog and force a
//   power-off.
