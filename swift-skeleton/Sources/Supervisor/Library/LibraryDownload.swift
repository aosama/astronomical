// LibraryDownload.swift — Supervisor.Library
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 2): apps/supervisor/src/library/* — download_catalog,
// download_coordinator, download_disk_preflight, catalog_endpoint,
// catalog_projection.
//
// Carried contracts:
// - Catalog and download bookkeeping stays with the supervisor's library
//   surface; the config crate stays free of download concerns.
// - Disk preflight checks free space in decimal SI units (1 GB =
//   1,000,000,000 bytes) for anything user-facing.
// - Download progress emits a live indicator, never silent output.
