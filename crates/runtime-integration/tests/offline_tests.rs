// The offline module root lives inside the offline/ folder (rather than
// beside it as tests/offline.rs) so cargo does not auto-discover the same
// file a second time as a standalone, uncompilable test target.
#[path = "offline/root.rs"]
mod offline;
