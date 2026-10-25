// The integration module root lives inside the integration/ folder (rather than
// beside it as tests/integration.rs) so cargo does not auto-discover the same
// file a second time as a standalone, uncompilable test target.
#[path = "integration/root.rs"]
mod integration;
