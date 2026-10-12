// The hermetic module root lives inside the hermetic/ folder (rather than
// beside it as tests/hermetic.rs) so cargo does not auto-discover the same
// file a second time as a standalone, uncompilable test target.
#[path = "hermetic/root.rs"]
mod hermetic;
