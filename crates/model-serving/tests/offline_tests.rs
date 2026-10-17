// The model contract helpers are shared with the root acceptance package's
// GPU contracts; a single path declaration keeps the offline lane compiling
// against the one copy without duplicating fixture resolution logic.
#[path = "../../../tests/acceptance/tests/gpu_contracts/common/mod.rs"]
mod common;

mod artifacts;
mod families;
mod structural;
mod worker;
