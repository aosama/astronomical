//! Shared machinery for the disposable-target journey tests: the sandboxed
//! fixture environment (`sandbox`) and the process/measurement utilities
//! (`process`). Kept beside the test root so the journey file itself reads
//! as the contracts it pins.

pub(crate) mod process;
pub(crate) mod sandbox;
