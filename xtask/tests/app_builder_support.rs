//! Shared machinery for the app-builder journey tests: the sandboxed
//! repository copy with its fixture-bin tool shims (`sandbox`), and the
//! journey contracts with their bundle-reading assertion helpers
//! (`journey`). Kept beside the test root so the journey file itself reads
//! as the four contracts it pins.

pub(crate) mod journey;
pub(crate) mod sandbox;
