//! Request state a text family keeps beyond its layers' attention and
//! recurrent state.
//!
//! The per-layer state the `astro_decoder_state_cache` layout describes is
//! enough for Qwen 3.5. A family that adds its own per-request memory (the
//! Flash-Next indexer's compressed keys and its n-gram history) puts it
//! here, so it travels with the request, joins every checkpoint and restore
//! the loop takes around a retried prompt chunk, and counts toward the
//! request's memory, without the shell knowing what the family stores.

use std::any::Any;

use mlx_bridge::MlxArray;
use runtime_integration::MlxRuntimeError;

/// One family's extra per-request state.
pub trait FamilyDecoderState: Any {
    /// Bytes the state holds, counted with the request's decoder state.
    fn payload_byte_count(&self) -> u64;

    /// Arrays a forward updates that no forward output depends on, so they
    /// must be evaluated with it; otherwise each step's lazy update would
    /// chain onto the last one's and the graph would grow without bound.
    fn append_evaluation_arrays<'state>(&'state self, arrays: &mut Vec<&'state MlxArray>);

    /// A copy a later restore returns to. Arrays are shared handles, so the
    /// copy costs no array memory until either side changes.
    fn snapshot(&self) -> Result<Box<dyn FamilyDecoderState>, MlxRuntimeError>;

    fn as_any(&self) -> &dyn Any;

    fn as_any_mut(&mut self) -> &mut dyn Any;
}
