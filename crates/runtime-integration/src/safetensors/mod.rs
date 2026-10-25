//! The custom C-ABI safetensors input/output engines handed to MLX.
//!
//! # Why custom engines exist
//!
//! MLX-C loads and saves safetensors through a pluggable input/output
//! interface: callers provide a small C vtable (open/good/tell/seek/read/
//! write/free) and MLX drives it while materializing tensors. Astronomical
//! implements that vtable in Rust four times, because the product's core
//! behaviors need more than a file path can express:
//!
//! - **Retained descriptors, not paths** — opening by an owned `File` pins
//!   the on-disk identity; a model directory updated mid-session cannot
//!   change the bytes a lazy array reads later.
//! - **Bounded multi-range reads** — expert paging must load only the
//!   router-selected expert rows out of a shard that is far larger than one
//!   forward pass needs; the virtual-payload reader below is that mechanism.
//! - **Streaming writes without duplication** — a prompt-cache capture can be
//!   many gigabytes; serializing it into one Rust byte vector would
//!   temporarily duplicate the complete capture, so the writers hand MLX a
//!   descriptor and let it materialize one tensor at a time.
//! - **Observability** — every positional read can be counted and timed
//!   without synchronizing MLX execution.
//!
//! # Module map
//!
//! - `name_maps` — the RAII owners for MLX's string-to-array and
//!   string-to-string maps, shared by the loaders and both writers.
//! - `lazy_map` — the loading entry points and their result owners
//!   ([`MlxSafetensors`], [`SafetensorsLoadResult`], plus
//!   [`BoundedReadInterval`]).
//! - `descriptor_file_reader` — whole-file reading through one retained
//!   read-only descriptor.
//! - `bounded_range_reader` — the virtual-payload reader that maps selected
//!   source byte ranges into a synthetic safetensors layout.
//! - `bounded_range_read_concurrency` — the process-wide permit pool and
//!   partitioning that bound physical read parallelism.
//! - `file_writer` — descriptor-backed saving
//!   ([`MlxRuntime::save_safetensors`]).
//! - `memory_writer` — bounded in-memory serialization
//!   ([`MlxRuntime::serialize_safetensors`]).
//! - `read_metrics` — the lock-free positional-read counters threaded
//!   through every reader above.
//!
//! # The lifetime rule everything else follows
//!
//! Arrays returned by these engines are **lazy**: MLX may issue reads through
//! the vtable at first evaluation, long after the loading call returns. Every
//! result type therefore keeps its reader (and the descriptor inside it)
//! alive for as long as the arrays can still trigger reads — dropping the
//! reader early would turn deferred evaluation into a use-after-free across
//! the C boundary.

pub mod bounded_range_read_concurrency;
pub mod bounded_range_reader;
pub mod descriptor_file_reader;
pub mod file_writer;
pub mod lazy_map;
pub mod memory_writer;
pub mod name_maps;
pub mod read_metrics;
