//! Reads expert records from storage straight into memory the caller owns.
//!
//! A model that does not fit in memory fetches experts for most reply
//! tokens. Measured on a layer's fetch of five experts, reading through
//! MLX's loader and then copying into the table that keeps them cost 1.47
//! ms; one scatter read per record straight into that table cost 0.89 ms
//! (`cargo measure-fetching-layer-stages`). This crate is that read: a record
//! is one contiguous span of a file, and one `preadv` fills a list of
//! destinations from it in order, so each tensor of an expert lands where it
//! is used. A batch of records is read from the calling thread and the
//! reader's long-lived helper threads at once.
//!
//! The reader itself is native code (`native/expert_record_reader.cpp`);
//! `record_reader` is its safe owner.

// The only module that needs unsafe code: it calls the native reader and
// hands it destination addresses. Every block carries its SAFETY argument.
#[allow(unsafe_code)]
mod record_reader;

pub use record_reader::{ExpertRecordReader, RecordRead, RecordReadError};
