//! Product-neutral Rust ownership for MLX paged-buffer storage.
//!
//! A slot is allocated once at its final size, filled by exact positional
//! ranges, committed, and then exposed through typed MLX views. Views retain the
//! shared storage independently, allowing the slot handle to leave scope without
//! copying payload bytes or invalidating lazy graphics-processor work.

use std::{ffi::CString, os::unix::ffi::OsStrExt, path::Path, ptr::NonNull};

use crate::{
    MlxArray, MlxDtype, MlxRuntimeError,
    mlx_runtime::{check_status, clear_captured_mlx_error},
    raw,
};

const PAGED_BUFFER_OPERATION: &str = "manage an MLX paged buffer slot";

/// One reusable native positional reader for a source file.
#[derive(Debug)]
pub struct MlxPagedFileReader {
    raw_reader: raw::mlx_paged_file_reader,
}

impl MlxPagedFileReader {
    pub fn new(source_file_path: &Path) -> Result<Self, MlxRuntimeError> {
        let source_file_path =
            CString::new(source_file_path.as_os_str().as_bytes()).map_err(|_| {
                MlxRuntimeError::RuntimeOperation {
                    operation: PAGED_BUFFER_OPERATION,
                    description: "paged source path contains an interior NUL byte".to_owned(),
                }
            })?;
        let mut raw_reader = raw::mlx_paged_file_reader {
            ctx: std::ptr::null_mut(),
        };
        clear_captured_mlx_error();
        // SAFETY: The path remains live for this copying constructor and the
        // output handle is uniquely writable.
        let status =
            unsafe { raw::mlx_paged_file_reader_new(&mut raw_reader, source_file_path.as_ptr()) };
        check_status(status, PAGED_BUFFER_OPERATION)?;
        NonNull::new(raw_reader.ctx).ok_or_else(|| MlxRuntimeError::RuntimeOperation {
            operation: PAGED_BUFFER_OPERATION,
            description: "MLX returned an empty paged file reader".to_owned(),
        })?;
        Ok(Self { raw_reader })
    }
}

impl Drop for MlxPagedFileReader {
    fn drop(&mut self) {
        // SAFETY: This owner releases its opaque reader exactly once.
        unsafe { raw::mlx_paged_file_reader_free(self.raw_reader) };
    }
}

/// Final aligned MLX-owned storage assembled from one or more source ranges.
///
/// Callers must finish every range write before `commit`; typed views are valid
/// only after commit and must remain within the allocated slot.
#[derive(Debug)]
pub struct MlxPagedBufferSlot {
    raw_slot: raw::mlx_paged_buffer_slot,
}

impl MlxPagedBufferSlot {
    pub fn new(byte_count: usize) -> Result<Self, MlxRuntimeError> {
        let mut raw_slot = raw::mlx_paged_buffer_slot {
            ctx: std::ptr::null_mut(),
        };
        clear_captured_mlx_error();
        // SAFETY: The output handle is uniquely writable.
        let status = unsafe { raw::mlx_paged_buffer_slot_new(&mut raw_slot, byte_count) };
        check_status(status, PAGED_BUFFER_OPERATION)?;
        NonNull::new(raw_slot.ctx).ok_or_else(|| MlxRuntimeError::RuntimeOperation {
            operation: PAGED_BUFFER_OPERATION,
            description: "MLX returned an empty paged buffer slot".to_owned(),
        })?;
        Ok(Self { raw_slot })
    }

    pub fn read_range(
        &self,
        source_reader: &MlxPagedFileReader,
        source_offset: u64,
        destination_offset: usize,
        byte_count: usize,
    ) -> Result<(), MlxRuntimeError> {
        // The native operation is synchronous: the descriptor and both opaque
        // owners may be stack-borrowed and no Rust slice crosses the boundary.
        let read_range = raw::mlx_paged_buffer_read_range {
            source_reader: source_reader.raw_reader,
            source_offset,
            destination_slot: self.raw_slot,
            destination_offset,
            byte_count,
        };
        clear_captured_mlx_error();
        // SAFETY: Both opaque owners and the range descriptor remain live for
        // the synchronous read operation.
        let status = unsafe { raw::mlx_read_paged_buffer_ranges(&read_range, 1) };
        check_status(status, PAGED_BUFFER_OPERATION)
    }

    pub fn commit(&self) -> Result<(), MlxRuntimeError> {
        clear_captured_mlx_error();
        // SAFETY: The opaque slot owner remains live.
        let status = unsafe { raw::mlx_paged_buffer_slot_commit(self.raw_slot) };
        check_status(status, PAGED_BUFFER_OPERATION)
    }

    pub fn view(
        &self,
        shape: &[i32],
        dtype: MlxDtype,
        byte_offset: usize,
        byte_count: usize,
    ) -> Result<MlxArray, MlxRuntimeError> {
        // MLX validates shape, dtype, offset, and byte length, then creates a
        // shared-storage array view. This is intentionally not a host copy.
        let mut output = MlxArray::empty();
        clear_captured_mlx_error();
        // SAFETY: The slot and shape remain live, and the output handle is
        // uniquely writable for this copying view constructor.
        let status = unsafe {
            raw::mlx_paged_buffer_slot_view(
                output.raw_mut(),
                self.raw_slot,
                shape.as_ptr(),
                shape.len(),
                dtype.to_raw(),
                byte_offset,
                byte_count,
            )
        };
        check_status(status, PAGED_BUFFER_OPERATION)?;
        output.require_populated(PAGED_BUFFER_OPERATION)?;
        Ok(output)
    }
}

impl Drop for MlxPagedBufferSlot {
    fn drop(&mut self) {
        // SAFETY: This owner releases its opaque slot exactly once. Views retain
        // the shared MLX storage independently.
        unsafe { raw::mlx_paged_buffer_slot_free(self.raw_slot) };
    }
}
