//! RAII owners for MLX's string-keyed maps, shared by loaders and writers.
//!
//! # Why one shared module
//!
//! The safetensors format is a name-to-tensor map plus a name-to-string
//! metadata map, and MLX-C represents both as opaque C handles. Two loading
//! paths (whole-file and bounded-range) and two writing paths (file-backed
//! and in-memory) all need the same pair of handles with the same rules:
//!
//! - the handle is created empty, then either pre-filled by the caller
//!   (writers) or filled by MLX-C itself as the output of a load;
//! - MLX-C copies keys (and retains array references) on insert, so nothing
//!   in this module borrows caller data beyond the insert call;
//! - the handle must be released exactly once, which the `Drop` impls below
//!   guarantee even when a later insert fails — the map enters RAII ownership
//!   *before* the first insertion for exactly that reason.
//!
//! Centralizing the owners here also keeps the memory writer from reaching
//! into the file writer's module for shared types: the writers are peers,
//! and this module is the dependency both are allowed to share.

use std::ffi::CString;

use crate::MlxRuntimeError;
use mlx_c_rust::{MlxArray, raw};

/// Owner of one MLX string-to-array map (the tensor map).
///
/// `Debug` is derived because [`MlxSafetensors`](super::lazy_map::MlxSafetensors)
/// holds this map as its lifetime keeper and is itself `Debug`.
#[derive(Debug)]
pub(crate) struct OwnedTensorMap(pub(crate) raw::mlx_map_string_to_array);

impl OwnedTensorMap {
    /// Creates an empty tensor map for a loading path to fill.
    pub(crate) fn new() -> Result<Self, MlxRuntimeError> {
        // SAFETY: The returned handle is placed under RAII ownership.
        let raw_map = unsafe { raw::mlx_map_string_to_array_new() };
        if raw_map.ctx.is_null() {
            return Err(empty_handle_error("allocate an MLX tensor map"));
        }
        Ok(Self(raw_map))
    }

    /// Creates a tensor map pre-filled with named arrays for a save or
    /// serialization call.
    ///
    /// Insertion order is preserved as given; the safetensors format has no
    /// ordering requirement, but deterministic input keeps serialized bytes
    /// reproducible for the same logical capture.
    pub(crate) fn from_entries(
        named_arrays: &[(&str, &MlxArray)],
    ) -> Result<Self, MlxRuntimeError> {
        // SAFETY: The returned handle enters RAII ownership immediately.
        let raw_map = unsafe { raw::mlx_map_string_to_array_new() };
        if raw_map.ctx.is_null() {
            return Err(empty_handle_error("allocate an MLX tensor map"));
        }
        // Enter RAII ownership before inserting anything so an invalid later
        // name or failed insert still releases the native map exactly once.
        let owned_map = Self(raw_map);
        for (tensor_name, tensor) in named_arrays {
            let tensor_name = checked_c_string(tensor_name, "insert a safetensors tensor name")?;
            // SAFETY: The map and tensor are live; MLX copies the key and retains array ownership.
            let insert_status = unsafe {
                raw::mlx_map_string_to_array_insert(owned_map.0, tensor_name.as_ptr(), tensor.raw())
            };
            crate::mlx_runtime::check_status(insert_status, "insert a safetensors tensor")?;
        }
        Ok(owned_map)
    }
}

impl Drop for OwnedTensorMap {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live map exactly once.
        unsafe {
            raw::mlx_map_string_to_array_free(self.0);
        }
    }
}

/// Owner of one MLX string-to-string map (the metadata map).
#[derive(Debug)]
pub(crate) struct OwnedMetadataMap(pub(crate) raw::mlx_map_string_to_string);

impl OwnedMetadataMap {
    /// Creates an empty metadata map for a loading path.
    ///
    /// Loading paths populate the map through the MLX-C loader itself (the
    /// metadata arrives from the file header), so this constructor only
    /// allocates the destination handle.
    pub(crate) fn new() -> Result<Self, MlxRuntimeError> {
        // SAFETY: The returned handle is placed under RAII ownership.
        let raw_map = unsafe { raw::mlx_map_string_to_string_new() };
        if raw_map.ctx.is_null() {
            return Err(empty_handle_error("allocate an MLX metadata map"));
        }
        Ok(Self(raw_map))
    }

    /// Creates a metadata map pre-filled with entries for a save or
    /// serialization call.
    pub(crate) fn from_entries(metadata_entries: &[(&str, &str)]) -> Result<Self, MlxRuntimeError> {
        // SAFETY: The returned handle enters RAII ownership immediately.
        let raw_map = unsafe { raw::mlx_map_string_to_string_new() };
        if raw_map.ctx.is_null() {
            return Err(empty_handle_error("allocate an MLX metadata map"));
        }
        let owned_map = Self(raw_map);
        for (metadata_name, metadata_value) in metadata_entries {
            let metadata_name = checked_c_string(metadata_name, "insert a metadata name")?;
            let metadata_value = checked_c_string(metadata_value, "insert a metadata value")?;
            // SAFETY: The map is live and MLX copies both strings during this call.
            let insert_status = unsafe {
                raw::mlx_map_string_to_string_insert(
                    owned_map.0,
                    metadata_name.as_ptr(),
                    metadata_value.as_ptr(),
                )
            };
            crate::mlx_runtime::check_status(insert_status, "insert safetensors metadata")?;
        }
        Ok(owned_map)
    }
}

impl Drop for OwnedMetadataMap {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live map exactly once.
        unsafe {
            raw::mlx_map_string_to_string_free(self.0);
        }
    }
}

/// Converts a Rust string for the C boundary, rejecting interior NUL bytes.
///
/// A C string cannot contain an interior NUL; failing loudly here is far
/// better than a silent truncation that would insert a tensor under a
/// different name than the caller asked for.
fn checked_c_string(text: &str, operation: &'static str) -> Result<CString, MlxRuntimeError> {
    CString::new(text).map_err(|_| MlxRuntimeError::RuntimeOperation {
        operation,
        description: "text contains an interior NUL byte".to_owned(),
    })
}

/// The shared shape for "MLX handed back an empty handle".
///
/// Every constructor above funnels null handles through one helper so the
/// wording stays identical across loading and writing paths.
pub(super) fn empty_handle_error(operation: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation,
        description: "MLX returned an empty handle".to_owned(),
    }
}
