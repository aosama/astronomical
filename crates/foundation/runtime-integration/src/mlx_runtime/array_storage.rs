//! Host access to the storage of an array MLX has already evaluated.
//!
//! On Apple silicon an MLX array's storage is memory the processor and the
//! graphics processor share. A reader that writes bytes into that storage
//! puts them exactly where a later kernel reads them, with no copy and no
//! array built to carry them. Measured on a five-expert page read from
//! storage: 0.89 ms straight into a table's storage, against 0.99 ms through
//! MLX's loader plus 0.45 ms to copy the page into the table
//! (`cargo measure-fetching-layer-stages`), and a pass over the rows costs
//! the same either way.

use mlx_bridge::MlxArray;

use crate::MlxRuntimeError;

/// The bytes of `array`'s own storage, for writing from the host.
///
/// MLX treats an array as immutable, so this is sound only as bookkeeping
/// the caller keeps: write while no graph that reads the array is waiting to
/// be evaluated and no kernel that reads it is running, or that reader sees
/// some bytes from before the write and some from after. Arrays that share
/// this storage (a retained reference, a slice) see the written bytes, which
/// is the point.
///
/// # Errors
/// Fails when the array cannot be evaluated, or when it is a view into
/// another array's storage: its bytes would then not be its rows in order.
pub fn evaluated_array_storage_mut(array: &mut MlxArray) -> Result<&mut [u8], MlxRuntimeError> {
    let is_row_contiguous = array.is_row_contiguous().map_err(MlxRuntimeError::from)?;
    if !is_row_contiguous {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: "borrow evaluated array storage",
            description: "the array is a view and does not own contiguous storage".to_owned(),
        });
    }
    let storage_byte_count = array.byte_count();
    // SAFETY: the array is populated and evaluated (the contiguity query
    // evaluates it), so MLX returns the address of `byte_count` bytes that
    // live as long as the array, which the exclusive borrow outlives.
    let storage_address = unsafe { mlx_bridge::raw::mlx_array_data_uint8(*array.raw_mut()) };
    if storage_address.is_null() {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: "borrow evaluated array storage",
            description: "MLX returned no storage for an evaluated array".to_owned(),
        });
    }
    // SAFETY: as above; the slice covers exactly the array's storage and the
    // exclusive borrow of the array keeps any other Rust access out.
    Ok(unsafe { std::slice::from_raw_parts_mut(storage_address.cast_mut(), storage_byte_count) })
}
