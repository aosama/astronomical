use std::{
    ffi::{CString, c_char, c_int, c_void},
    fs::File,
    os::unix::fs::FileExt,
    sync::{Arc, Mutex},
};

use crate::{
    MlxRuntimeError,
    mlx_array::MlxArray,
    mlx_bounded_safetensors_reader::{
        BoundedMultiRangeReaderState, ExpertSsdReadMetrics, OwnedBoundedMultiRangeReader,
        OwnedBoundedMultiRangeReaderHolder,
    },
    mlx_runtime::check_status,
    mlx_stream::MlxStream,
    raw,
};

const READER_LABEL: &[u8] = b"validated descriptor-backed weights\0";

/// One virtual interval mapping payload offsets to source file reads.
#[derive(Clone, Debug)]
pub struct BoundedReadInterval {
    /// Byte offset in the virtual payload where this interval starts.
    pub virtual_payload_offset: u64,
    /// Byte offset in the source file where reading begins.
    pub source_file_offset: u64,
    /// Number of bytes to read from the source file for this interval.
    pub source_byte_count: usize,
}

/// Loads safetensors tensors from bounded multi-range reads on a source file.
///
/// Constructs a synthetic safetensors header from the provided header bytes and
/// maps virtual payload offsets through the interval list to exact byte ranges
/// in the source file. This is the core I/O primitive for expert paging: it
/// reads only the selected expert rows from the safetensors shard, never the
/// full shard file.
///
/// The interval list must be contiguous in virtual payload space (each
/// interval's virtual_payload_offset equals the sum of all preceding
/// source_byte_counts) and must not contain overlapping source file ranges.
pub(crate) fn load_safetensors_from_bounded_ranges(
    source_file: File,
    synthetic_header_bytes: Vec<u8>,
    intervals: Vec<BoundedReadInterval>,
    total_payload_bytes: u64,
    stream: &MlxStream,
    expert_ssd_read_metrics: Option<Arc<ExpertSsdReadMetrics>>,
) -> Result<SafetensorsLoadResult, MlxRuntimeError> {
    let reader_state = BoundedMultiRangeReaderState::new(
        source_file,
        synthetic_header_bytes,
        intervals,
        total_payload_bytes,
        expert_ssd_read_metrics,
    )?;
    let reader = OwnedBoundedMultiRangeReader::new(reader_state)?;
    let mut tensor_map = OwnedTensorMap::new()?;
    let mut metadata_map = OwnedMetadataMap::new()?;
    // SAFETY: All output handles are live and uniquely borrowed. The reader
    // owns a valid callback table and remains alive in the returned owner.
    let status = unsafe {
        raw::mlx_load_safetensors_reader(
            &mut tensor_map.0,
            &mut metadata_map.0,
            reader.0,
            stream.raw(),
        )
    };
    check_status(
        status,
        "load safetensors through a bounded multi-range reader",
    )?;
    Ok(SafetensorsLoadResult {
        tensor_map,
        _metadata_map: metadata_map,
        _reader: OwnedBoundedMultiRangeReaderHolder(reader.0),
    })
}

/// Result of loading safetensors through a bounded multi-range reader.
pub struct SafetensorsLoadResult {
    tensor_map: OwnedTensorMap,
    _metadata_map: OwnedMetadataMap,
    _reader: OwnedBoundedMultiRangeReaderHolder,
}

impl SafetensorsLoadResult {
    /// Returns an owned reference-counted handle for one named tensor.
    pub fn tensor(&self, tensor_name: &str) -> Result<MlxArray, MlxRuntimeError> {
        let tensor_name =
            CString::new(tensor_name).map_err(|_| MlxRuntimeError::RuntimeOperation {
                operation: "look up a safetensors tensor",
                description: "tensor name contains an interior null byte".to_owned(),
            })?;
        let mut tensor = MlxArray::empty();
        // SAFETY: The map and output array are live, and the C string remains
        // valid for the duration of this non-retaining lookup call.
        let status = unsafe {
            raw::mlx_map_string_to_array_get(
                tensor.raw_mut(),
                self.tensor_map.0,
                tensor_name.as_ptr(),
            )
        };
        check_status(status, "look up a safetensors tensor")?;
        if tensor.is_empty() {
            return Err(empty_handle_error("look up a safetensors tensor"));
        }
        Ok(tensor)
    }

    /// Returns all tensor names in the loaded safetensors map.
    pub fn tensor_names(&self) -> Vec<String> {
        // MLX doesn't expose a direct iteration API for string-to-array maps.
        // The caller should know the tensor names from the manifest.
        Vec::new()
    }
}

/// Tensor map loaded lazily from one retained read-only weights descriptor.
#[derive(Debug)]
pub struct MlxSafetensors {
    tensor_map: OwnedTensorMap,
    _metadata_map: OwnedMetadataMap,
    _reader: OwnedFileReader,
}

impl MlxSafetensors {
    pub(crate) fn load(weights_file: File) -> Result<Self, MlxRuntimeError> {
        let reader = OwnedFileReader::new(weights_file)?;
        let stream = MlxStream::default_cpu()?;
        let mut tensor_map = OwnedTensorMap::new()?;
        let mut metadata_map = OwnedMetadataMap::new()?;
        // SAFETY: All output handles are live and uniquely borrowed. The reader
        // owns a valid callback table and remains alive in the returned owner.
        let status = unsafe {
            raw::mlx_load_safetensors_reader(
                &mut tensor_map.0,
                &mut metadata_map.0,
                reader.0,
                stream.raw(),
            )
        };
        check_status(
            status,
            "load safetensors through a retained file descriptor",
        )?;
        Ok(Self {
            tensor_map,
            _metadata_map: metadata_map,
            _reader: reader,
        })
    }

    /// Returns an owned reference-counted handle for one named tensor.
    pub fn tensor(&self, tensor_name: &str) -> Result<MlxArray, MlxRuntimeError> {
        let tensor_name =
            CString::new(tensor_name).map_err(|_| MlxRuntimeError::RuntimeOperation {
                operation: "look up a safetensors tensor",
                description: "tensor name contains an interior null byte".to_owned(),
            })?;
        let mut tensor = MlxArray::empty();
        // SAFETY: The map and output array are live, and the C string remains
        // valid for the duration of this non-retaining lookup call.
        let status = unsafe {
            raw::mlx_map_string_to_array_get(
                tensor.raw_mut(),
                self.tensor_map.0,
                tensor_name.as_ptr(),
            )
        };
        check_status(status, "look up a safetensors tensor")?;
        if tensor.is_empty() {
            return Err(empty_handle_error("look up a safetensors tensor"));
        }
        Ok(tensor)
    }
}

#[derive(Debug)]
struct OwnedTensorMap(raw::mlx_map_string_to_array);

impl OwnedTensorMap {
    fn new() -> Result<Self, MlxRuntimeError> {
        // SAFETY: The returned handle is placed under RAII ownership.
        let raw_map = unsafe { raw::mlx_map_string_to_array_new() };
        if raw_map.ctx.is_null() {
            return Err(empty_handle_error("allocate an MLX tensor map"));
        }
        Ok(Self(raw_map))
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

#[derive(Debug)]
struct OwnedMetadataMap(raw::mlx_map_string_to_string);

impl OwnedMetadataMap {
    fn new() -> Result<Self, MlxRuntimeError> {
        // SAFETY: The returned handle is placed under RAII ownership.
        let raw_map = unsafe { raw::mlx_map_string_to_string_new() };
        if raw_map.ctx.is_null() {
            return Err(empty_handle_error("allocate an MLX metadata map"));
        }
        Ok(Self(raw_map))
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

#[derive(Debug)]
struct OwnedFileReader(raw::mlx_io_reader);

impl OwnedFileReader {
    fn new(weights_file: File) -> Result<Self, MlxRuntimeError> {
        let descriptor = Box::into_raw(Box::new(Mutex::new(FileReaderState::new(weights_file))))
            .cast::<c_void>();
        let reader_vtable = raw::mlx_io_vtable {
            is_open: Some(reader_is_open),
            good: Some(reader_is_good),
            tell: Some(reader_tell),
            seek: Some(reader_seek),
            read: Some(reader_read),
            read_at_offset: Some(reader_read_at_offset),
            write: Some(reader_write_is_unsupported),
            label: Some(reader_label),
            free: Some(reader_free),
        };
        // SAFETY: `descriptor` points to an owned boxed reader state and every
        // callback matches the official C ABI. On success MLX owns descriptor
        // cleanup through the vtable.
        let raw_reader = unsafe { raw::mlx_io_reader_new(descriptor, reader_vtable) };
        if raw_reader.ctx.is_null() {
            // SAFETY: In the pinned MLX C constructor, ownership transfers only
            // after both holder allocations succeed. A null handle therefore
            // leaves the original descriptor with this caller.
            unsafe {
                drop(Box::from_raw(descriptor.cast::<Mutex<FileReaderState>>()));
            }
            return Err(empty_handle_error("allocate an MLX descriptor reader"));
        }
        Ok(Self(raw_reader))
    }
}

impl Drop for OwnedFileReader {
    fn drop(&mut self) {
        // SAFETY: This owner releases the MLX reader exactly once. MLX retains
        // callback state until all lazy arrays release their shared reader.
        unsafe {
            raw::mlx_io_reader_free(self.0);
        }
    }
}

#[derive(Debug)]
struct FileReaderState {
    file: File,
    position_bytes: u64,
    is_good: bool,
}

impl FileReaderState {
    fn new(file: File) -> Self {
        Self {
            file,
            position_bytes: 0,
            is_good: true,
        }
    }
}

unsafe extern "C" fn reader_is_open(descriptor: *mut c_void) -> bool {
    !descriptor.is_null()
}

unsafe extern "C" fn reader_is_good(descriptor: *mut c_void) -> bool {
    // SAFETY: MLX calls this vtable only while it owns the boxed descriptor.
    unsafe { with_reader_state(descriptor, |reader_state| reader_state.is_good) }.unwrap_or(false)
}

unsafe extern "C" fn reader_tell(descriptor: *mut c_void) -> usize {
    // SAFETY: MLX calls this vtable only while it owns the boxed descriptor.
    unsafe {
        with_reader_state(descriptor, |reader_state| {
            usize::try_from(reader_state.position_bytes).ok()
        })
    }
    .flatten()
    .unwrap_or(0)
}

unsafe extern "C" fn reader_seek(descriptor: *mut c_void, offset: i64, whence: c_int) {
    // SAFETY: MLX calls this vtable only while it owns the boxed descriptor.
    unsafe {
        with_reader_state(descriptor, |reader_state| {
            let base_position_bytes = match whence {
                libc::SEEK_SET => 0_i128,
                libc::SEEK_CUR => i128::from(reader_state.position_bytes),
                libc::SEEK_END => match reader_state.file.metadata() {
                    Ok(metadata) => i128::from(metadata.len()),
                    Err(_) => {
                        reader_state.is_good = false;
                        return;
                    }
                },
                _ => {
                    reader_state.is_good = false;
                    return;
                }
            };
            let requested_position_bytes = base_position_bytes + i128::from(offset);
            match u64::try_from(requested_position_bytes) {
                Ok(position_bytes) => reader_state.position_bytes = position_bytes,
                Err(_) => reader_state.is_good = false,
            }
        });
    }
}

unsafe extern "C" fn reader_read(
    descriptor: *mut c_void,
    destination: *mut c_char,
    byte_count: usize,
) {
    // SAFETY: MLX calls this vtable only while it owns the boxed descriptor.
    unsafe {
        with_reader_state(descriptor, |reader_state| {
            if read_exact_at(
                &reader_state.file,
                destination,
                byte_count,
                reader_state.position_bytes,
            )
            .is_err()
            {
                reader_state.is_good = false;
                return;
            }
            let Ok(consumed_byte_count) = u64::try_from(byte_count) else {
                reader_state.is_good = false;
                return;
            };
            match reader_state.position_bytes.checked_add(consumed_byte_count) {
                Some(position_bytes) => reader_state.position_bytes = position_bytes,
                None => reader_state.is_good = false,
            }
        });
    }
}

unsafe extern "C" fn reader_read_at_offset(
    descriptor: *mut c_void,
    destination: *mut c_char,
    byte_count: usize,
    offset_bytes: usize,
) {
    let Ok(offset_bytes) = u64::try_from(offset_bytes) else {
        // SAFETY: MLX calls this vtable only while it owns the boxed descriptor.
        unsafe {
            with_reader_state(descriptor, |reader_state| {
                reader_state.is_good = false;
            });
        }
        return;
    };
    // SAFETY: MLX calls this vtable only while it owns the boxed descriptor.
    unsafe {
        with_reader_state(descriptor, |reader_state| {
            if read_exact_at(&reader_state.file, destination, byte_count, offset_bytes).is_err() {
                reader_state.is_good = false;
            }
        });
    }
}

unsafe extern "C" fn reader_write_is_unsupported(
    descriptor: *mut c_void,
    _source: *const c_char,
    _byte_count: usize,
) {
    // SAFETY: MLX calls this vtable only while it owns the boxed descriptor.
    unsafe {
        with_reader_state(descriptor, |reader_state| {
            reader_state.is_good = false;
        });
    }
}

unsafe extern "C" fn reader_label(_descriptor: *mut c_void) -> *const c_char {
    READER_LABEL.as_ptr().cast::<c_char>()
}

unsafe extern "C" fn reader_free(descriptor: *mut c_void) {
    if descriptor.is_null() {
        return;
    }
    // SAFETY: `descriptor` originated from `Box::into_raw` in the constructor,
    // and MLX invokes this callback exactly once when its final reader dies.
    unsafe {
        drop(Box::from_raw(descriptor.cast::<Mutex<FileReaderState>>()));
    }
}

unsafe fn with_reader_state<Output>(
    descriptor: *mut c_void,
    operation: impl FnOnce(&mut FileReaderState) -> Output,
) -> Option<Output> {
    if descriptor.is_null() {
        return None;
    }
    // SAFETY: MLX retains the boxed mutex until the final reader release, and
    // no callback can execute after `reader_free` destroys the descriptor.
    let reader_mutex = unsafe { &*descriptor.cast::<Mutex<FileReaderState>>() };
    let mut reader_state = reader_mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    Some(operation(&mut reader_state))
}

fn read_exact_at(
    file: &File,
    destination: *mut c_char,
    byte_count: usize,
    offset_bytes: u64,
) -> std::io::Result<()> {
    if byte_count == 0 {
        return Ok(());
    }
    if destination.is_null() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "MLX reader supplied a null destination",
        ));
    }
    // SAFETY: MLX supplies writable storage for exactly `byte_count` bytes and
    // does not retain this temporary Rust slice after the callback returns.
    let destination_bytes =
        unsafe { std::slice::from_raw_parts_mut(destination.cast::<u8>(), byte_count) };
    file.read_exact_at(destination_bytes, offset_bytes)
}

fn empty_handle_error(operation: &'static str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation,
        description: "MLX returned an empty handle".to_owned(),
    }
}
