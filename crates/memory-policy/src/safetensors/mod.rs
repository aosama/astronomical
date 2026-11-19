mod header;

pub use header::SAFETENSORS_HEADER_LENGTH_PREFIX_BYTES;
pub use header::{
    BoundedSafetensorsHeaderError, SafetensorsTensorView, read_bounded_safetensors_json_header,
};
