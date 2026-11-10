//! Reading and writing model weight files, shared by the model converters.
//!
//! Each converter (`convert-laya`, `convert-kev`) turns one published
//! checkpoint layout into the one Astronomical serves. What they share is
//! the file format underneath: the safetensors container, the 16-bit float
//! encodings its tensors use, and the restricted readers for formats a
//! converter must read but never execute.

pub(crate) mod bfloat16;
pub(crate) mod conversion_progress;
pub(crate) mod half_float;
pub(crate) mod local_paths;
pub(crate) mod safetensors_file;
pub(crate) mod stored_zip;
pub(crate) mod tensor_values;
#[cfg(test)]
pub(crate) mod test_support;
pub(crate) mod torch_pickle;
