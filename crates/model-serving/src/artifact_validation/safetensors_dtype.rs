use ::safetensors::Dtype;

use super::ArtifactValidationError;

pub(crate) fn dtype_bits_per_element(
    dtype_string: &str,
    file_name: &str,
    tensor_name: &str,
) -> Result<u64, ArtifactValidationError> {
    match dtype_string {
        "F64" => Ok(64),
        "F32" | "I32" | "U32" => Ok(32),
        "F16" | "BF16" | "I16" => Ok(16),
        "I64" => Ok(64),
        "I8" | "U8" | "BOOL" => Ok(8),
        _ => Err(ArtifactValidationError::UnknownSafetensorsDtype {
            file_name: file_name.to_owned(),
            tensor_name: tensor_name.to_owned(),
            dtype_string: dtype_string.to_owned(),
        }),
    }
}

pub(crate) fn parse_safetensors_dtype(
    dtype_string: &str,
    file_name: &str,
    tensor_name: &str,
) -> Result<Dtype, ArtifactValidationError> {
    match dtype_string {
        "F64" => Ok(Dtype::F64),
        "F32" => Ok(Dtype::F32),
        "F16" => Ok(Dtype::F16),
        "BF16" => Ok(Dtype::BF16),
        "I64" => Ok(Dtype::I64),
        "I32" => Ok(Dtype::I32),
        "I16" => Ok(Dtype::I16),
        "I8" => Ok(Dtype::I8),
        "U32" => Ok(Dtype::U32),
        "U8" => Ok(Dtype::U8),
        "BOOL" => Ok(Dtype::BOOL),
        _ => Err(ArtifactValidationError::UnknownSafetensorsDtype {
            file_name: file_name.to_owned(),
            tensor_name: tensor_name.to_owned(),
            dtype_string: dtype_string.to_owned(),
        }),
    }
}
