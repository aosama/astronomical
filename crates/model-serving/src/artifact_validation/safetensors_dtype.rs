use ::safetensors::Dtype;

use super::ArtifactValidationError;

pub(crate) fn checked_safetensors_payload_bytes(
    element_count: u64,
    bits_per_element: u64,
) -> Result<Option<u64>, ArtifactValidationError> {
    // Eight elements always occupy exactly `bits_per_element` bytes. Grouping
    // first avoids overflowing an intermediate bit count when bytes still fit.
    let complete_eight_element_groups = element_count / 8;
    let trailing_element_count = element_count % 8;
    let complete_group_bytes = complete_eight_element_groups
        .checked_mul(bits_per_element)
        .ok_or(ArtifactValidationError::TensorPayloadSizeOverflow)?;
    let trailing_bits = trailing_element_count
        .checked_mul(bits_per_element)
        .ok_or(ArtifactValidationError::TensorPayloadSizeOverflow)?;
    if trailing_bits % 8 != 0 {
        return Ok(None);
    }
    let payload_bytes = complete_group_bytes
        .checked_add(trailing_bits / 8)
        .ok_or(ArtifactValidationError::TensorPayloadSizeOverflow)?;
    Ok(Some(payload_bytes))
}

pub(crate) fn dtype_bits_per_element(
    dtype_string: &str,
    file_name: &str,
    tensor_name: &str,
) -> Result<u64, ArtifactValidationError> {
    match dtype_string {
        "F4" => Ok(4),
        "F6_E2M3" | "F6_E3M2" => Ok(6),
        "BOOL" | "U8" | "I8" | "F8_E5M2" | "F8_E4M3" | "F8_E8M0" | "F8_E4M3FNUZ"
        | "F8_E5M2FNUZ" => Ok(8),
        "I16" | "U16" | "F16" | "BF16" => Ok(16),
        "I32" | "U32" | "F32" => Ok(32),
        "C64" | "F64" | "I64" | "U64" => Ok(64),
        _ => Err(unknown_safetensors_dtype(
            dtype_string,
            file_name,
            tensor_name,
        )),
    }
}

pub(crate) fn parse_safetensors_dtype(
    dtype_string: &str,
    file_name: &str,
    tensor_name: &str,
) -> Result<Dtype, ArtifactValidationError> {
    match dtype_string {
        "BOOL" => Ok(Dtype::BOOL),
        "F4" => Ok(Dtype::F4),
        "F6_E2M3" => Ok(Dtype::F6_E2M3),
        "F6_E3M2" => Ok(Dtype::F6_E3M2),
        "U8" => Ok(Dtype::U8),
        "I8" => Ok(Dtype::I8),
        "F8_E5M2" => Ok(Dtype::F8_E5M2),
        "F8_E4M3" => Ok(Dtype::F8_E4M3),
        "F8_E8M0" => Ok(Dtype::F8_E8M0),
        "F8_E4M3FNUZ" => Ok(Dtype::F8_E4M3FNUZ),
        "F8_E5M2FNUZ" => Ok(Dtype::F8_E5M2FNUZ),
        "I16" => Ok(Dtype::I16),
        "U16" => Ok(Dtype::U16),
        "F16" => Ok(Dtype::F16),
        "BF16" => Ok(Dtype::BF16),
        "I32" => Ok(Dtype::I32),
        "U32" => Ok(Dtype::U32),
        "F32" => Ok(Dtype::F32),
        "F64" => Ok(Dtype::F64),
        "I64" => Ok(Dtype::I64),
        "C64" => Ok(Dtype::C64),
        "U64" => Ok(Dtype::U64),
        _ => Err(unknown_safetensors_dtype(
            dtype_string,
            file_name,
            tensor_name,
        )),
    }
}

/// Parses every dtype understood by the pinned SafeTensors format dependency.
pub(crate) fn parse_raw_safetensors_dtype(
    dtype_string: &str,
    file_name: &str,
    tensor_name: &str,
) -> Result<Dtype, ArtifactValidationError> {
    match dtype_string {
        "BOOL" => Ok(Dtype::BOOL),
        "F4" => Ok(Dtype::F4),
        "F6_E2M3" => Ok(Dtype::F6_E2M3),
        "F6_E3M2" => Ok(Dtype::F6_E3M2),
        "U8" => Ok(Dtype::U8),
        "I8" => Ok(Dtype::I8),
        "F8_E5M2" => Ok(Dtype::F8_E5M2),
        "F8_E4M3" => Ok(Dtype::F8_E4M3),
        "F8_E8M0" => Ok(Dtype::F8_E8M0),
        "F8_E4M3FNUZ" => Ok(Dtype::F8_E4M3FNUZ),
        "F8_E5M2FNUZ" => Ok(Dtype::F8_E5M2FNUZ),
        "I16" => Ok(Dtype::I16),
        "U16" => Ok(Dtype::U16),
        "F16" => Ok(Dtype::F16),
        "BF16" => Ok(Dtype::BF16),
        "I32" => Ok(Dtype::I32),
        "U32" => Ok(Dtype::U32),
        "F32" => Ok(Dtype::F32),
        "C64" => Ok(Dtype::C64),
        "F64" => Ok(Dtype::F64),
        "I64" => Ok(Dtype::I64),
        "U64" => Ok(Dtype::U64),
        _ => Err(unknown_safetensors_dtype(
            dtype_string,
            file_name,
            tensor_name,
        )),
    }
}

fn unknown_safetensors_dtype(
    dtype_string: &str,
    file_name: &str,
    tensor_name: &str,
) -> ArtifactValidationError {
    ArtifactValidationError::UnknownSafetensorsDtype {
        file_name: file_name.to_owned(),
        tensor_name: tensor_name.to_owned(),
        dtype_string: dtype_string.to_owned(),
    }
}
