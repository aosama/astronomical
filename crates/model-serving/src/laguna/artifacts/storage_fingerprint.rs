use sha2::{Digest, Sha256};

use super::artifact_error::LagunaArtifactValidationError;
use super::canonical_tensor_contract::LagunaTensorContract;
use crate::laguna::LagunaTargetContract;

/// Hashes canonical metadata only; model payload bytes never enter this digest.
pub(super) fn storage_fingerprint(
    target_contract: &LagunaTargetContract,
    tensor_contract: &LagunaTensorContract,
    total_tensor_payload_bytes: u64,
) -> Result<[u8; 32], LagunaArtifactValidationError> {
    let mut fingerprint_hasher = Sha256::new();
    append_bytes(&mut fingerprint_hasher, b"astronomical-laguna-storage-v5")?;
    append_bytes(
        &mut fingerprint_hasher,
        format!("{target_contract:?}").as_bytes(),
    )?;
    fingerprint_hasher.update(total_tensor_payload_bytes.to_be_bytes());
    for (tensor_id, tensor_descriptor) in tensor_contract.descriptors() {
        append_bytes(&mut fingerprint_hasher, format!("{tensor_id:?}").as_bytes())?;
        append_bytes(
            &mut fingerprint_hasher,
            tensor_descriptor
                .canonical_module_name()
                .unwrap_or("<unprofiled>")
                .as_bytes(),
        )?;
        append_shape(&mut fingerprint_hasher, tensor_descriptor.logical_shape())?;
        append_bytes(
            &mut fingerprint_hasher,
            format!("{:?}", tensor_descriptor.execution_dtype()).as_bytes(),
        )?;
        append_bytes(
            &mut fingerprint_hasher,
            format!("{:?}", tensor_descriptor.storage_dtype()).as_bytes(),
        )?;
        append_bytes(
            &mut fingerprint_hasher,
            format!("{:?}", tensor_descriptor.storage_encoding()).as_bytes(),
        )?;
        append_bytes(
            &mut fingerprint_hasher,
            format!("{:?}", tensor_descriptor.assembly_kind()).as_bytes(),
        )?;
        for source in tensor_descriptor.sources() {
            append_bytes(
                &mut fingerprint_hasher,
                format!("{:?}", source.role()).as_bytes(),
            )?;
            append_bytes(&mut fingerprint_hasher, source.shard_file_name().as_bytes())?;
            fingerprint_hasher.update(source.payload_bytes().to_be_bytes());
            append_shape(&mut fingerprint_hasher, source.raw_shape())?;
            append_bytes(
                &mut fingerprint_hasher,
                format!("{:?}", source.raw_dtype()).as_bytes(),
            )?;
        }
    }
    for metadata in tensor_contract.non_executable_metadata() {
        append_bytes(
            &mut fingerprint_hasher,
            format!("{:?}", metadata.tensor_id()).as_bytes(),
        )?;
        for source in metadata.sources() {
            append_bytes(
                &mut fingerprint_hasher,
                format!("{:?}", source.role()).as_bytes(),
            )?;
            append_bytes(&mut fingerprint_hasher, source.shard_file_name().as_bytes())?;
            fingerprint_hasher.update(source.payload_bytes().to_be_bytes());
            append_shape(&mut fingerprint_hasher, source.raw_shape())?;
            append_bytes(
                &mut fingerprint_hasher,
                format!("{:?}", source.raw_dtype()).as_bytes(),
            )?;
        }
    }
    Ok(fingerprint_hasher.finalize().into())
}

fn append_shape(
    fingerprint_hasher: &mut Sha256,
    shape: &[usize],
) -> Result<(), LagunaArtifactValidationError> {
    let dimension_count = u64::try_from(shape.len())
        .map_err(|_| LagunaArtifactValidationError::TensorGeometryOverflow)?;
    fingerprint_hasher.update(dimension_count.to_be_bytes());
    for dimension in shape {
        let dimension = u64::try_from(*dimension)
            .map_err(|_| LagunaArtifactValidationError::TensorGeometryOverflow)?;
        fingerprint_hasher.update(dimension.to_be_bytes());
    }
    Ok(())
}

fn append_bytes(
    fingerprint_hasher: &mut Sha256,
    bytes: &[u8],
) -> Result<(), LagunaArtifactValidationError> {
    let byte_count = u64::try_from(bytes.len())
        .map_err(|_| LagunaArtifactValidationError::TensorGeometryOverflow)?;
    fingerprint_hasher.update(byte_count.to_be_bytes());
    fingerprint_hasher.update(bytes);
    Ok(())
}
