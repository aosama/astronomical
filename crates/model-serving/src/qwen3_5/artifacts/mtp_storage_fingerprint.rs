//! Metadata-only fingerprinting for independently packaged MTP storage.

use sha2::{Digest, Sha256};

use crate::artifact_validation::ValidatedSafetensorsSource;
use crate::{TensorInventory, TensorProfile};

pub(super) fn standalone_mtp_storage_fingerprint(
    revision: &str,
    inventory: &TensorInventory,
    profiles: &[TensorProfile],
    sources: &[ValidatedSafetensorsSource],
) -> String {
    let mut fingerprint = Sha256::new();
    update_field(
        &mut fingerprint,
        b"astronomical-qwen-standalone-mtp-storage-v1",
    );
    update_field(&mut fingerprint, revision.as_bytes());
    for profile in profiles {
        update_field(&mut fingerprint, profile.name.as_bytes());
        update_field(&mut fingerprint, format!("{:?}", profile.dtype).as_bytes());
        for dimension in &profile.shape {
            fingerprint.update(dimension.to_le_bytes());
        }
    }
    for source in sources {
        update_field(&mut fingerprint, source.file_name().as_bytes());
        fingerprint.update(source.payload_bytes().to_le_bytes());
        for (stored_name, dtype, shape) in source.physical_tensor_metadata() {
            update_field(&mut fingerprint, stored_name.as_bytes());
            update_field(&mut fingerprint, dtype.as_bytes());
            for dimension in shape {
                fingerprint.update(dimension.to_le_bytes());
            }
            if let Some(location) = inventory.locations().find(|location| {
                location.source_id() == source.source_id() && location.stored_name() == stored_name
            }) {
                update_field(&mut fingerprint, location.canonical_name().as_bytes());
            }
        }
    }
    fingerprint
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn update_field(fingerprint: &mut Sha256, field: &[u8]) {
    fingerprint.update((field.len() as u64).to_le_bytes());
    fingerprint.update(field);
}
