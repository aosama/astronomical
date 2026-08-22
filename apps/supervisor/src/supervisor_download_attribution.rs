//! Validated download-specific fields embedded in supervisor performance records.

use std::{
    io,
    path::{Component, Path},
};

use serde::Serialize;

use crate::library::{is_valid_huggingface_id, is_valid_immutable_revision};

const MAXIMUM_RELATIVE_PATH_BYTES: usize = 1_024;

#[derive(Clone, Debug, Serialize)]
pub(crate) struct SupervisorDownloadMeasurementDetail {
    huggingface_id: String,
    revision: String,
    #[serde(flatten)]
    pub(crate) operation_detail: SupervisorDownloadOperationDetail,
}

impl SupervisorDownloadMeasurementDetail {
    pub(crate) fn new(
        huggingface_id: impl Into<String>,
        revision: impl Into<String>,
        operation_detail: SupervisorDownloadOperationDetail,
    ) -> io::Result<Self> {
        let huggingface_id = huggingface_id.into();
        let revision = revision.into();
        if !is_valid_huggingface_id(&huggingface_id) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "supervisor attribution requires a validated Hugging Face identity",
            ));
        }
        if !is_valid_immutable_revision(&revision) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "supervisor attribution requires a validated immutable revision",
            ));
        }
        Ok(Self::validated(huggingface_id, revision, operation_detail))
    }

    pub(crate) fn validated(
        huggingface_id: impl Into<String>,
        revision: impl Into<String>,
        operation_detail: SupervisorDownloadOperationDetail,
    ) -> Self {
        Self {
            huggingface_id: huggingface_id.into(),
            revision: revision.into(),
            operation_detail,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub(crate) enum SupervisorDownloadOperationDetail {
    DiskPreflight {
        required_bytes: u64,
        available_bytes: u64,
    },
    ManifestFetch {
        manifest_file_count: usize,
        manifest_total_bytes: u64,
    },
    FileTransfer {
        relative_file_path: String,
        resume_offset_bytes: u64,
        transferred_bytes: u64,
    },
    Verification {
        verified_file_count: usize,
        verified_bytes: u64,
    },
    Publication {},
    DiscoveryRefresh {},
}

pub(crate) fn is_safe_relative_path(relative_path: &str) -> bool {
    if relative_path.is_empty()
        || relative_path.len() > MAXIMUM_RELATIVE_PATH_BYTES
        || !relative_path.is_ascii()
        || relative_path.contains('\\')
        || relative_path.chars().any(char::is_control)
    {
        return false;
    }
    let path = Path::new(relative_path);
    if path.is_absolute()
        || !path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
    {
        return false;
    }
    path.components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
        == relative_path
}
