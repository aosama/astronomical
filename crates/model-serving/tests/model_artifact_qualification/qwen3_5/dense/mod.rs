mod mtp;
mod vision;

pub(super) fn qwen3_6_27b_oq4e_mtp_model_directory() -> std::path::PathBuf {
    crate::common::configured_model_artifact_directory_by_id("Qwen3.6-27B-oQ4e-mtp")
}
