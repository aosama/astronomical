mod mtp;
mod mtp_depth_qualification;
mod vision;

pub(super) fn qwen3_8_27b_mtplx_model_directory() -> std::path::PathBuf {
    crate::common::configured_model_artifact_directory_by_id("Qwen3.8-27B-MTPLX-4bit")
}
