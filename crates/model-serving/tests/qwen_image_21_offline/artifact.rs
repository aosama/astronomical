//! Hermetic accept/reject tests for the Qwen-Image-2.1 artifact validator.
//!
//! Every test writes a synthetic wire-shaped package into a temporary directory (sparse
//! safetensors — headers plus zero-filled payload extents, no 10.5 GB materialized) and checks
//! the validator's decision and its bounded, path-free rejection reason.

use std::path::Path;

use astronomical_model_serving::{
    QwenImage21ArtifactError, QwenImage21ArtifactProvenance, QwenImage21ArtifactValidator,
};

use super::artifact_fixture::SyntheticQwenImage21Artifact;

fn temporary_artifact_root(label: &str) -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix(&format!("qwen_image_21_artifact_{label}"))
        .tempdir()
        .expect("the temporary artifact directory should be created")
}

fn official_provenance() -> QwenImage21ArtifactProvenance {
    QwenImage21ArtifactProvenance::new(
        "mlx-community/Qwen-Image-2.1-MLX-4bit",
        "0123456789abcdef0123456789abcdef01234567",
        "qwen-research",
    )
}

fn write_reviewed(root: &Path) {
    SyntheticQwenImage21Artifact::reviewed().write(root);
}

#[test]
fn should_accept_reviewed_artifact_and_retain_component_files() {
    let root = temporary_artifact_root("accept");
    write_reviewed(root.path());

    let artifact = QwenImage21ArtifactValidator::new()
        .validate(root.path(), official_provenance())
        .expect("the reviewed synthetic package should validate");

    assert_eq!(artifact.transformer_inventory().tensor_count(), 761);
    assert_eq!(artifact.text_encoder_inventory().tensor_count(), 1438);
    assert_eq!(artifact.vae_inventory().tensor_count(), 238);
    // Payload accounting must equal the index `total_size` the validator already checked.
    assert_eq!(
        artifact.transformer_inventory().payload_bytes(),
        artifact
            .transformer_inventory()
            .descriptors()
            .iter()
            .map(|descriptor| descriptor.payload_bytes())
            .sum::<u64>()
    );
    let retained = artifact
        .into_retained_files()
        .expect("retained file transfer should succeed");
    assert!(retained.transformer().size_bytes() > 0);
    assert!(retained.text_encoder().size_bytes() > 0);
    assert!(retained.vae().size_bytes() > 0);
    assert_eq!(retained.processor_sidecars().len(), 9);
    assert_eq!(retained.document_files().len(), 8);
}

#[test]
fn should_reject_unavailable_model_directory() {
    let root = temporary_artifact_root("missing-dir");
    let error = QwenImage21ArtifactValidator::new()
        .validate(root.path().join("does-not-exist"), official_provenance())
        .expect_err("a missing directory should be rejected");
    assert!(matches!(
        error,
        QwenImage21ArtifactError::ModelDirectoryUnavailable
    ));
}

#[test]
fn should_reject_wrong_license_provenance() {
    let root = temporary_artifact_root("license");
    write_reviewed(root.path());
    let error = QwenImage21ArtifactValidator::new()
        .validate(
            root.path(),
            QwenImage21ArtifactProvenance::new(
                "mlx-community/Qwen-Image-2.1-MLX-4bit",
                "0123456789abcdef0123456789abcdef01234567",
                "apache-2.0",
            ),
        )
        .expect_err("a non-research license should be rejected");
    assert!(matches!(
        error,
        QwenImage21ArtifactError::UnsupportedProvenance { .. }
    ));
}

#[test]
fn should_reject_unrecognized_model_id() {
    let root = temporary_artifact_root("model-id");
    write_reviewed(root.path());
    let error = QwenImage21ArtifactValidator::new()
        .validate(
            root.path(),
            QwenImage21ArtifactProvenance::new(
                "someone-else/Qwen-Image-2.1-MLX-8bit",
                "0123456789abcdef0123456789abcdef01234567",
                "qwen-research",
            ),
        )
        .expect_err("an unrecognized model id should be rejected");
    assert!(matches!(
        error,
        QwenImage21ArtifactError::UnsupportedProvenance { .. }
    ));
}

#[test]
fn should_reject_non_immutable_revision() {
    let root = temporary_artifact_root("revision");
    write_reviewed(root.path());
    let error = QwenImage21ArtifactValidator::new()
        .validate(
            root.path(),
            QwenImage21ArtifactProvenance::new(
                "mlx-community/Qwen-Image-2.1-MLX-4bit",
                "main",
                "qwen-research",
            ),
        )
        .expect_err("a branch name is not an immutable revision");
    assert!(matches!(
        error,
        QwenImage21ArtifactError::UnsupportedProvenance { .. }
    ));
}

#[test]
fn should_reject_missing_required_file() {
    let root = temporary_artifact_root("missing-file");
    let mut artifact = SyntheticQwenImage21Artifact::reviewed();
    artifact.delete_vae_weights();
    artifact.write(root.path());
    let error = QwenImage21ArtifactValidator::new()
        .validate(root.path(), official_provenance())
        .expect_err("a missing weight file should be rejected");
    assert!(matches!(
        error,
        QwenImage21ArtifactError::ArtifactFile { .. }
    ));
}

#[test]
fn should_reject_missing_processor_sidecar() {
    let root = temporary_artifact_root("missing-sidecar");
    write_reviewed(root.path());
    fs_remove(root.path().join("processor/tokenizer.json"));
    let error = QwenImage21ArtifactValidator::new()
        .validate(root.path(), official_provenance())
        .expect_err("a missing processor sidecar should be rejected");
    assert!(matches!(
        error,
        QwenImage21ArtifactError::ArtifactFile { .. }
    ));
}

fn fs_remove(path: std::path::PathBuf) {
    std::fs::remove_file(path).expect("the fixture file should be removable");
}

#[test]
fn should_reject_wrong_pipeline_class_name() {
    let root = temporary_artifact_root("pipeline-class");
    let mut artifact = SyntheticQwenImage21Artifact::reviewed();
    artifact.set_pipeline_class("Flux2KleinPipeline");
    artifact.write(root.path());
    let error = QwenImage21ArtifactValidator::new()
        .validate(root.path(), official_provenance())
        .expect_err("a wrong pipeline class should be rejected");
    assert!(matches!(error, QwenImage21ArtifactError::Configuration(_)));
}

#[test]
fn should_reject_unreviewed_transformer_quantization() {
    let root = temporary_artifact_root("quantization");
    let mut artifact = SyntheticQwenImage21Artifact::reviewed();
    artifact.use_bits_eight_transformer_quantization();
    artifact.write(root.path());
    let error = QwenImage21ArtifactValidator::new()
        .validate(root.path(), official_provenance())
        .expect_err("8-bit quantization is outside the reviewed profile");
    assert!(matches!(error, QwenImage21ArtifactError::Configuration(_)));
}

#[test]
fn should_reject_missing_transformer_tensor() {
    let root = temporary_artifact_root("missing-tensor");
    let mut artifact = SyntheticQwenImage21Artifact::reviewed();
    artifact.drop_transformer_tensor();
    artifact.write(root.path());
    let error = QwenImage21ArtifactValidator::new()
        .validate(root.path(), official_provenance())
        .expect_err("a missing tensor should be rejected");
    assert!(matches!(
        error,
        QwenImage21ArtifactError::MissingTensor {
            tensor_name, ..
        } if tensor_name == "proj_out.weight"
    ));
}

#[test]
fn should_reject_extra_transformer_tensor() {
    let root = temporary_artifact_root("extra-tensor");
    let mut artifact = SyntheticQwenImage21Artifact::reviewed();
    artifact.add_extra_transformer_tensor();
    artifact.write(root.path());
    let error = QwenImage21ArtifactValidator::new()
        .validate(root.path(), official_provenance())
        .expect_err("an unexpected tensor should be rejected");
    assert!(matches!(
        error,
        QwenImage21ArtifactError::UnsupportedTensor {
            tensor_name, ..
        } if tensor_name == "transformer_blocks.0.attn.extra_projection.weight"
    ));
}

#[test]
fn should_reject_unreviewed_vae_dtype() {
    let root = temporary_artifact_root("dtype");
    let mut artifact = SyntheticQwenImage21Artifact::reviewed();
    artifact.flip_first_vae_dtype();
    artifact.write(root.path());
    let error = QwenImage21ArtifactValidator::new()
        .validate(root.path(), official_provenance())
        .expect_err("an unreviewed dtype should be rejected");
    assert!(matches!(
        error,
        QwenImage21ArtifactError::TensorDtype {
            component: "vae",
            ..
        }
    ));
}

#[test]
fn should_reject_widened_transformer_tensor() {
    let root = temporary_artifact_root("shape");
    let mut artifact = SyntheticQwenImage21Artifact::reviewed();
    artifact.widen_first_transformer_tensor();
    artifact.write(root.path());
    let error = QwenImage21ArtifactValidator::new()
        .validate(root.path(), official_provenance())
        .expect_err("a re-shaped tensor should be rejected");
    assert!(matches!(
        error,
        QwenImage21ArtifactError::TensorShape {
            component: "transformer",
            ..
        }
    ));
}

#[test]
fn should_reject_shard_index_total_size_mismatch() {
    let root = temporary_artifact_root("index-size");
    let mut artifact = SyntheticQwenImage21Artifact::reviewed();
    artifact.set_index_total_size_delta(1024);
    artifact.write(root.path());
    let error = QwenImage21ArtifactValidator::new()
        .validate(root.path(), official_provenance())
        .expect_err("a size disagreement should be rejected");
    assert!(matches!(
        error,
        QwenImage21ArtifactError::ShardIndexTotalSizeMismatch { .. }
    ));
}

#[test]
fn should_reject_unexpected_shard_name() {
    let root = temporary_artifact_root("index-shard");
    let mut artifact = SyntheticQwenImage21Artifact::reviewed();
    artifact.set_index_shard_override("img_in.weight");
    artifact.write(root.path());
    let error = QwenImage21ArtifactValidator::new()
        .validate(root.path(), official_provenance())
        .expect_err("a multi-shard index should be rejected for this single-shard family");
    assert!(matches!(
        error,
        QwenImage21ArtifactError::UnexpectedShardName {
            tensor_name, ..
        } if tensor_name == "img_in.weight"
    ));
}
