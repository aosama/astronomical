use std::fs;

use astronomical_model_serving::{
    FLUX2_KLEIN_OFFICIAL_MODEL_ID, FLUX2_KLEIN_OFFICIAL_REVISION, FLUX2_KLEIN_PROVIDER_MODEL_ID,
    Flux2KleinArtifactError, Flux2KleinArtifactProvenance, Flux2KleinArtifactValidator,
};

use super::support::SyntheticFlux2KleinArtifact;

#[test]
fn should_validate_the_nested_official_artifact_and_retain_runtime_descriptors() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    SyntheticFlux2KleinArtifact::official().write(model_directory.path());

    let artifact = Flux2KleinArtifactValidator::new()
        .validate(
            model_directory.path(),
            Flux2KleinArtifactProvenance::official(),
        )
        .expect("the complete synthetic official artifact should validate");

    assert_eq!(artifact.revision(), FLUX2_KLEIN_OFFICIAL_REVISION);
    assert_eq!(artifact.license().identifier(), "Apache-2.0");
    assert_eq!(artifact.text_shard_count(), 2);
    assert_eq!(
        artifact.transformer_inventory().double_stream_block_count(),
        5
    );
    assert_eq!(
        artifact.transformer_inventory().single_stream_block_count(),
        20
    );
    assert_eq!(artifact.transformer_inventory().descriptors().len(), 169);
    assert_eq!(artifact.vae_inventory().up_block_count(), 4);
    assert_eq!(artifact.vae_inventory().descriptors().len(), 251);
    let decoder_owned_payload_bytes = artifact
        .vae_inventory()
        .vae_decoder_owned_payload_bytes()
        .expect("validated descriptor payloads should not overflow");
    let complete_vae_payload_bytes = artifact.vae_inventory().payload_bytes();
    assert!(decoder_owned_payload_bytes > 0);
    assert!(decoder_owned_payload_bytes < complete_vae_payload_bytes);
    let decoder_descriptors = artifact
        .vae_inventory()
        .descriptors()
        .iter()
        .filter(|descriptor| descriptor.is_owned_by_vae_decoder())
        .collect::<Vec<_>>();
    assert!(
        decoder_descriptors
            .iter()
            .any(|descriptor| descriptor.tensor_name().starts_with("decoder."))
    );
    assert!(
        decoder_descriptors
            .iter()
            .any(|descriptor| descriptor.tensor_name().starts_with("post_quant_conv."))
    );
    assert!(
        decoder_descriptors
            .iter()
            .any(|descriptor| descriptor.tensor_name() == "bn.running_mean")
    );
    assert!(
        artifact
            .vae_inventory()
            .descriptors()
            .iter()
            .filter(|descriptor| {
                descriptor.tensor_name().starts_with("encoder.")
                    || descriptor.tensor_name().starts_with("quant_conv.")
                    || descriptor.tensor_name() == "bn.num_batches_tracked"
            })
            .all(|descriptor| !descriptor.is_owned_by_vae_decoder())
    );

    let retained_files = artifact
        .into_retained_files()
        .expect("descriptor identities should transfer");
    fs::remove_dir_all(model_directory.path()).expect("the model paths should be removable");
    assert_eq!(retained_files.text_shards().len(), 2);
    assert!(retained_files.transformer().size_bytes() > 0);
    assert!(retained_files.vae().size_bytes() > 0);
    assert_eq!(retained_files.tokenizer_sidecars().len(), 7);
}

#[test]
fn should_validate_repackaged_text_shards_from_the_index_instead_of_fixed_names() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    SyntheticFlux2KleinArtifact::official().write(model_directory.path());
    let repacked_directory = model_directory.path().join("text_encoder/repacked");
    fs::create_dir(&repacked_directory).expect("the nested shard directory should exist");
    for (original_name, repacked_name) in [
        ("model-00001-of-00002.safetensors", "encoder-a.safetensors"),
        ("model-00002-of-00002.safetensors", "encoder-b.safetensors"),
    ] {
        fs::rename(
            model_directory
                .path()
                .join("text_encoder")
                .join(original_name),
            repacked_directory.join(repacked_name),
        )
        .expect("the indexed text shard should be repackaged");
    }
    let index_path = model_directory
        .path()
        .join("text_encoder/model.safetensors.index.json");
    let index_text = fs::read_to_string(&index_path)
        .expect("the synthetic text index should remain readable")
        .replace(
            "model-00001-of-00002.safetensors",
            "repacked/encoder-a.safetensors",
        )
        .replace(
            "model-00002-of-00002.safetensors",
            "repacked/encoder-b.safetensors",
        );
    fs::write(index_path, index_text).expect("the repackaged text index should be written");

    let artifact = Flux2KleinArtifactValidator::new()
        .validate(
            model_directory.path(),
            Flux2KleinArtifactProvenance::official(),
        )
        .expect("safe index-owned text shard names should validate");

    assert_eq!(artifact.text_shard_count(), 2);
}

#[test]
fn should_ignore_the_duplicate_root_transformer_and_reject_index_disagreement() {
    let valid_directory = tempfile::tempdir().expect("the test should create a model directory");
    let mut valid_fixture = SyntheticFlux2KleinArtifact::official();
    valid_fixture.root_transformer_bytes = b"not safetensors and intentionally ignored".to_vec();
    valid_fixture.write(valid_directory.path());
    Flux2KleinArtifactValidator::new()
        .validate(
            valid_directory.path(),
            Flux2KleinArtifactProvenance::official(),
        )
        .expect("the duplicate root transformer must not own runtime tensors");

    let invalid_directory = tempfile::tempdir().expect("the test should create a model directory");
    let mut invalid_fixture = SyntheticFlux2KleinArtifact::official();
    invalid_fixture.move_indexed_text_tensor_to_wrong_shard();
    invalid_fixture.write(invalid_directory.path());
    assert!(matches!(
        Flux2KleinArtifactValidator::new().validate(
            invalid_directory.path(),
            Flux2KleinArtifactProvenance::official(),
        ),
        Err(Flux2KleinArtifactError::TextShardIndexDisagreement { .. })
    ));
}

#[test]
fn should_reject_text_index_parameter_count_that_disagrees_with_physical_geometry() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    SyntheticFlux2KleinArtifact::official().write(model_directory.path());
    let index_path = model_directory
        .path()
        .join("text_encoder/model.safetensors.index.json");
    let mut index_document: serde_json::Value = serde_json::from_slice(
        &fs::read(&index_path).expect("the synthetic text index should be readable"),
    )
    .expect("the synthetic text index should contain JSON");
    index_document["metadata"]["total_parameters"] = serde_json::json!(1);
    fs::write(
        index_path,
        serde_json::to_vec(&index_document).expect("the changed index should serialize"),
    )
    .expect("the changed index should be writable");

    assert!(matches!(
        Flux2KleinArtifactValidator::new().validate(
            model_directory.path(),
            Flux2KleinArtifactProvenance::official(),
        ),
        Err(Flux2KleinArtifactError::TextShardIndexTotalParameterMismatch { .. })
    ));
}

#[test]
fn should_reject_non_official_provenance_and_never_render_local_paths() {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    SyntheticFlux2KleinArtifact::official().write(model_directory.path());
    let unsupported_provenance = [
        Flux2KleinArtifactProvenance::new(
            FLUX2_KLEIN_OFFICIAL_MODEL_ID,
            FLUX2_KLEIN_OFFICIAL_REVISION,
            "Apache-2.0",
        ),
        Flux2KleinArtifactProvenance::new(
            "another-owner/FLUX.2-klein-4B",
            FLUX2_KLEIN_OFFICIAL_REVISION,
            "Apache-2.0",
        ),
        Flux2KleinArtifactProvenance::new(
            FLUX2_KLEIN_PROVIDER_MODEL_ID,
            "0000000000000000000000000000000000000000",
            "Apache-2.0",
        ),
        Flux2KleinArtifactProvenance::new(
            FLUX2_KLEIN_PROVIDER_MODEL_ID,
            FLUX2_KLEIN_OFFICIAL_REVISION,
            "unknown",
        ),
    ];

    for provenance in unsupported_provenance {
        let error = Flux2KleinArtifactValidator::new()
            .validate(model_directory.path(), provenance)
            .expect_err("every changed provenance field must fail closed");
        assert!(matches!(
            error,
            Flux2KleinArtifactError::UnsupportedProvenance { .. }
        ));
        assert!(
            !error
                .to_string()
                .contains(&model_directory.path().display().to_string())
        );
    }
}

#[test]
fn should_reject_wrong_model_derived_shape_and_non_bf16_storage_during_deep_validation() {
    let wrong_shape_directory =
        tempfile::tempdir().expect("the shape test should create a model directory");
    let mut wrong_shape_fixture = SyntheticFlux2KleinArtifact::official();
    wrong_shape_fixture.invalidate_transformer_shape();
    wrong_shape_fixture.write(wrong_shape_directory.path());
    assert!(matches!(
        Flux2KleinArtifactValidator::new().validate(
            wrong_shape_directory.path(),
            Flux2KleinArtifactProvenance::official(),
        ),
        Err(Flux2KleinArtifactError::TensorShape {
            component: "transformer",
            ..
        })
    ));

    let wrong_dtype_directory =
        tempfile::tempdir().expect("the dtype test should create a model directory");
    let mut wrong_dtype_fixture = SyntheticFlux2KleinArtifact::official();
    wrong_dtype_fixture.invalidate_transformer_dtype();
    wrong_dtype_fixture.write(wrong_dtype_directory.path());
    assert!(matches!(
        Flux2KleinArtifactValidator::new().validate(
            wrong_dtype_directory.path(),
            Flux2KleinArtifactProvenance::official(),
        ),
        Err(Flux2KleinArtifactError::TensorDtype {
            component: "transformer",
            ..
        })
    ));
}
