use std::fs;
use std::path::Path;

use astronomical_config::{
    Flux2KleinDirectoryVerificationError, ModelCapabilities, ModelFamily, ModelLicense,
    classify_model_directory, verify_flux2_klein_model_directory,
};
use serde_json::{Map, Value, json};

const CANONICAL_MODEL_ID: &str = "flux2-klein-4b-4bit";
const PROVIDER_MODEL_ID: &str = "mlx-community/flux2-klein-4b-4bit";
const REVIEWED_REVISION: &str = "860e87183ceb29e39627c0612ebd66d8ea66e68c";
const SHARD_PAYLOAD_BYTES: usize = 64;
const MODEL_CARD: &str = "---\nlicense: apache-2.0\ntags:\n- mlx\nbase_model: black-forest-labs/FLUX.2-klein-4B\npipeline_tag: text-to-image\n---\n\n# FLUX.2 Klein 4B, 4-bit\n";

#[test]
fn should_discover_the_4bit_build_under_its_own_model_id() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let model_directory = temporary_directory.path().join("flux2-klein-4b-4bit");
    write_mlx_4bit_artifact(&model_directory);

    let directory_scans = super::discover_configured_models(&temporary_directory);
    let discovered_model = directory_scans[0]
        .discovered_models
        .first()
        .expect("the 4-bit build should be advertised");

    assert_eq!(discovered_model.model_id, CANONICAL_MODEL_ID);
    assert_eq!(
        discovered_model.provider_model_id.as_deref(),
        Some(PROVIDER_MODEL_ID)
    );
    assert_eq!(discovered_model.model_family, ModelFamily::Flux2Klein);
    assert_eq!(discovered_model.revision, REVIEWED_REVISION);
    assert_eq!(discovered_model.license, Some(ModelLicense::Apache20));
    // Two transformer shards, two text shards and one VAE shard.
    assert_eq!(
        discovered_model.model_size_bytes,
        5 * SHARD_PAYLOAD_BYTES as u64
    );
    let ModelCapabilities::ImageGeneration(image_capabilities) = &discovered_model.capabilities
    else {
        panic!("the 4-bit build must expose image-generation capabilities");
    };
    assert_eq!(image_capabilities.default_steps, 4);
    assert_eq!(image_capabilities.maximum_dimension_pixels, 1_024);
}

#[test]
fn should_classify_the_4bit_tree_as_flux2_klein_without_a_pipeline_index() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    write_mlx_4bit_artifact(temporary_directory.path());

    assert_eq!(
        classify_model_directory(temporary_directory.path())
            .expect("the 4-bit tree should classify"),
        Some(ModelFamily::Flux2Klein)
    );
}

#[test]
fn should_take_the_revision_from_the_hugging_face_snapshot_folder() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let snapshot_directory = temporary_directory
        .path()
        .join("models--mlx-community--flux2-klein-4b-4bit/snapshots")
        .join(REVIEWED_REVISION);
    write_mlx_4bit_artifact(&snapshot_directory);
    fs::remove_file(snapshot_directory.join(".astronomical-library-provenance.json"))
        .expect("the Library provenance should be removable");

    let evidence = verify_flux2_klein_model_directory(&snapshot_directory)
        .expect("a Hugging Face snapshot should verify");

    assert_eq!(evidence.revision, REVIEWED_REVISION);
    assert_eq!(evidence.provider_model_id, PROVIDER_MODEL_ID);
}

#[test]
fn should_reject_a_4bit_tree_that_does_not_prove_the_reviewed_build() {
    for (invalid_artifact, expected_error) in [
        (
            InvalidArtifact::Klein9BBlocks,
            Flux2KleinDirectoryVerificationError::InvalidQuantizedWeightIndex {
                component: "transformer",
            },
        ),
        (
            InvalidArtifact::TruncatedTextEncoder,
            Flux2KleinDirectoryVerificationError::InvalidQuantizedWeightIndex {
                component: "text encoder",
            },
        ),
        (
            InvalidArtifact::OtherLicense,
            Flux2KleinDirectoryVerificationError::InvalidLicense,
        ),
        (
            InvalidArtifact::MissingModelCard,
            Flux2KleinDirectoryVerificationError::InvalidLicense,
        ),
        (
            InvalidArtifact::MissingTokenizer,
            Flux2KleinDirectoryVerificationError::MissingOrInvalidSidecar {
                sidecar: "tokenizer/tokenizer.json",
            },
        ),
        (
            InvalidArtifact::MissingShard,
            Flux2KleinDirectoryVerificationError::MissingOrInvalidWeightFile { component: "vae" },
        ),
        (
            InvalidArtifact::NoRevision,
            Flux2KleinDirectoryVerificationError::MissingRevision,
        ),
    ] {
        let temporary_directory =
            tempfile::tempdir().expect("temporary directory should be created");
        write_mlx_4bit_artifact(temporary_directory.path());
        invalidate_artifact(temporary_directory.path(), invalid_artifact);

        assert_eq!(
            verify_flux2_klein_model_directory(temporary_directory.path()),
            Err(expected_error),
            "{invalid_artifact:?}"
        );
    }
}

#[derive(Clone, Copy, Debug)]
enum InvalidArtifact {
    Klein9BBlocks,
    TruncatedTextEncoder,
    OtherLicense,
    MissingModelCard,
    MissingTokenizer,
    MissingShard,
    NoRevision,
}

fn invalidate_artifact(model_directory: &Path, invalid_artifact: InvalidArtifact) {
    match invalid_artifact {
        InvalidArtifact::Klein9BBlocks => write_json(
            &model_directory.join("transformer/model.safetensors.index.json"),
            transformer_index(8, 24),
        ),
        InvalidArtifact::TruncatedTextEncoder => write_json(
            &model_directory.join("text_encoder/model.safetensors.index.json"),
            text_encoder_index(35),
        ),
        InvalidArtifact::OtherLicense => fs::write(
            model_directory.join("README.md"),
            MODEL_CARD.replace("apache-2.0", "mit"),
        )
        .expect("the model card should be rewritten"),
        InvalidArtifact::MissingModelCard => remove(model_directory, "README.md"),
        InvalidArtifact::MissingTokenizer => remove(model_directory, "tokenizer/tokenizer.json"),
        InvalidArtifact::MissingShard => remove(model_directory, "vae/0.safetensors"),
        InvalidArtifact::NoRevision => {
            remove(model_directory, ".astronomical-library-provenance.json");
        }
    }
}

fn write_mlx_4bit_artifact(model_directory: &Path) {
    write_json(
        &model_directory.join("transformer/model.safetensors.index.json"),
        transformer_index(5, 20),
    );
    write_json(
        &model_directory.join("text_encoder/model.safetensors.index.json"),
        text_encoder_index(36),
    );
    write_json(
        &model_directory.join("vae/model.safetensors.index.json"),
        mflux_index([("decoder.conv_in.weight".to_owned(), "0.safetensors")]),
    );
    for shard_path in [
        "transformer/0.safetensors",
        "transformer/1.safetensors",
        "text_encoder/0.safetensors",
        "text_encoder/1.safetensors",
        "vae/0.safetensors",
    ] {
        fs::write(
            model_directory.join(shard_path),
            vec![0_u8; SHARD_PAYLOAD_BYTES],
        )
        .expect("the shard should be written");
    }
    fs::create_dir_all(model_directory.join("tokenizer"))
        .expect("the tokenizer folder should exist");
    for tokenizer_file in [
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
    ] {
        fs::write(model_directory.join("tokenizer").join(tokenizer_file), "{}")
            .expect("the tokenizer file should be written");
    }
    fs::write(model_directory.join("README.md"), MODEL_CARD)
        .expect("the model card should be written");
    fs::write(
        model_directory.join(".astronomical-library-provenance.json"),
        format!(
            "{{\"schema_version\":1,\"provider_model_id\":\"{PROVIDER_MODEL_ID}\",\"revision\":\"{REVIEWED_REVISION}\"}}"
        ),
    )
    .expect("the Library provenance should be written");
}

fn transformer_index(double_stream_block_count: usize, single_stream_block_count: usize) -> Value {
    let mut entries = vec![(
        "time_guidance_embed.linear_1.weight".to_owned(),
        "0.safetensors",
    )];
    for block_index in 0..double_stream_block_count {
        entries.push((
            format!("transformer_blocks.{block_index}.attn.to_q.weight"),
            "0.safetensors",
        ));
    }
    for block_index in 0..single_stream_block_count {
        entries.push((
            format!("single_transformer_blocks.{block_index}.attn.to_out.weight"),
            "1.safetensors",
        ));
    }
    mflux_index(entries)
}

fn text_encoder_index(layer_count: usize) -> Value {
    let mut entries = vec![("embed_tokens.weight".to_owned(), "0.safetensors")];
    for layer_index in 0..layer_count {
        entries.push((
            format!("layers.{layer_index}.mlp.down_proj.weight"),
            "1.safetensors",
        ));
    }
    mflux_index(entries)
}

fn mflux_index(entries: impl IntoIterator<Item = (String, &'static str)>) -> Value {
    let weight_map = entries
        .into_iter()
        .map(|(tensor_name, shard_file_name)| (tensor_name, json!(shard_file_name)))
        .collect::<Map<_, _>>();
    json!({
        "metadata": {
            "quantization_level": "4",
            "quantization_group_size": "64",
            "mflux_version": "0.17.5"
        },
        "weight_map": weight_map
    })
}

fn remove(model_directory: &Path, relative_path: &str) {
    fs::remove_file(model_directory.join(relative_path))
        .expect("the fixture file should be removable");
}

fn write_json(file_path: &Path, document: Value) {
    fs::create_dir_all(file_path.parent().expect("the file should have a parent"))
        .expect("the parent folder should exist");
    fs::write(file_path, document.to_string()).expect("the JSON document should be written");
}
