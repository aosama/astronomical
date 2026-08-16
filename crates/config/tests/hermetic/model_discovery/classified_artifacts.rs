use std::fs;

use astronomical_config::{ModelFamily, discover_classified_model_artifacts};

use super::write_minimal_model_config;

#[test]
fn should_find_local_and_hugging_face_classified_artifacts_without_advertising_execution() {
    let configured_root = tempfile::tempdir().expect("the test should create a root");
    let local_model = configured_root
        .path()
        .join("example-org")
        .join("Laguna-Test");
    fs::create_dir_all(local_model.join(".cache/huggingface/download"))
        .expect("the local model directory should be created");
    write_minimal_model_config(&local_model, "laguna", 4_096);
    fs::write(
        local_model.join(".cache/huggingface/download/config.json.metadata"),
        "1111111111111111111111111111111111111111\nfixture-etag\n0\n",
    )
    .expect("the local revision metadata should be written");

    let cache_snapshot = configured_root
        .path()
        .join("models--example-org--DeepSeek-Test")
        .join("snapshots")
        .join("2222222222222222222222222222222222222222");
    fs::create_dir_all(&cache_snapshot).expect("the cache snapshot should be created");
    write_minimal_model_config(&cache_snapshot, "deepseek_v4", 4_096);

    let artifacts = discover_classified_model_artifacts(&[configured_root.path().to_path_buf()])
        .expect("classified artifact discovery should complete");

    assert_eq!(artifacts.len(), 2);
    assert!(artifacts.iter().any(|artifact| {
        artifact.model_id == "example-org/Laguna-Test"
            && artifact.upstream_revision.as_deref()
                == Some("1111111111111111111111111111111111111111")
            && artifact.model_family == ModelFamily::Laguna
    }));
    assert!(artifacts.iter().any(|artifact| {
        artifact.model_id == "example-org/DeepSeek-Test"
            && artifact.upstream_revision.as_deref()
                == Some("2222222222222222222222222222222222222222")
            && artifact.model_family == ModelFamily::DeepSeekV4
    }));
}
