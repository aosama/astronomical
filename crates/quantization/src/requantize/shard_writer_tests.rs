//! Unit tests for shard naming, flushing cadence, and the index, with
//! byte-vector tensors and a recording flush.

use super::*;

/// Records every flush and writes an empty file so finalize can rename it.
#[derive(Default)]
struct RecordingFlush {
    flushed: Vec<(PathBuf, Vec<String>)>,
}

impl ShardFlush<Vec<u8>> for RecordingFlush {
    fn flush_shard(
        &mut self,
        shard_path: &Path,
        tensors: Vec<(String, Vec<u8>)>,
    ) -> Result<u64, QuantizationError> {
        let bytes: u64 = tensors.iter().map(|(_, tensor)| tensor.len() as u64).sum();
        std::fs::write(shard_path, b"").expect("the fixture shard should be writable");
        self.flushed.push((
            shard_path.to_path_buf(),
            tensors.into_iter().map(|(key, _)| key).collect(),
        ));
        Ok(bytes)
    }
}

#[test]
#[ignore = "pending implementation of ShardWriter::final_shard_name"]
fn should_name_a_single_shard_model_safetensors() {
    assert_eq!(
        ShardWriter::<Vec<u8>>::final_shard_name(0, 1),
        "model.safetensors"
    );
}

#[test]
#[ignore = "pending implementation of ShardWriter::final_shard_name"]
fn should_name_multiple_shards_one_based_with_five_digits() {
    assert_eq!(
        ShardWriter::<Vec<u8>>::final_shard_name(0, 3),
        "model-00001-of-00003.safetensors"
    );
    assert_eq!(
        ShardWriter::<Vec<u8>>::final_shard_name(2, 3),
        "model-00003-of-00003.safetensors"
    );
}

#[test]
#[ignore = "pending implementation of ShardWriter::add and flush"]
fn should_flush_when_the_buffer_reaches_the_limit() {
    let output_directory = tempfile::tempdir().expect("a temporary directory should be creatable");
    let mut writer = ShardWriter::<Vec<u8>>::new(output_directory.path(), 8);
    let mut flush = RecordingFlush::default();

    writer
        .add("a".to_owned(), vec![0; 4], 4, &mut flush)
        .expect("buffering should succeed");
    assert!(flush.flushed.is_empty());
    writer
        .add("b".to_owned(), vec![0; 4], 4, &mut flush)
        .expect("buffering should succeed");

    assert_eq!(flush.flushed.len(), 1);
    assert_eq!(flush.flushed[0].1, vec!["a".to_owned(), "b".to_owned()]);
}

#[test]
#[ignore = "pending implementation of ShardWriter::finalize"]
fn should_write_an_index_mapping_every_key_to_its_final_shard() {
    let output_directory = tempfile::tempdir().expect("a temporary directory should be creatable");
    let mut writer = ShardWriter::<Vec<u8>>::new(output_directory.path(), 4);
    let mut flush = RecordingFlush::default();
    writer
        .add("a".to_owned(), vec![0; 4], 4, &mut flush)
        .expect("buffering should succeed");
    writer
        .add("b".to_owned(), vec![0; 4], 4, &mut flush)
        .expect("buffering should succeed");

    let shard_count = writer
        .finalize(&mut flush)
        .expect("finalize should succeed");

    assert_eq!(shard_count, 2);
    let index_text =
        std::fs::read_to_string(output_directory.path().join("model.safetensors.index.json"))
            .expect("the index should exist");
    let index: serde_json::Value =
        serde_json::from_str(&index_text).expect("the index should be JSON");
    assert_eq!(index["weight_map"]["a"], "model-00001-of-00002.safetensors");
    assert_eq!(index["weight_map"]["b"], "model-00002-of-00002.safetensors");
    assert_eq!(index["metadata"]["total_size"], 8);
    assert!(
        output_directory
            .path()
            .join("model-00001-of-00002.safetensors")
            .exists()
    );
}
