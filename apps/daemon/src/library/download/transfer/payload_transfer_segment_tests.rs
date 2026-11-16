use bytes::Bytes;
use tempfile::TempDir;

use super::SegmentedFileAssembler;

async fn open_assembler(models_folder: &TempDir) -> SegmentedFileAssembler {
    let staging_folder = models_folder.path().join(".incomplete/owner/model");
    std::fs::create_dir_all(&staging_folder).expect("the staging folder is created");
    SegmentedFileAssembler::open(models_folder.path(), &staging_folder.join("weights.bin"), 0)
        .await
        .expect("the staged file opens")
}

#[tokio::test]
async fn should_count_a_later_piece_as_received_before_the_prefix_reaches_it() {
    let models_folder = TempDir::new().expect("a temporary models folder");
    let assembler = open_assembler(&models_folder).await;

    assembler
        .write_chunk(100, Bytes::from(vec![7_u8; 50]))
        .await
        .expect("the later piece is buffered");

    assert_eq!(assembler.contiguous_end_bytes().await, 0);
    assert_eq!(assembler.received_bytes().await, 50);
}

#[tokio::test]
async fn should_keep_the_received_count_unchanged_when_buffered_bytes_join_the_prefix() {
    let models_folder = TempDir::new().expect("a temporary models folder");
    let assembler = open_assembler(&models_folder).await;
    assembler
        .write_chunk(100, Bytes::from(vec![7_u8; 50]))
        .await
        .expect("the later piece is buffered");

    assembler
        .write_chunk(0, Bytes::from(vec![3_u8; 100]))
        .await
        .expect("the first piece fills the gap");

    assert_eq!(assembler.contiguous_end_bytes().await, 150);
    assert_eq!(assembler.received_bytes().await, 150);
}

#[tokio::test]
async fn should_report_nothing_received_before_any_byte_arrives() {
    let models_folder = TempDir::new().expect("a temporary models folder");
    let assembler = open_assembler(&models_folder).await;

    assert_eq!(assembler.received_bytes().await, 0);
}
