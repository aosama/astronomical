use std::fs;

use astronomical_runtime_integration::{MlxDtype, MlxPagedBufferSlot, MlxPagedFileReader};

#[test]
fn should_read_direct_ranges_through_product_neutral_mlx_c_paged_buffer_handles() {
    let fixture_directory = tempfile::tempdir().expect("the MLX-C paged fixture should exist");
    let source_file_path = fixture_directory.path().join("source.bin");
    let source_values = [11_u32, 22, 33, 44];
    fs::write(
        &source_file_path,
        source_values
            .iter()
            .flat_map(|source_value| source_value.to_ne_bytes())
            .collect::<Vec<_>>(),
    )
    .expect("the MLX-C paged source fixture should be writable");
    let source_reader = MlxPagedFileReader::new(&source_file_path)
        .expect("the MLX-C paged reader should open the source");
    let paged_slot = MlxPagedBufferSlot::new(source_values.len() * size_of::<u32>())
        .expect("the MLX-C paged slot should allocate final storage");
    paged_slot
        .read_range(
            &source_reader,
            2 * size_of::<u32>() as u64,
            0,
            2 * size_of::<u32>(),
        )
        .expect("the trailing source range should read into the slot prefix");
    paged_slot
        .read_range(
            &source_reader,
            0,
            2 * size_of::<u32>(),
            2 * size_of::<u32>(),
        )
        .expect("the leading source range should read into the slot suffix");
    paged_slot
        .commit()
        .expect("the complete MLX-C paged slot should commit");
    let reordered_values = paged_slot
        .view(
            &[source_values.len() as i32],
            MlxDtype::UInt32,
            0,
            source_values.len() * size_of::<u32>(),
        )
        .expect("the committed MLX-C paged slot should expose a typed view");
    assert_eq!(
        reordered_values
            .copy_evaluated_u32_values()
            .expect("the MLX-C paged view should copy for assertion"),
        vec![33, 44, 11, 22]
    );
}
