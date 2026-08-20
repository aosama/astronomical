use astronomical_model_serving::{Flux2KleinVaeTilePlan, Flux2KleinVaeTilingConfig};

#[test]
fn should_assign_every_output_pixel_to_exactly_one_overlapped_decoder_tile() {
    let config = Flux2KleinVaeTilingConfig::new(4, 2).expect("valid tiling configuration");
    let plan = Flux2KleinVaeTilePlan::new(9, 7, config).expect("valid tile plan");
    let mut ownership_counts = vec![0_u8; 9 * 7];

    for tile in plan.tiles() {
        assert!(tile.source_row_start() <= tile.owned_row_start());
        assert!(tile.source_column_start() <= tile.owned_column_start());
        assert!(tile.source_row_end() >= tile.owned_row_end());
        assert!(tile.source_column_end() >= tile.owned_column_end());
        for row in tile.owned_row_start()..tile.owned_row_end() {
            for column in tile.owned_column_start()..tile.owned_column_end() {
                ownership_counts[row * 9 + column] += 1;
            }
        }
    }

    assert!(ownership_counts.into_iter().all(|count| count == 1));
}

#[test]
fn should_reject_tiling_without_a_positive_owned_core() {
    assert!(Flux2KleinVaeTilingConfig::new(0, 2).is_err());
}
