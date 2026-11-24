use super::*;

#[test]
fn should_name_towers_for_mlx_from_each_layout() {
    assert_eq!(
        mlx_tower_tensor_name("model.visual.blocks.0.attn.proj.weight").as_deref(),
        Some("vision_tower.blocks.0.attn.proj.weight")
    );
    assert_eq!(
        mlx_tower_tensor_name("visual.merger.norm.bias").as_deref(),
        Some("vision_tower.merger.norm.bias")
    );
    assert_eq!(
        mlx_tower_tensor_name("vision_tower.pos_embed.weight").as_deref(),
        Some("vision_tower.pos_embed.weight")
    );
}

#[test]
fn should_refuse_a_layout_it_cannot_convert() {
    assert_eq!(mlx_tower_tensor_name("vision_model.encoder.weight"), None);
    assert_eq!(mlx_tower_tensor_name("visual"), None);
    assert_eq!(mlx_tower_tensor_name(""), None);
}

#[test]
fn should_reorder_only_an_upstream_five_dimensional_patch_embedding() {
    let upstream = "model.visual.patch_embed.proj.weight";
    assert!(is_channels_first_patch_embedding(
        upstream,
        &[4, 3, 2, 2, 2]
    ));
    assert!(!is_channels_first_patch_embedding(upstream, &[4, 3]));
    assert!(!is_channels_first_patch_embedding(
        "vision_tower.patch_embed.proj.weight",
        &[4, 2, 2, 2, 3]
    ));
    assert!(!is_channels_first_patch_embedding(
        "model.visual.patch_embed.proj.bias",
        &[4, 3, 2, 2, 2]
    ));
}

#[test]
fn should_match_the_real_towers_shape_change() {
    assert_eq!(
        channels_last_shape(&[1152, 3, 2, 16, 16]),
        [1152, 2, 16, 16, 3]
    );
}

/// Each element holds its own channels-first coordinates, so a wrong reorder
/// cannot pass by coincidence.
#[test]
fn should_move_every_element_to_its_channels_last_position() {
    let shape = [2_u64, 3, 2, 2, 2];
    let mut source_bytes = Vec::new();
    for output in 0..2_u8 {
        for input in 0..3_u8 {
            for depth in 0..2_u8 {
                for height in 0..2_u8 {
                    for width in 0..2_u8 {
                        source_bytes.extend_from_slice(&[
                            output * 16 + input,
                            depth * 4 + height * 2 + width,
                        ]);
                    }
                }
            }
        }
    }
    let reordered = channels_first_to_channels_last(&source_bytes, &shape, 2);
    assert_eq!(reordered.len(), source_bytes.len());
    let mut element_position = 0;
    for output in 0..2_u8 {
        for depth in 0..2_u8 {
            for height in 0..2_u8 {
                for width in 0..2_u8 {
                    for input in 0..3_u8 {
                        assert_eq!(reordered[element_position], output * 16 + input);
                        assert_eq!(
                            reordered[element_position + 1],
                            depth * 4 + height * 2 + width
                        );
                        element_position += 2;
                    }
                }
            }
        }
    }
}
