use astronomical_model_serving::{
    Qwen3_5MoEImageGrid, Qwen3_5MoEVisionConfig, Qwen3_5MoEVisionInputPlan,
};

const CERTIFIED_VISION_CONFIG_JSON: &str = r#"
{
  "vision_config": {
    "depth": 27,
    "hidden_size": 1152,
    "in_channels": 3,
    "intermediate_size": 4304,
    "model_type": "qwen3_5_moe_vision",
    "num_heads": 16,
    "num_position_embeddings": 2304,
    "out_hidden_size": 2048,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "hidden_act": "gelu_pytorch_tanh",
    "deepstack_visual_indexes": []
  }
}
"#;

#[test]
fn should_plan_rectangular_vision_positions_in_spatial_merge_order() {
    let vision_config =
        Qwen3_5MoEVisionConfig::from_json_bytes(CERTIFIED_VISION_CONFIG_JSON.as_bytes())
            .expect("the certified Ornith vision config should parse");
    let image_grid = Qwen3_5MoEImageGrid {
        temporal_patch_count: 1,
        height_patch_count: 4,
        width_patch_count: 2,
    };

    let vision_input_plan = Qwen3_5MoEVisionInputPlan::new(&[image_grid], &vision_config)
        .expect("the rectangular image grid should produce a vision input plan");

    assert_eq!(vision_input_plan.patch_count(), 8);
    assert_eq!(vision_input_plan.merged_patch_count(), 2);
    assert_eq!(vision_input_plan.attention_sequence_boundaries(), &[0, 8]);
    assert_eq!(
        vision_input_plan.rotary_position_coordinates(),
        &[
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [2, 0],
            [2, 1],
            [3, 0],
            [3, 1],
        ]
    );

    let bilinear_corner_indices = vision_input_plan.bilinear_corner_indices();
    assert_eq!(
        bilinear_corner_indices[0],
        [0, 47, 720, 767, 1_488, 1_535, 2_256, 2_303]
    );
    assert_eq!(
        bilinear_corner_indices[2],
        [48, 95, 768, 815, 1_536, 1_583, 2_256, 2_303]
    );

    let bilinear_corner_weights = vision_input_plan.bilinear_corner_weights();
    assert!((bilinear_corner_weights[0][2] - (1.0 / 3.0)).abs() < 1e-6);
    assert!((bilinear_corner_weights[2][2] - (2.0 / 3.0)).abs() < 1e-6);
    assert!((bilinear_corner_weights[0][4] - (2.0 / 3.0)).abs() < 1e-6);
    assert!((bilinear_corner_weights[2][4] - (1.0 / 3.0)).abs() < 1e-6);
}
