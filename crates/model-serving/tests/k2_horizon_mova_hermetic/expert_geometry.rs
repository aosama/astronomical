use astronomical_model_serving::{
    ExpertLayerGeometry, K2HorizonMoVAConfig, K2HorizonMoVASparseLayerExpertPayload,
    k2_horizon_mova_expert_layer_geometries,
};

use super::support::family_member_config_json;

#[test]
fn k2_horizon_mova_plan_slots_are_feed_forward_then_mixture_of_values() {
    let config =
        K2HorizonMoVAConfig::from_json_bytes(family_member_config_json(4, &[0], 4, 2).as_bytes())
            .expect("tiny family config should parse");
    let geometries = k2_horizon_mova_expert_layer_geometries(
        &config,
        &[
            K2HorizonMoVASparseLayerExpertPayload {
                decoder_layer_index: 1,
                feed_forward_stack_bytes: 40,
                mixture_of_values_stack_bytes: 16,
            },
            K2HorizonMoVASparseLayerExpertPayload {
                decoder_layer_index: 2,
                feed_forward_stack_bytes: 40,
                mixture_of_values_stack_bytes: 16,
            },
            K2HorizonMoVASparseLayerExpertPayload {
                decoder_layer_index: 3,
                feed_forward_stack_bytes: 40,
                mixture_of_values_stack_bytes: 16,
            },
        ],
    )
    .expect("sparse layers should map onto plan slots");
    assert_eq!(
        geometries,
        vec![
            ExpertLayerGeometry {
                layer_index: 0,
                complete_layer_payload_bytes: 40,
                expert_payload_bytes: 10,
                expert_capacity: 4,
                experts_per_token: 2,
            },
            ExpertLayerGeometry {
                layer_index: 1,
                complete_layer_payload_bytes: 16,
                expert_payload_bytes: 8,
                expert_capacity: 2,
                experts_per_token: 1,
            },
            ExpertLayerGeometry {
                layer_index: 2,
                complete_layer_payload_bytes: 40,
                expert_payload_bytes: 10,
                expert_capacity: 4,
                experts_per_token: 2,
            },
            ExpertLayerGeometry {
                layer_index: 3,
                complete_layer_payload_bytes: 16,
                expert_payload_bytes: 8,
                expert_capacity: 2,
                experts_per_token: 1,
            },
            ExpertLayerGeometry {
                layer_index: 4,
                complete_layer_payload_bytes: 40,
                expert_payload_bytes: 10,
                expert_capacity: 4,
                experts_per_token: 2,
            },
            ExpertLayerGeometry {
                layer_index: 5,
                complete_layer_payload_bytes: 16,
                expert_payload_bytes: 8,
                expert_capacity: 2,
                experts_per_token: 1,
            },
        ]
    );
}
