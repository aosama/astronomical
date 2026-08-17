//! Resident-MoE binding failures must precede the first model forward.

use astronomical_model_serving::{
    LagunaExpertProjection, LagunaLayerTensorRole, LagunaNativeWeights, LagunaTargetNormalizer,
};

use super::rows::generic_moe_rows;
use super::tensor_fixture::build_tensor_inventories;
use super::tensor_identity::layer_id;

#[tokio::test]
async fn should_reject_incomplete_resident_moe_assemblies_and_correction_bias() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = super::test_runtime();
    let row = generic_moe_rows()
        .into_iter()
        .find(|row| row.row_name == "native_mixed_shared")
        .expect("the shared resident-MoE row should exist");
    let contract = LagunaTargetNormalizer::normalize(
        &serde_json::to_vec(&row.target_config).expect("binding config should serialize"),
    )
    .expect("binding contract should normalize");
    let sparse_layer_index = 1;

    let mut missing_routed_projection = build_tensor_inventories(&runtime, &contract, &row);
    missing_routed_projection
        .production_tensors
        .remove(&layer_id(
            sparse_layer_index,
            LagunaLayerTensorRole::RoutedExpert(LagunaExpertProjection::Up),
        ));
    assert!(
        LagunaNativeWeights::bind(
            &runtime,
            missing_routed_projection.production_tensors,
            &contract,
        )
        .is_err()
    );

    let mut missing_shared_projection = build_tensor_inventories(&runtime, &contract, &row);
    missing_shared_projection
        .production_tensors
        .remove(&layer_id(
            sparse_layer_index,
            LagunaLayerTensorRole::SharedExpert(LagunaExpertProjection::Down),
        ));
    assert!(
        LagunaNativeWeights::bind(
            &runtime,
            missing_shared_projection.production_tensors,
            &contract,
        )
        .is_err()
    );

    let mut malformed_correction_bias = build_tensor_inventories(&runtime, &contract, &row);
    malformed_correction_bias.production_tensors.insert(
        layer_id(
            sparse_layer_index,
            LagunaLayerTensorRole::RouterCorrectionBias,
        ),
        runtime
            .array_from_f32(&[0.0; 5], &[5])
            .expect("malformed correction-bias fixture should construct"),
    );
    assert!(
        LagunaNativeWeights::bind(
            &runtime,
            malformed_correction_bias.production_tensors,
            &contract,
        )
        .is_err()
    );
}
