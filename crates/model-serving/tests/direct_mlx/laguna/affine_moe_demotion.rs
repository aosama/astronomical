//! Direct-MLX acceptance journey for the Laguna sorted-reduction capability
//! demotion: a forced-unsupported verdict must route the sparse forward
//! through the unsorted MLX fallback with logits equal to the supported run.

use std::time::Duration;

use astronomical_model_serving::{
    CustomKernelVerdict, CustomMetalKernelFamily, KernelUnsupportedReason, LagunaDecoderState,
    LagunaModel, PerformanceAttribution, WorkerKernelCapabilities,
};
use astronomical_runtime_integration::MlxRuntime;

use crate::common::direct_mlx_test_guard;
use crate::direct_mlx::laguna::affine_moe::{
    affine_sparse_contract, bind_affine_sparse_model, test_runtime,
};

#[tokio::test]
async fn should_produce_equal_sparse_logits_when_the_sorted_reduction_kernel_is_demoted() {
    let _direct_mlx_guard = direct_mlx_test_guard().await;
    let bounded_demotion_journey = tokio::time::timeout(Duration::from_secs(120), async {
        let runtime = test_runtime();
        // 40 prompt tokens with two experts per token produce 80 assignments,
        // above the 64-assignment floor, so the supported model genuinely takes
        // the sorted custom-kernel route.
        let prompt_token_ids: Vec<u32> = (1..=40).map(|token_id| token_id % 8).collect();
        let prompt_tokens = runtime
            .array_from_u32(&prompt_token_ids, &[40])
            .expect("the 40-token prompt should be valid");

        let supported_capabilities = crate::common::test_worker_kernel_capabilities(&runtime);
        let (supported_contract, supported_weights) =
            bind_affine_sparse_model(&runtime, 4, 32, true)
                .unwrap_or_else(|_| panic!("4-bit affine sparse weights should bind"));
        let supported_logits = LagunaModel::new(
            affine_sparse_contract(4, 32, 128),
            supported_weights,
            supported_capabilities,
        )
        .expect("the supported model should construct")
        .forward(
            &runtime,
            &prompt_tokens,
            &mut LagunaDecoderState::empty(&supported_contract)
                .expect("decoder state should allocate"),
            &mut PerformanceAttribution::disabled(),
        )
        .expect("the supported sparse forward should execute");

        let demoted_capabilities = WorkerKernelCapabilities::with_forced_verdicts_for_tests([(
            CustomMetalKernelFamily::SortedExpertWeightedSum,
            CustomKernelVerdict::Unsupported(KernelUnsupportedReason::OutputMismatch {
                description: "forced demotion for the parity journey".to_owned(),
            }),
        )]);
        let (demoted_contract, demoted_weights) = bind_affine_sparse_model(&runtime, 4, 32, true)
            .unwrap_or_else(|_| panic!("4-bit affine sparse weights should bind"));
        let demoted_logits = LagunaModel::new(
            affine_sparse_contract(4, 32, 128),
            demoted_weights,
            &demoted_capabilities,
        )
        .expect("the demoted model should construct")
        .forward(
            &runtime,
            &prompt_tokens,
            &mut LagunaDecoderState::empty(&demoted_contract)
                .expect("decoder state should allocate"),
            &mut PerformanceAttribution::disabled(),
        )
        .expect("the demoted sparse forward should execute through the MLX fallback");

        let supported_values = supported_logits
            .to_vec_f32()
            .expect("supported host values");
        let demoted_values = demoted_logits.to_vec_f32().expect("demoted host values");
        assert_eq!(supported_values.len(), demoted_values.len());
        for (supported_value, demoted_value) in supported_values.iter().zip(demoted_values.iter()) {
            let comparison_scale = supported_value.abs().max(1.0);
            assert!(
                (supported_value - demoted_value).abs() <= 1e-5 * comparison_scale,
                "demoted logits {demoted_value} must match supported logits {supported_value}"
            );
        }
    })
    .await;
    assert!(
        bounded_demotion_journey.is_ok(),
        "the demotion parity journey must finish within the 120-second bound"
    );
}
