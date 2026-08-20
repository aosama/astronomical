//! Serialization contracts for the image-specific report and operation catalog.

use std::time::Duration;

use astronomical_model_serving::{
    FLUX2_KLEIN_OFFICIAL_MODEL_ID, FLUX2_KLEIN_OFFICIAL_REVISION, PerformanceAttribution,
    PerformanceAttributionOutcome, PerformanceOperation,
};

#[test]
fn should_serialize_image_identity_controls_memory_and_success_outcome() {
    let report = PerformanceAttribution::enabled()
        .finish_image_generation(
            PerformanceAttributionOutcome::Success,
            47,
            FLUX2_KLEIN_OFFICIAL_MODEL_ID.to_owned(),
            FLUX2_KLEIN_OFFICIAL_REVISION.to_owned(),
            1_024,
            768,
            4,
            1_000,
            73,
            Some(12_345),
            (Some(101), Some(102), Some(103)),
            (Some(201), Some(202), Some(203)),
            None,
        )
        .expect("enabled image attribution should produce a report");
    let report_json = serde_json::to_value(report).expect("the image report should serialize");

    assert_eq!(report_json["report_kind"], "image_generation");
    assert_eq!(report_json["request_id"], 47);
    assert_eq!(report_json["model_id"], FLUX2_KLEIN_OFFICIAL_MODEL_ID);
    assert_eq!(report_json["model_revision"], FLUX2_KLEIN_OFFICIAL_REVISION);
    assert_eq!(report_json["width_pixels"], 1_024);
    assert_eq!(report_json["height_pixels"], 768);
    assert_eq!(report_json["steps"], 4);
    assert_eq!(report_json["guidance_thousandths"], 1_000);
    assert_eq!(report_json["seed"], 73);
    assert_eq!(report_json["encoded_bytes"], 12_345);
    assert_eq!(report_json["outcome"], "success");
    assert_eq!(report_json["memory_snapshots"][0]["phase"], "request_start");
    assert_eq!(report_json["memory_snapshots"][1]["phase"], "final_cleanup");
}

#[test]
fn should_serialize_every_image_operation_with_graph_and_synchronization_separate() {
    let image_operations = [
        (
            PerformanceOperation::ArtifactValidation,
            "artifact_validation",
        ),
        (PerformanceOperation::PromptRendering, "prompt_rendering"),
        (
            PerformanceOperation::PromptTokenization,
            "prompt_tokenization",
        ),
        (
            PerformanceOperation::ImageRuntimeSetup,
            "image_runtime_setup",
        ),
        (
            PerformanceOperation::ImageTextComponentMapping,
            "image_text_component_mapping",
        ),
        (
            PerformanceOperation::ImageTextComponentLoading,
            "image_text_component_loading",
        ),
        (
            PerformanceOperation::ImageQwenLayerGraphConstruction,
            "image_qwen_layer_graph_construction",
        ),
        (
            PerformanceOperation::ImageQwenLayerSynchronizationWait,
            "image_qwen_layer_synchronization_wait",
        ),
        (
            PerformanceOperation::SeededNoiseGraphConstruction,
            "seeded_noise_graph_construction",
        ),
        (
            PerformanceOperation::SeededNoiseSynchronizationWait,
            "seeded_noise_synchronization_wait",
        ),
        (
            PerformanceOperation::ImageScheduleConstruction,
            "image_schedule_construction",
        ),
        (
            PerformanceOperation::ImagePositionGraphConstruction,
            "image_position_graph_construction",
        ),
        (
            PerformanceOperation::ImagePositionSynchronizationWait,
            "image_position_synchronization_wait",
        ),
        (
            PerformanceOperation::ImageTransformerComponentMapping,
            "image_transformer_component_mapping",
        ),
        (
            PerformanceOperation::ImageTransformerComponentLoading,
            "image_transformer_component_loading",
        ),
        (
            PerformanceOperation::ImageDenoisingStepSpan,
            "image_denoising_step_span",
        ),
        (
            PerformanceOperation::ImageTransformerBlockGroupGraphConstruction,
            "image_transformer_block_group_graph_construction",
        ),
        (
            PerformanceOperation::ImageTransformerBlockGroupSynchronizationWait,
            "image_transformer_block_group_synchronization_wait",
        ),
        (
            PerformanceOperation::ImageSchedulerUpdateGraphConstruction,
            "image_scheduler_update_graph_construction",
        ),
        (
            PerformanceOperation::ImageSchedulerUpdateSynchronizationWait,
            "image_scheduler_update_synchronization_wait",
        ),
        (
            PerformanceOperation::ImageTransformerRelease,
            "image_transformer_release",
        ),
        (
            PerformanceOperation::ImageVaeComponentMapping,
            "image_vae_component_mapping",
        ),
        (
            PerformanceOperation::ImageVaeComponentLoading,
            "image_vae_component_loading",
        ),
        (
            PerformanceOperation::ImageVaeCompleteDecodeGraphConstruction,
            "image_vae_complete_decode_graph_construction",
        ),
        (
            PerformanceOperation::ImageVaeDecodeSynchronizationWait,
            "image_vae_decode_synchronization_wait",
        ),
        (
            PerformanceOperation::ImagePixelConversionGraphConstruction,
            "image_pixel_conversion_graph_construction",
        ),
        (
            PerformanceOperation::ImagePixelTransfer,
            "image_pixel_transfer",
        ),
        (PerformanceOperation::ImagePngEncoding, "image_png_encoding"),
        (
            PerformanceOperation::ImageCancellationSynchronization,
            "image_cancellation_synchronization",
        ),
        (
            PerformanceOperation::ImageComponentRelease,
            "image_component_release",
        ),
        (
            PerformanceOperation::ImageFinalCleanup,
            "image_final_cleanup",
        ),
    ];
    let mut attribution = PerformanceAttribution::enabled();
    for (operation_index, (operation, _identifier)) in image_operations.iter().enumerate() {
        attribution.record_completed_operation(
            *operation,
            Duration::from_nanos(operation_index as u64 * 2),
            Duration::from_nanos(operation_index as u64 * 2 + 1),
        );
    }
    let report = attribution
        .finish_image_generation(
            PerformanceAttributionOutcome::Failed,
            48,
            FLUX2_KLEIN_OFFICIAL_MODEL_ID.to_owned(),
            FLUX2_KLEIN_OFFICIAL_REVISION.to_owned(),
            64,
            64,
            4,
            1_000,
            74,
            None,
            (None, None, None),
            (None, None, None),
            Some("bounded image execution failed".to_owned()),
        )
        .expect("enabled image attribution should produce a report");
    let report_json = serde_json::to_value(report).expect("the image report should serialize");
    let identifiers = report_json["operations"]
        .as_array()
        .expect("operations should serialize as an array")
        .iter()
        .filter_map(|operation| operation["operation"].as_str())
        .collect::<Vec<_>>();

    assert_eq!(
        identifiers,
        image_operations
            .iter()
            .map(|(_operation, identifier)| *identifier)
            .collect::<Vec<_>>(),
    );
    assert_eq!(report_json["outcome"], "failed");
    assert_eq!(
        report_json["failure_description"],
        "bounded image execution failed"
    );
}

#[test]
fn should_keep_disabled_image_attribution_inert() {
    let mut attribution = PerformanceAttribution::disabled();
    let mut closure_executed = false;
    attribution.measure_operation(PerformanceOperation::ImagePngEncoding, |_| {
        closure_executed = true;
    });

    assert!(closure_executed);
    assert!(
        attribution
            .operation_measurement(PerformanceOperation::ImagePngEncoding)
            .is_none()
    );
    assert!(
        attribution
            .finish_image_generation(
                PerformanceAttributionOutcome::Success,
                49,
                String::new(),
                String::new(),
                64,
                64,
                4,
                1_000,
                75,
                None,
                (None, None, None),
                (None, None, None),
                None,
            )
            .is_none()
    );
}
