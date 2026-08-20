use astronomical_model_serving::{
    Flux2KleinDimensionError, Flux2KleinImageDimensions, Flux2KleinMemoryAdmission,
    Flux2KleinMemoryAdmissionError, Flux2KleinMemoryGeometry, Flux2KleinResidencyMode,
};

#[test]
fn should_validate_model_geometry_and_worst_case_base64_transport_together() {
    let dimensions = Flux2KleinImageDimensions::validate(1_024, 1_024, 32 * 1024 * 1024)
        .expect("the official square image should fit the transport");

    assert_eq!(dimensions.pixel_count(), 1_048_576);
    assert_eq!(dimensions.latent_element_count(), 524_288);
    assert!(dimensions.maximum_base64_png_bytes() <= 32 * 1024 * 1024);

    assert!(matches!(
        Flux2KleinImageDimensions::validate(1_023, 1_024, 32 * 1024 * 1024),
        Err(Flux2KleinDimensionError::Unaligned { .. })
    ));
    assert!(matches!(
        Flux2KleinImageDimensions::validate(4_096, 256, 32 * 1024 * 1024),
        Err(Flux2KleinDimensionError::UnsupportedAspectRatio { .. })
    ));
    assert!(matches!(
        Flux2KleinImageDimensions::validate(1_024, 1_024, 1_000),
        Err(Flux2KleinDimensionError::TransportLimitExceeded { .. })
    ));
}

#[test]
fn should_report_the_vae_dominated_intrinsic_minimum_and_replan_from_high_residency() {
    let geometry = Flux2KleinMemoryGeometry {
        text_encoder_payload_bytes: 8_000,
        transformer_payload_bytes: 10_000,
        transformer_block_payload_bytes: vec![400; 25],
        vae_payload_bytes: 2_000,
        largest_component_load_page_bytes: 100,
        conditioning_bytes: 800,
        latent_state_bytes: 600,
        denoising_workspace_bytes: 500,
        vae_workspace_bytes: 1_500,
        host_rgb_bytes: 300,
        maximum_png_bytes: 400,
        maximum_base64_bytes: 536,
    };

    let resident = Flux2KleinMemoryAdmission::plan(20_000, &geometry)
        .expect("the complete components should fit sequentially");
    let minimum_mlx_memory_ceiling_bytes =
        Flux2KleinMemoryAdmission::minimum_mlx_memory_ceiling_bytes(&geometry)
            .expect("the intrinsic streamed execution minimum should be geometry-derived");
    assert_eq!(minimum_mlx_memory_ceiling_bytes, 3_800);
    assert_eq!(
        resident.minimum_mlx_memory_ceiling_bytes(),
        minimum_mlx_memory_ceiling_bytes
    );
    assert!(resident.peak_required_bytes() > minimum_mlx_memory_ceiling_bytes);
    assert_eq!(
        resident.text_encoder_mode(),
        Flux2KleinResidencyMode::Complete
    );
    assert_eq!(resident.retained_transformer_block_count(), 25);
    assert_eq!(resident.vae_mode(), Flux2KleinResidencyMode::Complete);
    assert!(resident.releases_text_encoder_before_transformer());
    assert!(resident.releases_transformer_before_vae());

    let minimum_ceiling_plan =
        Flux2KleinMemoryAdmission::plan(minimum_mlx_memory_ceiling_bytes, &geometry)
            .expect("a model admitted at a high ceiling should replan at its reported minimum");
    assert_eq!(
        minimum_ceiling_plan.peak_required_bytes(),
        minimum_mlx_memory_ceiling_bytes
    );
    assert!(
        minimum_ceiling_plan.retained_transformer_block_count()
            < resident.retained_transformer_block_count()
    );

    let staged = Flux2KleinMemoryAdmission::plan(4_000, &geometry)
        .expect("streaming should adapt while the complete VAE fits");
    assert_eq!(
        staged.text_encoder_mode(),
        Flux2KleinResidencyMode::Streamed
    );
    assert_eq!(staged.conditioning_peak_bytes(), 900);
    assert!(staged.retained_transformer_block_count() < 25);
    assert_eq!(staged.vae_mode(), Flux2KleinResidencyMode::Complete);

    assert!(matches!(
        Flux2KleinMemoryAdmission::plan(minimum_mlx_memory_ceiling_bytes - 1, &geometry),
        Err(
            Flux2KleinMemoryAdmissionError::CompleteVaeRequiresMoreMemory {
                required_bytes: 3_800,
                ceiling_bytes: 3_799,
            }
        )
    ));

    assert!(matches!(
        Flux2KleinMemoryAdmission::plan(1_000, &geometry),
        Err(Flux2KleinMemoryAdmissionError::InsufficientMemory { .. })
    ));

    let mut overflowing_geometry = geometry.clone();
    overflowing_geometry.conditioning_bytes = u64::MAX;
    assert_eq!(
        Flux2KleinMemoryAdmission::minimum_mlx_memory_ceiling_bytes(&overflowing_geometry),
        Err(Flux2KleinMemoryAdmissionError::GeometryOverflow)
    );

    let mut incomplete_geometry = geometry;
    incomplete_geometry.transformer_block_payload_bytes.pop();
    assert!(matches!(
        Flux2KleinMemoryAdmission::plan(20_000, &incomplete_geometry),
        Err(Flux2KleinMemoryAdmissionError::InvalidTransformerBlockCount { actual_count: 24 })
    ));
}
