//! User-journey tests over fake components, including failure and cancellation cleanup.

use std::sync::{Arc, Mutex};

use astronomical_ipc_protocol::{ImageGenerationFailureReason, ImageGenerationPhase, RequestId};
use astronomical_model_serving::{
    FLUX2_KLEIN_OFFICIAL_MODEL_ID, FLUX2_KLEIN_PROVIDER_MODEL_ID, ImageGenerationEngine,
    ImageGenerationEngineStep,
};

use super::lifecycle_support::{
    assert_final_cleanup_memory, bounded_conditioning_fake_engine, bounded_fake_engine,
    cloned_events, fake_engine, valid_command,
};

#[test]
fn should_generate_one_png_and_attribute_success_through_the_complete_user_journey() {
    let lifecycle_events = Arc::new(Mutex::new(Vec::new()));
    let mut engine = fake_engine(Arc::clone(&lifecycle_events), None);

    let loaded = engine
        .load()
        .expect("the fake official artifact should load");
    assert_eq!(loaded.model_id(), FLUX2_KLEIN_OFFICIAL_MODEL_ID);
    assert_eq!(loaded.minimum_mlx_memory_ceiling_bytes(), 400_000_000);

    let command = valid_command(41, 73);
    engine
        .start_generation(command)
        .expect("the valid image request should start");
    let mut completed = None;
    for _advance_index in 0..8 {
        let step = engine
            .advance_generation(RequestId::new(41))
            .expect("each bounded image phase should advance");
        if matches!(step, ImageGenerationEngineStep::Completed { .. }) {
            completed = Some(step);
            break;
        }
    }
    let ImageGenerationEngineStep::Completed {
        generated_image,
        result_metadata,
    } = completed.expect("the request should publish one completed image")
    else {
        panic!("the terminal engine step should contain the image");
    };
    assert_eq!(generated_image.mime_type, "image/png");
    assert_eq!(generated_image.encoded_bytes, b"complete png");
    assert_eq!(result_metadata.seed, 73);
    assert_eq!(
        (result_metadata.width_pixels, result_metadata.height_pixels),
        (64, 64)
    );
    assert_eq!(result_metadata.steps, 4);
    assert_final_cleanup_memory(engine.take_post_cleanup_memory_telemetry());
    assert_eq!(engine.take_post_cleanup_memory_telemetry(), None);
    assert_eq!(
        cloned_events(&lifecycle_events),
        vec![
            "load",
            "start:73",
            "condition:Romeo and Juliet",
            "noise:73",
            "denoise:0",
            "denoise:1",
            "denoise:2",
            "denoise:3",
            "decode",
            "encode",
            "finalize:success:12",
        ]
    );
}

#[test]
fn should_reject_non_official_controls_without_starting_components() {
    let lifecycle_events = Arc::new(Mutex::new(Vec::new()));
    let mut engine = fake_engine(Arc::clone(&lifecycle_events), None);
    engine
        .load()
        .expect("the fake official artifact should load");
    let mut invalid_command = valid_command(42, 9);
    invalid_command.settings.steps = 5;

    let failure = engine
        .start_generation(invalid_command)
        .expect_err("the distilled profile requires exactly four steps");

    assert!(matches!(
        failure,
        ImageGenerationFailureReason::InvalidRequest { .. }
    ));
    assert_eq!(cloned_events(&lifecycle_events), vec!["load"]);
}

#[test]
fn should_reject_provider_provenance_as_a_request_routing_identity() {
    let lifecycle_events = Arc::new(Mutex::new(Vec::new()));
    let mut engine = fake_engine(Arc::clone(&lifecycle_events), None);
    engine
        .load()
        .expect("the canonical FLUX.2 Klein engine should load");
    let mut provider_routed_command = valid_command(49, 17);
    provider_routed_command.model = FLUX2_KLEIN_PROVIDER_MODEL_ID.to_owned();

    let failure = engine
        .start_generation(provider_routed_command)
        .expect_err("provider provenance must not route a serving request");

    assert!(matches!(
        failure,
        ImageGenerationFailureReason::InvalidRequest { .. }
    ));
    assert_eq!(cloned_events(&lifecycle_events), vec!["load"]);
}

#[test]
fn should_attribute_failure_or_cancellation_before_cleanup_without_output() {
    let failure_events = Arc::new(Mutex::new(Vec::new()));
    let mut failing_engine = fake_engine(Arc::clone(&failure_events), Some("denoise:1"));
    failing_engine
        .load()
        .expect("the fake official artifact should load");
    failing_engine
        .start_generation(valid_command(43, 11))
        .expect("the valid request should start");
    for _advance_index in 0..3 {
        let _ = failing_engine.advance_generation(RequestId::new(43));
    }
    let failure = failing_engine
        .advance_generation(RequestId::new(43))
        .expect_err("the injected denoising failure should stop publication");
    assert!(matches!(
        failure,
        ImageGenerationFailureReason::FatalExecution { .. }
    ));
    let ImageGenerationFailureReason::FatalExecution { reason } = failure else {
        panic!("the injected execution failure should remain fatal");
    };
    assert!(
        !reason.contains('/'),
        "public execution failures must not expose paths"
    );
    assert!(!cloned_events(&failure_events).contains(&"encode".to_owned()));
    assert!(
        cloned_events(&failure_events)
            .contains(&"finalize:failed:injected failure at denoise:1".to_owned())
    );
    assert_final_cleanup_memory(failing_engine.take_post_cleanup_memory_telemetry());

    let cancellation_events = Arc::new(Mutex::new(Vec::new()));
    let mut cancelled_engine = fake_engine(Arc::clone(&cancellation_events), None);
    cancelled_engine
        .load()
        .expect("the fake official artifact should load");
    cancelled_engine
        .start_generation(valid_command(44, 12))
        .expect("the valid request should start");
    cancelled_engine
        .advance_generation(RequestId::new(44))
        .expect("conditioning should finish before cancellation");
    cancelled_engine
        .cancel_generation(RequestId::new(44))
        .expect("cancellation should synchronize and clear request state");
    assert_eq!(
        cloned_events(&cancellation_events)
            .last()
            .map(String::as_str),
        Some("finalize:cancelled:0")
    );
    assert!(!cloned_events(&cancellation_events).contains(&"encode".to_owned()));
    assert_final_cleanup_memory(cancelled_engine.take_post_cleanup_memory_telemetry());
}

#[test]
fn should_hold_the_euler_step_open_across_groups_cancel_and_reuse_the_engine() {
    let lifecycle_events = Arc::new(Mutex::new(Vec::new()));
    let mut engine = bounded_fake_engine(Arc::clone(&lifecycle_events), 3);
    engine
        .load()
        .expect("the fake official artifact should load");
    engine
        .start_generation(valid_command(45, 13))
        .expect("the first request should start");
    engine
        .advance_generation(RequestId::new(45))
        .expect("conditioning should complete");
    engine
        .advance_generation(RequestId::new(45))
        .expect("noise should complete");

    for _group_index in 0..2 {
        let ImageGenerationEngineStep::Progress {
            completed_steps, ..
        } = engine
            .advance_generation(RequestId::new(45))
            .expect("one transformer group should complete")
        else {
            panic!("a partial transformer forward must not publish an image");
        };
        assert_eq!(completed_steps, 0);
    }
    engine
        .cancel_generation(RequestId::new(45))
        .expect("cancellation should clean the partial forward");
    assert!(!cloned_events(&lifecycle_events).contains(&"group:0:2".to_owned()));
    assert!(!cloned_events(&lifecycle_events).contains(&"denoise:0".to_owned()));
    assert!(!cloned_events(&lifecycle_events).contains(&"encode".to_owned()));

    engine
        .start_generation(valid_command(46, 14))
        .expect("the engine should accept another request after cancellation");
    engine
        .advance_generation(RequestId::new(46))
        .expect("conditioning should complete again");
    engine
        .advance_generation(RequestId::new(46))
        .expect("noise should complete again");
    for _group_index in 0..3 {
        engine
            .advance_generation(RequestId::new(46))
            .expect("each bounded transformer group should complete");
    }
    let ImageGenerationEngineStep::Progress {
        completed_steps, ..
    } = engine
        .advance_generation(RequestId::new(46))
        .expect("the final output and Euler update should complete together")
    else {
        panic!("the completed Euler update should report progress");
    };
    assert_eq!(completed_steps, 1);
}

#[test]
fn should_yield_between_conditioning_layers_cancel_without_later_reads_and_reuse_the_engine() {
    let lifecycle_events = Arc::new(Mutex::new(Vec::new()));
    let mut engine = bounded_conditioning_fake_engine(Arc::clone(&lifecycle_events), 3);
    engine
        .load()
        .expect("the fake official artifact should load");
    engine
        .start_generation(valid_command(47, 15))
        .expect("the first Romeo and Juliet request should start");

    let ImageGenerationEngineStep::Progress {
        phase,
        completed_steps,
        ..
    } = engine
        .advance_generation(RequestId::new(47))
        .expect("one text-encoder layer group should complete")
    else {
        panic!("partial conditioning must report progress");
    };
    assert_eq!(phase, ImageGenerationPhase::EncodingPrompt);
    assert_eq!(completed_steps, 0);
    engine
        .cancel_generation(RequestId::new(47))
        .expect("cancellation should release partial conditioning before acknowledgement");
    assert!(cloned_events(&lifecycle_events).contains(&"condition-layer:0".to_owned()));
    assert!(!cloned_events(&lifecycle_events).contains(&"condition-layer:1".to_owned()));

    engine
        .start_generation(valid_command(48, 16))
        .expect("the engine should accept a second Romeo and Juliet request");
    for expected_layer_index in 0..3 {
        let ImageGenerationEngineStep::Progress {
            phase,
            completed_steps,
            ..
        } = engine
            .advance_generation(RequestId::new(48))
            .expect("each text-encoder layer group should yield")
        else {
            panic!("bounded conditioning must remain in progress");
        };
        assert_eq!(phase, ImageGenerationPhase::EncodingPrompt);
        assert_eq!(completed_steps, 0);
        let expected_event = format!("condition-layer:{expected_layer_index}");
        assert_eq!(
            cloned_events(&lifecycle_events).last(),
            Some(&expected_event)
        );
    }
    let ImageGenerationEngineStep::Progress {
        phase,
        completed_steps,
        ..
    } = engine
        .advance_generation(RequestId::new(48))
        .expect("tap concatenation should finish conditioning")
    else {
        panic!("completed conditioning should still report prompt progress");
    };
    assert_eq!(phase, ImageGenerationPhase::EncodingPrompt);
    assert_eq!(completed_steps, 0);
    assert_eq!(
        cloned_events(&lifecycle_events).last().map(String::as_str),
        Some("condition-complete")
    );

    let ImageGenerationEngineStep::Progress {
        phase,
        completed_steps,
        ..
    } = engine
        .advance_generation(RequestId::new(48))
        .expect("keyed noise preparation should transition monotonically into denoising")
    else {
        panic!("completed noise preparation should report denoising progress");
    };
    assert_eq!(phase, ImageGenerationPhase::Denoising);
    assert_eq!(completed_steps, 0);
}
