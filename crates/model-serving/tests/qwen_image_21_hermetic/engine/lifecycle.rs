//! User-journey tests over the scripted components owner, including failure, cancellation,
//! repeat requests, and the admission guards a worker relies on.

use std::sync::{Arc, Mutex};

use astronomical_ipc_protocol::{ImageGenerationFailureReason, ImageGenerationPhase, RequestId};
use astronomical_model_serving::{
    ImageGenerationEngine, ImageGenerationEngineStep, QWEN_IMAGE_21_OFFICIAL_MODEL_ID,
    QWEN_IMAGE_21_PROVIDER_MODEL_ID, QwenImage21RenderAdvance,
};

use super::lifecycle_support::{
    assert_final_cleanup_memory, cloned_events, fake_engine, scripted_advances_engine,
    six_step_render_advances, valid_command,
};

#[test]
fn should_generate_one_png_and_attribute_success_through_the_complete_user_journey() {
    let lifecycle_events = Arc::new(Mutex::new(Vec::new()));
    let mut engine =
        scripted_advances_engine(Arc::clone(&lifecycle_events), six_step_render_advances());

    let loaded = engine
        .load()
        .expect("the scripted official artifact should load");
    assert_eq!(loaded.model_id(), QWEN_IMAGE_21_OFFICIAL_MODEL_ID);
    assert_eq!(loaded.minimum_mlx_memory_ceiling_bytes(), 400_000_000);
    assert_eq!(loaded.capabilities().maximum_steps, 40);
    assert_eq!(loaded.capabilities().dimension_multiple_pixels, 32);
    assert_eq!(loaded.capabilities().minimum_width_pixels, 256);
    assert_eq!(loaded.capabilities().maximum_width_pixels, 1_024);
    assert_eq!(loaded.capabilities().maximum_guidance_thousandths, 1_000);

    let command = valid_command(41, 73);
    engine
        .start_generation(command)
        .expect("the valid Romeo and Juliet request should start");
    let mut completed = None;
    for _advance_index in 0..12 {
        let step = engine
            .advance_generation(RequestId::new(41))
            .expect("each bounded render boundary should advance");
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
    let decoded_png = image::load_from_memory(&generated_image.encoded_bytes)
        .expect("the published bytes should be one decodable PNG");
    assert_eq!((decoded_png.width(), decoded_png.height()), (8, 8));
    assert_eq!(result_metadata.seed, 73);
    assert_eq!(
        (result_metadata.width_pixels, result_metadata.height_pixels),
        (1_024, 1_024)
    );
    assert_eq!(result_metadata.steps, 6);
    assert_eq!(result_metadata.guidance_thousandths, 1_000);

    // Success attribution must report the exact byte count of the published image.
    let success_events = cloned_events(&lifecycle_events);
    let published_byte_count = generated_image.encoded_bytes.len();
    let expected_finalization = format!("finalize:success:{published_byte_count}");
    assert_eq!(
        success_events.last().map(String::as_str),
        Some(expected_finalization.as_str())
    );
    assert_eq!(
        success_events,
        vec![
            "load",
            "start:41:1024x1024:6:73",
            "prepare",
            "advance:1",
            "advance:2",
            "advance:3",
            "advance:4",
            "advance:5",
            "advance:6",
            "advance:7",
            "advance:8",
            "advance:9",
            "advance:10",
            expected_finalization.as_str(),
        ]
    );
    assert_final_cleanup_memory(engine.take_post_cleanup_memory_telemetry());
    assert_eq!(engine.take_post_cleanup_memory_telemetry(), None);
}

#[test]
fn should_serve_a_second_request_after_the_first_publishes() {
    // The native owner constructs one pipeline per render (weights release between requests),
    // so the engine must accept and complete a second request immediately after the first.
    let lifecycle_events = Arc::new(Mutex::new(Vec::new()));
    let mut engine =
        scripted_advances_engine(Arc::clone(&lifecycle_events), six_step_render_advances());
    engine
        .load()
        .expect("the scripted official artifact should load");
    engine
        .start_generation(valid_command(51, 101))
        .expect("the first request should start");
    for _advance_index in 0..12 {
        if matches!(
            engine
                .advance_generation(RequestId::new(51))
                .expect("the first request should complete"),
            ImageGenerationEngineStep::Completed { .. }
        ) {
            break;
        }
    }
    engine
        .start_generation(valid_command(52, 202))
        .expect("a completed engine must accept the next request");
    let mut second_completion = None;
    for _advance_index in 0..12 {
        let step = engine
            .advance_generation(RequestId::new(52))
            .expect("the second request should advance");
        if let ImageGenerationEngineStep::Completed {
            result_metadata, ..
        } = step
        {
            second_completion = Some(result_metadata);
            break;
        }
    }
    let second_metadata =
        second_completion.expect("the second request should publish one completed image");
    assert_eq!(second_metadata.seed, 202);
    let events = cloned_events(&lifecycle_events);
    assert_eq!(
        events
            .iter()
            .filter(|event| event.as_str() == "load")
            .count(),
        1,
        "one loaded engine serves both requests"
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| event.starts_with("finalize:success:"))
            .count(),
        2,
        "both requests finalize with their own success attribution"
    );
}

#[test]
fn should_reject_requests_outside_the_official_envelope_without_starting_components() {
    // Every invalid control must stop before `start_request`, leaving only the load event.
    let invalid_controls = [
        ("width is not a multiple of 32", 1_024, 1_000, 6, 1_000),
        ("width below the official minimum", 224, 1_024, 6, 1_000),
        ("height above the official maximum", 1_024, 1_056, 6, 1_000),
        ("no denoising steps", 1_024, 1_024, 0, 1_000),
        ("steps above the reference default", 1_024, 1_024, 41, 1_000),
        ("guidance must stay exactly 1.0", 1_024, 1_024, 6, 999),
    ];
    for (case_description, width_pixels, height_pixels, steps, guidance_thousandths) in
        invalid_controls
    {
        let lifecycle_events = Arc::new(Mutex::new(Vec::new()));
        let mut engine =
            scripted_advances_engine(Arc::clone(&lifecycle_events), six_step_render_advances());
        engine
            .load()
            .expect("the scripted official artifact should load");
        let mut invalid_command = valid_command(42, 9);
        invalid_command.settings.width_pixels = width_pixels;
        invalid_command.settings.height_pixels = height_pixels;
        invalid_command.settings.steps = steps;
        invalid_command.settings.guidance_thousandths = guidance_thousandths;
        let rejection_message = format!("the request should be rejected: {case_description}");

        let failure = engine
            .start_generation(invalid_command)
            .expect_err(&rejection_message);

        assert!(
            matches!(failure, ImageGenerationFailureReason::InvalidRequest { .. }),
            "{case_description} should be a request-scoped rejection, got {failure:?}"
        );
        assert_eq!(
            cloned_events(&lifecycle_events),
            vec!["load"],
            "{case_description} must not reach the components"
        );
    }
}

#[test]
fn should_reject_non_canonical_routing_identities_without_starting_components() {
    for non_canonical_model_id in [QWEN_IMAGE_21_PROVIDER_MODEL_ID, "qwen/qwen-image-2.1"] {
        let lifecycle_events = Arc::new(Mutex::new(Vec::new()));
        let mut engine =
            scripted_advances_engine(Arc::clone(&lifecycle_events), six_step_render_advances());
        engine
            .load()
            .expect("the scripted official artifact should load");
        let mut misrouted_command = valid_command(49, 17);
        misrouted_command.model = non_canonical_model_id.to_owned();

        let failure = engine
            .start_generation(misrouted_command)
            .expect_err("only the canonical serving identity may route a request");

        assert!(matches!(
            failure,
            ImageGenerationFailureReason::InvalidRequest { .. }
        ));
        assert_eq!(cloned_events(&lifecycle_events), vec!["load"]);
    }
}

#[test]
fn should_refuse_generation_before_load_and_while_busy() {
    let lifecycle_events = Arc::new(Mutex::new(Vec::new()));
    let mut engine =
        scripted_advances_engine(Arc::clone(&lifecycle_events), six_step_render_advances());

    let unloaded_failure = engine
        .start_generation(valid_command(53, 31))
        .expect_err("generation must not start before the engine loads");

    assert!(matches!(
        unloaded_failure,
        ImageGenerationFailureReason::FatalExecution { .. }
    ));
    assert_eq!(cloned_events(&lifecycle_events), Vec::<String>::new());

    engine
        .load()
        .expect("the scripted official artifact should load");
    engine
        .start_generation(valid_command(54, 32))
        .expect("the first request should start");
    let busy_failure = engine
        .start_generation(valid_command(55, 33))
        .expect_err("one render at a time keeps memory bounded");

    assert!(matches!(
        busy_failure,
        ImageGenerationFailureReason::EngineBusy
    ));
    assert_eq!(
        cloned_events(&lifecycle_events),
        vec!["load", "start:54:1024x1024:6:32"]
    );

    let limit_failure = engine
        .update_mlx_memory_limit(8_000_000_000)
        .expect_err("the memory limit must stay fixed while a request renders");

    assert!(matches!(
        limit_failure,
        ImageGenerationFailureReason::EngineBusy
    ));
}

#[test]
fn should_keep_the_active_request_when_the_advance_belongs_to_another_request() {
    let lifecycle_events = Arc::new(Mutex::new(Vec::new()));
    let mut engine =
        scripted_advances_engine(Arc::clone(&lifecycle_events), six_step_render_advances());
    engine
        .load()
        .expect("the scripted official artifact should load");
    engine
        .start_generation(valid_command(56, 41))
        .expect("the valid request should start");

    let mismatch_failure = engine
        .advance_generation(RequestId::new(99))
        .expect_err("an unmatched request identifier must not advance a render");

    assert!(matches!(
        mismatch_failure,
        ImageGenerationFailureReason::InvalidRequest { .. }
    ));

    let step = engine
        .advance_generation(RequestId::new(56))
        .expect("the active request should still advance after the mismatch");

    assert!(matches!(step, ImageGenerationEngineStep::Progress { .. }));
    let cancel_mismatch = engine
        .cancel_generation(RequestId::new(99))
        .expect_err("cancellation must not target another request");

    assert!(matches!(
        cancel_mismatch,
        ImageGenerationFailureReason::InvalidRequest { .. }
    ));
    engine
        .cancel_generation(RequestId::new(56))
        .expect("the matching request should still cancel cleanly");
}

#[test]
fn should_attribute_failure_before_cleanup_without_publishing_an_image() {
    let failure_events = Arc::new(Mutex::new(Vec::new()));
    let mut failing_engine = scripted_advances_engine(
        Arc::clone(&failure_events),
        vec![
            Ok(QwenImage21RenderAdvance::Preparing),
            Ok(QwenImage21RenderAdvance::ConditioningCompleted),
            Ok(QwenImage21RenderAdvance::NoisePrepared),
            Err("injected denoising failure".to_owned()),
        ],
    );
    failing_engine
        .load()
        .expect("the scripted official artifact should load");
    failing_engine
        .start_generation(valid_command(43, 11))
        .expect("the valid request should start");
    // Three boundaries succeed; the scripted denoising failure surfaces on the fourth.
    for _advance_index in 0..3 {
        failing_engine
            .advance_generation(RequestId::new(43))
            .expect("the scripted boundaries before the failure should progress");
    }
    let failure = failing_engine
        .advance_generation(RequestId::new(43))
        .expect_err("the injected denoising failure should stop publication");
    let ImageGenerationFailureReason::FatalExecution { reason } = failure else {
        panic!("an injected mid-render failure should remain fatal");
    };
    assert!(
        !reason.contains('/'),
        "public execution failures must not expose paths"
    );
    assert!(
        !cloned_events(&failure_events)
            .iter()
            .any(|event| event.starts_with("finalize:success")),
        "a failed render must not publish success attribution"
    );
    assert_eq!(
        cloned_events(&failure_events).last().map(String::as_str),
        Some("finalize:failed:injected denoising failure")
    );
    assert_final_cleanup_memory(failing_engine.take_post_cleanup_memory_telemetry());

    let unload_attempt = failing_engine
        .advance_generation(RequestId::new(43))
        .expect_err("the failed request is closed and cannot advance further");
    assert!(matches!(
        unload_attempt,
        ImageGenerationFailureReason::InvalidRequest { .. }
    ));
}

#[test]
fn should_attribute_a_load_failure_and_close_the_engine_to_generation() {
    let lifecycle_events = Arc::new(Mutex::new(Vec::new()));
    let mut failing_engine = fake_engine(Arc::clone(&lifecycle_events), Some("load"));

    let load_failure = failing_engine
        .load()
        .expect_err("a component load failure must surface as fatal");

    assert!(matches!(
        load_failure,
        ImageGenerationFailureReason::FatalExecution { .. }
    ));
    let ImageGenerationFailureReason::FatalExecution { reason } = load_failure else {
        panic!("a load failure should be fatal, not a request rejection");
    };
    assert!(
        !reason.contains('/'),
        "public load failures must not expose paths"
    );
    assert!(matches!(
        failing_engine.start_generation(valid_command(57, 51)),
        Err(ImageGenerationFailureReason::FatalExecution { .. })
    ));
}

#[test]
fn should_cancel_mid_render_without_publishing_and_reuse_the_engine() {
    let cancellation_events = Arc::new(Mutex::new(Vec::new()));
    let mut cancelled_engine =
        scripted_advances_engine(Arc::clone(&cancellation_events), six_step_render_advances());
    cancelled_engine
        .load()
        .expect("the scripted official artifact should load");
    cancelled_engine
        .start_generation(valid_command(44, 12))
        .expect("the valid request should start");
    cancelled_engine
        .advance_generation(RequestId::new(44))
        .expect("the preparation boundary should finish before cancellation");
    cancelled_engine
        .cancel_generation(RequestId::new(44))
        .expect("cancellation should release the render and its request state");
    assert_eq!(
        cloned_events(&cancellation_events)
            .last()
            .map(String::as_str),
        Some("finalize:cancelled:0")
    );
    assert!(
        !cloned_events(&cancellation_events)
            .iter()
            .any(|event| event.starts_with("finalize:success")),
        "a cancelled render must not publish success attribution"
    );
    assert_final_cleanup_memory(cancelled_engine.take_post_cleanup_memory_telemetry());

    cancelled_engine
        .start_generation(valid_command(45, 13))
        .expect("the engine should accept another request after cancellation");
    cancelled_engine
        .advance_generation(RequestId::new(45))
        .expect("the reused engine should advance the next request");
}

#[test]
fn should_map_each_render_boundary_onto_the_progress_phases_in_order() {
    let lifecycle_events = Arc::new(Mutex::new(Vec::new()));
    let mut engine =
        scripted_advances_engine(Arc::clone(&lifecycle_events), six_step_render_advances());
    engine
        .load()
        .expect("the scripted official artifact should load");
    engine
        .start_generation(valid_command(46, 61))
        .expect("the valid request should start");

    let expected_phase_sequence = [
        (ImageGenerationPhase::Preparing, 0),
        (ImageGenerationPhase::EncodingPrompt, 0),
        (ImageGenerationPhase::Denoising, 0),
        (ImageGenerationPhase::Denoising, 1),
        (ImageGenerationPhase::Denoising, 2),
        (ImageGenerationPhase::Denoising, 3),
        (ImageGenerationPhase::Denoising, 4),
        (ImageGenerationPhase::Denoising, 5),
        (ImageGenerationPhase::Denoising, 6),
        (ImageGenerationPhase::Decoding, 6),
    ];
    for (expected_phase, expected_completed_steps) in expected_phase_sequence {
        let ImageGenerationEngineStep::Progress {
            phase,
            completed_steps,
            total_steps,
            ..
        } = engine
            .advance_generation(RequestId::new(46))
            .expect("each scripted boundary should report progress")
        else {
            panic!("no scripted boundary should publish an image early");
        };
        assert_eq!(phase, expected_phase);
        assert_eq!(completed_steps, expected_completed_steps);
        assert_eq!(total_steps, 6);
    }
    assert!(matches!(
        engine
            .advance_generation(RequestId::new(46))
            .expect("the render should complete after the decoding boundary"),
        ImageGenerationEngineStep::Completed { .. }
    ));
}
