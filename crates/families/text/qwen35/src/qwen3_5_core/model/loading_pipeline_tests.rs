//! Unit contracts for the pure decision inside the artifact-loading pipeline:
//! which vision binding a request and an artifact resolve to. Mapping the
//! shards and binding the weights need a live MLX runtime and real artifacts,
//! which the engine's loading tests cover.

use super::VisionBinding;

#[test]
fn should_bind_a_sidecar_when_the_artifact_ships_one_and_vision_was_asked_for() {
    assert_eq!(
        VisionBinding::for_request(true, true, true),
        VisionBinding::SeparateSidecar
    );
}

#[test]
fn should_prefer_the_sidecar_even_when_the_artifact_reports_no_embedded_tower() {
    assert_eq!(
        VisionBinding::for_request(true, true, false),
        VisionBinding::SeparateSidecar
    );
}

#[test]
fn should_bind_the_embedded_tower_when_there_is_no_sidecar_and_the_artifact_takes_images() {
    assert_eq!(
        VisionBinding::for_request(true, false, true),
        VisionBinding::EmbeddedInModelShards
    );
}

#[test]
fn should_bind_no_vision_when_the_artifact_neither_ships_a_sidecar_nor_takes_images() {
    assert_eq!(
        VisionBinding::for_request(true, false, false),
        VisionBinding::NotBound
    );
}

#[test]
fn should_bind_no_vision_when_the_caller_did_not_ask_for_it() {
    assert_eq!(
        VisionBinding::for_request(false, true, true),
        VisionBinding::NotBound
    );
    assert_eq!(
        VisionBinding::for_request(false, false, true),
        VisionBinding::NotBound
    );
}
