use super::{RoutedReply, RoutedReplyDecodeError};

fn recording_bytes(
    layer_count: u16,
    experts_per_token: u16,
    experts_per_layer: u16,
    token_count: u32,
    route_bytes: &[u8],
) -> Vec<u8> {
    let mut recording = b"ASTROROUTES1".to_vec();
    recording.extend_from_slice(&layer_count.to_le_bytes());
    recording.extend_from_slice(&experts_per_token.to_le_bytes());
    recording.extend_from_slice(&experts_per_layer.to_le_bytes());
    recording.extend_from_slice(&token_count.to_le_bytes());
    recording.extend_from_slice(route_bytes);
    recording
}

#[test]
fn should_decode_every_route_in_token_then_layer_order() {
    let recording = recording_bytes(2, 2, 8, 2, &[0, 1, 2, 3, 4, 5, 6, 7]);

    let routed_reply = RoutedReply::from_recording_bytes(&recording).expect("a valid recording");

    assert_eq!(routed_reply.token_count(), 2);
    assert_eq!(routed_reply.layer_count(), 2);
    assert_eq!(routed_reply.layer_route(0, 1), &[2, 3]);
    assert_eq!(routed_reply.layer_route(1, 0), &[4, 5]);
}

#[test]
fn should_decode_a_recording_of_no_tokens() {
    let recording = recording_bytes(4, 8, 256, 0, &[]);

    let routed_reply = RoutedReply::from_recording_bytes(&recording).expect("an empty recording");

    assert_eq!(routed_reply.token_count(), 0);
}

#[test]
fn should_refuse_bytes_that_are_not_a_recording() {
    assert_eq!(
        RoutedReply::from_recording_bytes(b"not a recording at all"),
        Err(RoutedReplyDecodeError::NotARecording)
    );
    assert_eq!(
        RoutedReply::from_recording_bytes(b"ASTROROUTES1\x01"),
        Err(RoutedReplyDecodeError::IncompleteHeader)
    );
}

#[test]
fn should_refuse_a_recording_cut_short() {
    let recording = recording_bytes(2, 2, 8, 2, &[0, 1, 2, 3, 4, 5, 6]);

    assert_eq!(
        RoutedReply::from_recording_bytes(&recording),
        Err(RoutedReplyDecodeError::WrongRouteByteCount {
            expected_bytes: 8,
            actual_bytes: 7,
        })
    );
}

#[test]
fn should_refuse_an_expert_beyond_its_layer() {
    let recording = recording_bytes(1, 2, 4, 1, &[3, 4]);

    assert_eq!(
        RoutedReply::from_recording_bytes(&recording),
        Err(RoutedReplyDecodeError::ExpertBeyondLayer {
            token_index: 0,
            layer_index: 0,
            expert_id: 4,
        })
    );
}

#[test]
fn should_refuse_layers_too_large_for_one_byte_per_expert() {
    let recording = recording_bytes(1, 2, 257, 0, &[]);

    assert_eq!(
        RoutedReply::from_recording_bytes(&recording),
        Err(RoutedReplyDecodeError::TooManyExpertsPerLayer {
            experts_per_layer: 257,
        })
    );
}

#[test]
fn should_write_routes_it_was_built_from_and_read_them_back_unchanged() {
    let token_routes = vec![
        vec![vec![0, 255], vec![7, 9]],
        vec![vec![3, 4], vec![255, 0]],
        vec![vec![1, 2], vec![5, 6]],
    ];

    let routed_reply =
        RoutedReply::from_token_routes(256, &token_routes).expect("even routes build");
    let read_back = RoutedReply::from_recording_bytes(&routed_reply.to_recording_bytes())
        .expect("its own bytes decode");

    assert_eq!(read_back, routed_reply);
    assert_eq!(read_back.token_count(), 3);
    assert_eq!(read_back.layer_count(), 2);
    assert_eq!(read_back.experts_per_token(), 2);
    assert_eq!(read_back.layer_route(1, 1), &[255, 0]);
}

#[test]
fn should_build_an_empty_recording_from_no_tokens() {
    let routed_reply = RoutedReply::from_token_routes(256, &[]).expect("no tokens build");

    assert_eq!(routed_reply.token_count(), 0);
    assert_eq!(
        RoutedReply::from_recording_bytes(&routed_reply.to_recording_bytes()),
        Ok(routed_reply)
    );
}

#[test]
fn should_refuse_a_layer_that_routes_to_a_different_number_of_experts() {
    let token_routes = vec![vec![vec![0, 1], vec![2, 3]], vec![vec![0, 1], vec![2]]];

    assert_eq!(
        RoutedReply::from_token_routes(256, &token_routes),
        Err(RoutedReplyDecodeError::UnevenRoute {
            token_index: 1,
            layer_index: 1,
            expected_expert_count: 2,
            actual_expert_count: 1,
        })
    );
}

#[test]
fn should_refuse_a_layer_missing_from_a_later_token() {
    let token_routes = vec![vec![vec![0], vec![1]], vec![vec![0]]];

    assert!(matches!(
        RoutedReply::from_token_routes(256, &token_routes),
        Err(RoutedReplyDecodeError::UnevenRoute {
            token_index: 1,
            layer_index: 1,
            ..
        })
    ));
}

#[test]
fn should_refuse_to_build_routes_naming_an_expert_beyond_the_layer() {
    let token_routes = vec![vec![vec![0, 8]]];

    assert!(matches!(
        RoutedReply::from_token_routes(8, &token_routes),
        Err(RoutedReplyDecodeError::ExpertBeyondLayer { expert_id: 8, .. })
    ));
}
