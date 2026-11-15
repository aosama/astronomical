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
