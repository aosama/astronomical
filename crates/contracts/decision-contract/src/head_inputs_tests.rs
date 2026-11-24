//! Unit tests for head inputs and their exact wire form.

use super::*;

fn kev_inputs() -> HeadInputs {
    HeadInputs::Kev {
        decision_state: FloatVector(vec![1.5, -0.0, f32::MIN_POSITIVE, 3.402_823_5e38]),
        option_states: vec![FloatVector(vec![0.1, 0.2]), FloatVector(vec![-7.25, 1e-30])],
    }
}

#[test]
fn should_round_trip_every_bit_of_every_float() {
    let wire_text = serde_json::to_string(&kev_inputs()).unwrap();
    let read_back: HeadInputs = serde_json::from_str(&wire_text).unwrap();
    let HeadInputs::Kev {
        decision_state,
        option_states,
    } = read_back
    else {
        panic!("a Kev input came back as another family");
    };
    let mut original_bits = Vec::new();
    let mut returned_bits = Vec::new();
    for float_value in [1.5_f32, -0.0, f32::MIN_POSITIVE, 3.402_823_5e38] {
        original_bits.push(float_value.to_bits());
    }
    for float_value in &decision_state.0 {
        returned_bits.push(float_value.to_bits());
    }
    assert_eq!(returned_bits, original_bits);
    assert_eq!(option_states[1].0, vec![-7.25, 1e-30]);
}

#[test]
fn should_name_the_family_and_encode_vectors_as_base64() {
    let wire_text = serde_json::to_string(&HeadInputs::Laya {
        marker_states: vec![FloatVector(vec![1.0])],
    })
    .unwrap();
    // 1.0f32 is 00 00 80 3f in little-endian bytes.
    assert_eq!(
        wire_text,
        r#"{"family":"laya","marker_states":["AACAPw=="]}"#
    );
}

#[test]
fn should_accept_an_empty_vector() {
    let read_back: FloatVector = serde_json::from_str(r#""""#).unwrap();
    assert!(read_back.0.is_empty());
}

#[test]
fn should_refuse_text_that_is_not_base64_or_not_whole_floats() {
    assert!(serde_json::from_str::<FloatVector>(r#""not base64!""#).is_err());
    // Three bytes cannot be a 32-bit float.
    assert!(serde_json::from_str::<FloatVector>(r#""AAAA""#).is_err());
    assert!(
        serde_json::from_str::<HeadInputs>(r#"{"family":"laya","marker_states":[],"extra":1}"#)
            .is_err()
    );
}

#[test]
fn should_count_options_only_when_every_clef_list_agrees() {
    let agreeing = HeadInputs::Clef {
        field_vectors: vec![FloatVector::default(); 2],
        option_vectors: vec![FloatVector::default(); 2],
        lexical_priors: vec![0.1, 0.2],
        cosines: vec![0.3, 0.4],
    };
    let disagreeing = HeadInputs::Clef {
        field_vectors: vec![FloatVector::default(); 2],
        option_vectors: vec![FloatVector::default(); 2],
        lexical_priors: vec![0.1],
        cosines: vec![0.3, 0.4],
    };
    assert_eq!(agreeing.option_count(), Some(2));
    assert_eq!(disagreeing.option_count(), None);
    assert_eq!(kev_inputs().option_count(), Some(2));
}

#[test]
fn should_omit_head_inputs_unless_asked() {
    assert_eq!(HeadInputRequest::default(), HeadInputRequest::Omit);
    assert_eq!(
        serde_json::to_string(&HeadInputRequest::Include).unwrap(),
        r#""include""#
    );
}
