use super::{MAXIMUM_AUDIO_BYTES_AT_NATURAL_PACE, maximum_audio_bytes_for};

// Positive: at natural pace or faster, the bound is the natural one.
#[test]
fn should_keep_the_natural_bound_at_natural_pace_or_faster() {
    assert_eq!(
        maximum_audio_bytes_for(1.0),
        MAXIMUM_AUDIO_BYTES_AT_NATURAL_PACE
    );
    assert_eq!(
        maximum_audio_bytes_for(4.0),
        MAXIMUM_AUDIO_BYTES_AT_NATURAL_PACE
    );
}

// Edge: the slowest pace stretches the same text four times.
#[test]
fn should_grow_the_bound_for_the_slowest_pace() {
    assert_eq!(
        maximum_audio_bytes_for(0.25),
        MAXIMUM_AUDIO_BYTES_AT_NATURAL_PACE * 4
    );
}

// Negative: a pace the contract would refuse never raises the bound past
// the slowest pace's, and NaN keeps the natural one.
#[test]
fn should_cap_the_bound_for_a_pace_the_contract_refuses() {
    assert_eq!(
        maximum_audio_bytes_for(0.0),
        MAXIMUM_AUDIO_BYTES_AT_NATURAL_PACE * 4
    );
    assert_eq!(
        maximum_audio_bytes_for(f32::NAN),
        MAXIMUM_AUDIO_BYTES_AT_NATURAL_PACE
    );
}
