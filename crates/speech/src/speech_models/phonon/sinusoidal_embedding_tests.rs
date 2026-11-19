use super::sinusoidal_embedding;

#[test]
fn should_embed_zero_as_ones_then_zeros() {
    assert_eq!(
        sinusoidal_embedding(0.0, 8, 10_000.0),
        Some(vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    );
}

#[test]
fn should_match_the_reference_formula_for_a_nonzero_value() {
    let embedding = sinusoidal_embedding(2.0, 4, 10_000.0).expect("width 4 is valid");
    let expected = [2f32.cos(), 0.0002f32.cos(), 2f32.sin(), 0.0002f32.sin()];
    for (found, wanted) in embedding.iter().zip(expected) {
        assert!((found - wanted).abs() < 1e-6, "{embedding:?}");
    }
}

// Negative and boundary: widths with no defined frequencies.
#[test]
fn should_refuse_an_odd_or_too_small_width() {
    assert_eq!(sinusoidal_embedding(1.0, 5, 10_000.0), None);
    assert_eq!(sinusoidal_embedding(1.0, 2, 10_000.0), None);
}
