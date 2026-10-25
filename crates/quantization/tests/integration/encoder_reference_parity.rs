//! The pure reference encoder against the Python range search on a fixed
//! tensor: packed codes must be identical; scales and biases within float
//! rounding. The GPU encoder is held to the reference in the direct-MLX
//! contracts, so this chain pins the Metal path to the Python tool through a
//! CPU oracle that runs on every host.

use astronomical_quantization::bit_width::{BitWidth, GroupSize};
use astronomical_quantization::encoder::reference::{
    encode_slab, pack_affine_codes, unpack_affine_codes,
};

use super::fixtures;

const SCALE_TOLERANCE: f32 = 1e-6;

#[test]
#[ignore = "pending implementation of encode_slab"]
fn should_reproduce_the_python_range_search_at_two_and_three_bits() {
    let fixture = fixtures::encoder_parity();

    for bits in [2_u32, 3] {
        let case = &fixture.cases[&bits.to_string()];
        let slab = encode_slab(
            &fixture.weights_row_major,
            fixture.rows,
            fixture.columns,
            BitWidth::new(bits).expect("valid"),
            GroupSize::new(fixture.group_size).expect("valid"),
            true,
        )
        .expect("the fixture slab is group-quantizable");

        assert_eq!(
            slab.packed_words, case.search_packed_words,
            "packed codes differ at {bits} bits"
        );
        for (rust_scale, python_scale) in slab.scales.iter().zip(&case.search_scales) {
            assert!((rust_scale - python_scale).abs() <= SCALE_TOLERANCE);
        }
        for (rust_bias, python_bias) in slab.biases.iter().zip(&case.search_biases) {
            assert!((rust_bias - python_bias).abs() <= SCALE_TOLERANCE);
        }
    }
}

#[test]
#[ignore = "pending implementation of encode_slab"]
fn should_reproduce_stock_min_max_at_four_and_eight_bits_when_the_search_is_off() {
    let fixture = fixtures::encoder_parity();

    for bits in [4_u32, 8] {
        let case = &fixture.cases[&bits.to_string()];
        let slab = encode_slab(
            &fixture.weights_row_major,
            fixture.rows,
            fixture.columns,
            BitWidth::new(bits).expect("valid"),
            GroupSize::new(fixture.group_size).expect("valid"),
            false,
        )
        .expect("the fixture slab is group-quantizable");

        assert_eq!(
            slab.packed_words, case.stock_packed_words,
            "packed codes differ at {bits} bits"
        );
    }
}

#[test]
#[ignore = "pending implementation of pack_affine_codes"]
fn should_pack_codes_identically_to_mlx_for_every_fixture_width() {
    // Unpack the Python words, repack, and expect the same words: pins the
    // packing rule independently of the search.
    let fixture = fixtures::encoder_parity();

    for (bits_text, case) in &fixture.cases {
        let bits: u32 = bits_text.parse().expect("fixture bit keys are integers");
        let bit_width = BitWidth::new(bits).expect("valid");
        let codes = unpack_affine_codes(
            &case.search_packed_words,
            fixture.rows,
            fixture.columns,
            bit_width,
        );

        let repacked = pack_affine_codes(&codes, fixture.columns, bit_width);

        assert_eq!(
            repacked, case.search_packed_words,
            "repacking differs at {bits} bits"
        );
    }
}

#[test]
#[ignore = "pending implementation of a float32 KL helper in the reference module"]
fn should_reproduce_the_python_kl_divergence_on_the_fixture_logits() {
    // The GPU KL lives in the probe implementation; a CPU float32 reference
    // belongs beside the encoder reference so the formula is pinned here:
    // KL(reference ‖ probe) = mean over positions of
    // sum(softmax(ref) * (log_softmax(ref) - log_softmax(probe))).
    let fixture = fixtures::encoder_parity();
    let kl_fixture = &fixture.kl_divergence;

    let kl = astronomical_quantization::encoder::reference::kl_divergence_reference_vs_probe(
        &kl_fixture.reference_logits,
        &kl_fixture.probe_logits,
        kl_fixture.shape[2],
    );

    assert!((kl - kl_fixture.kl_reference_vs_probe).abs() < 1e-6);
}
