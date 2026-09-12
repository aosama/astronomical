//! Contract tests for the two `qwen4_exp` rules that cannot be guessed from
//! tensor shapes: the n-gram row identity and the hyper-connection stream
//! algebra.
//!
//! Expected values are pinned exactly for integer arithmetic and to `f32`
//! tolerance for float arithmetic. They were derived from the public reference
//! implementations recorded in `docs/qwen4-exp-architecture.md`, and the
//! n-gram layout closes against published artifact geometry: the sixteen
//! per-head primes sum to 320,001,446 rows, which pad by 90 to the published
//! 320,001,536 across 128 shards of 2,500,012 rows.

use astronomical_model_serving::{
    GatedResidualWeights, NgramIdentityConfiguration, NgramPlanError, NgramRowIdentity,
    StreamAlgebraError, StreamMixingPlan, average_combine, average_mix, gated_combine, gated_mix,
    grouped_rms_norm, is_prime_u64, nth_prime_after, splitmix64,
};

const EOS: u32 = 248_044;
const VOCAB: u32 = 248_320;

fn family_configuration() -> NgramIdentityConfiguration {
    NgramIdentityConfiguration {
        ngram_size: 3,
        heads_per_ngram: 8,
        unigram_vocab_size: VOCAB,
        ngram_vocab_size_base: 20_000_000,
        vocabulary_divisor: 128,
        seed: 1234,
        ple_layer_ordinal: 0,
        eos_token_id: EOS,
    }
}

#[test]
fn splitmix64_matches_the_reference_finalizer() {
    assert_eq!(splitmix64(0), 16_294_208_416_658_607_535);
    assert_eq!(splitmix64(1), 10_451_216_379_200_822_465);
    assert_eq!(splitmix64(1 << 63), 5_196_802_822_362_493_915);
    assert_eq!(splitmix64(0xDEAD_BEEF), 5_395_234_354_446_855_067);
}

#[test]
fn primality_rejects_composites_and_accepts_the_reference_primes() {
    assert!(!is_prime_u64(0));
    assert!(!is_prime_u64(1));
    assert!(is_prime_u64(2));
    assert!(!is_prime_u64(20_000_001));
    assert!(is_prime_u64(20_000_003));
    assert!(!is_prime_u64(20_000_009));
    assert!(!is_prime_u64(20_000_017));
    assert!(is_prime_u64(20_000_023));
}

#[test]
fn per_head_tables_are_the_first_sixteen_primes_after_the_base() {
    let expected: [u64; 16] = [
        20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069,
        20_000_077, 20_000_081, 20_000_093, 20_000_107, 20_000_147, 20_000_153, 20_000_159,
        20_000_161, 20_000_171,
    ];
    for (head, &size) in expected.iter().enumerate() {
        assert_eq!(nth_prime_after(20_000_000 - 1, head as u64 + 1), size);
    }
}

#[test]
fn vocabulary_layout_closes_against_published_geometry() {
    let identity = NgramRowIdentity::build(&family_configuration())
        .expect("family configuration builds a row identity");
    let layout = identity.layout();
    assert_eq!(layout.head_count(), 16);
    let sizes: Vec<u64> = (0..16).map(|head| layout.head_size(head)).collect();
    assert_eq!(
        sizes,
        vec![
            20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069,
            20_000_077, 20_000_081, 20_000_093, 20_000_107, 20_000_147, 20_000_153, 20_000_159,
            20_000_161, 20_000_171,
        ]
    );
    assert_eq!(layout.head_offset(0), 0);
    assert_eq!(layout.head_offset(1), 20_000_003);
    assert_eq!(layout.head_offset(15), 300_001_275);
    assert_eq!(layout.unpadded_row_count(), 320_001_446);
    assert_eq!(layout.padded_row_count(), 320_001_536);
    assert_eq!(layout.padded_row_count() % 128, 0);
    assert_eq!(layout.padded_row_count() / 128, 2_500_012);
}

#[test]
fn hash_multipliers_match_the_reference_derivation() {
    let identity = NgramRowIdentity::build(&family_configuration())
        .expect("family configuration builds a row identity");
    assert_eq!(
        identity.multipliers(),
        &[
            23_703_573_157_769_u64,
            20_109_073_645_365,
            8_052_911_324_071
        ]
    );
    for multiplier in identity.multipliers() {
        assert_eq!(multiplier % 2, 1, "multipliers are odd");
        assert!(
            *multiplier < (1_u64 << 63) / u64::from(VOCAB),
            "token times multiplier stays inside signed 63 bits"
        );
    }
}

#[test]
fn row_indices_match_the_pinned_reference_values() {
    let identity = NgramRowIdentity::build(&family_configuration())
        .expect("family configuration builds a row identity");
    let tokens: [u32; 7] = [11, 5, 7, 9, EOS, 13, 17];
    let rows = identity
        .row_ids(&[EOS, EOS], &tokens)
        .expect("context matches the n-gram width");
    let expected_first_token: [u64; 16] = [
        13_555_483,
        24_877_159,
        54_278_900,
        63_631_058,
        98_394_898,
        117_447_587,
        136_774_765,
        157_274_238,
        173_487_882,
        192_109_366,
        213_364_329,
        236_577_563,
        258_758_149,
        262_164_274,
        283_572_184,
        312_653_920,
    ];
    assert_eq!(rows[0], expected_first_token);
    let expected_eos_token: [u64; 16] = [
        2_701_124,
        21_500_242,
        55_309_614,
        75_581_644,
        88_972_381,
        114_376_771,
        133_365_251,
        146_996_157,
        163_351_759,
        197_139_898,
        211_909_746,
        231_573_307,
        249_579_864,
        268_644_755,
        288_568_344,
        309_949_535,
    ];
    assert_eq!(rows[4], expected_eos_token);
    let expected_last_token: [u64; 16] = [
        7_961_118,
        35_595_600,
        53_154_041,
        65_925_962,
        86_478_454,
        100_794_091,
        133_015_820,
        144_041_389,
        176_216_895,
        183_261_499,
        211_010_573,
        235_637_741,
        252_030_661,
        269_649_351,
        289_128_081,
        308_564_037,
    ];
    assert_eq!(rows[6], expected_last_token);
    for row in &rows {
        assert_eq!(row.len(), 16);
        assert!(row.iter().all(|id| *id < 320_001_446));
    }
}

#[test]
fn row_windows_never_cross_an_end_of_sequence_boundary() {
    let identity = NgramRowIdentity::build(&family_configuration())
        .expect("family configuration builds a row identity");
    let tokens: [u32; 4] = [100, EOS, 200, 300];
    let rows = identity
        .row_ids(&[EOS, EOS], &tokens)
        .expect("context matches the n-gram width");
    let expected: [&[u64]; 4] = [
        &[
            5_727_835,
            21_884_476,
            43_702_108,
            66_434_789,
            84_094_509,
            104_112_171,
            134_886_528,
            157_314_943,
            162_363_667,
            189_466_439,
            200_618_315,
            232_121_522,
            258_547_276,
            266_199_001,
            289_022_207,
            305_180_972,
        ],
        &[
            9_663_979,
            26_558_231,
            56_120_240,
            74_755_659,
            80_459_717,
            109_265_651,
            132_697_467,
            151_022_725,
            170_054_832,
            192_967_038,
            200_687_722,
            225_763_581,
            259_275_737,
            272_983_544,
            297_596_484,
            300_986_548,
        ],
        &[
            15_114_226,
            39_594_710,
            55_572_462,
            70_127_295,
            80_775_559,
            118_455_748,
            125_723_378,
            143_475_569,
            177_805_529,
            191_773_675,
            200_937_106,
            235_341_796,
            255_204_161,
            276_293_028,
            290_261_926,
            302_150_093,
        ],
        &[
            4_401_500,
            35_569_074,
            54_889_021,
            74_121_415,
            88_777_613,
            107_793_429,
            137_064_387,
            157_487_094,
            164_417_016,
            186_620_958,
            218_728_649,
            227_273_627,
            251_258_918,
            276_471_191,
            285_147_944,
            310_576_508,
        ],
    ];
    for (row, expected_row) in rows.iter().zip(expected) {
        assert_eq!(row, expected_row);
    }
}

#[test]
fn row_identity_rejects_an_unusable_configuration() {
    let base = family_configuration();
    let cases = [
        NgramIdentityConfiguration {
            ngram_size: 1,
            ..base.clone()
        },
        NgramIdentityConfiguration {
            heads_per_ngram: 0,
            ..base.clone()
        },
        NgramIdentityConfiguration {
            unigram_vocab_size: 0,
            ..base.clone()
        },
        NgramIdentityConfiguration {
            ngram_vocab_size_base: 1,
            ..base.clone()
        },
        NgramIdentityConfiguration {
            vocabulary_divisor: 0,
            ..base.clone()
        },
    ];
    let expected = [
        NgramPlanError::NgramSizeTooSmall { provided: 1 },
        NgramPlanError::ZeroHeadsPerNgram,
        NgramPlanError::ZeroUnigramVocabulary,
        NgramPlanError::VocabBaseTooSmall { provided: 1 },
        NgramPlanError::ZeroVocabularyDivisor,
    ];
    for (configuration, expected_error) in cases.into_iter().zip(expected) {
        assert_eq!(
            NgramRowIdentity::build(&configuration).expect_err("must reject"),
            expected_error
        );
    }
}

#[test]
fn row_identity_rejects_a_mismatched_context_length() {
    let identity = NgramRowIdentity::build(&family_configuration())
        .expect("family configuration builds a row identity");
    let error = identity
        .row_ids(&[EOS], &[11, 5])
        .expect_err("one context token cannot feed a trigram window");
    assert_eq!(
        error,
        astronomical_model_serving::NgramIdentityError::ContextLengthMismatch {
            expected: 2,
            provided: 1,
        }
    );
}

fn mixing_plan() -> StreamMixingPlan {
    StreamMixingPlan {
        stream_count: 2,
        stream_width: 4,
        low_rank: 3,
        rms_norm_epsilon: 1.0e-6,
    }
}

fn mixing_inputs() -> (Vec<f32>, GatedResidualWeights<'static>) {
    let hyper_input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let norm_weights = vec![0.1, -0.2, 0.3, 0.0, 0.5, -0.5, 0.25, 0.75];
    let down = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let up = vec![
        1.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, //
        0.0, 0.0, 1.0, //
        1.0, 1.0, 0.0, //
        0.0, 0.0, 1.0, //
        1.0, 0.0, 1.0, //
        0.0, 1.0, 1.0, //
        1.0, 1.0, 1.0,
    ];
    let inject = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    ];
    // Leaking is acceptable in a test: the weights must outlive the borrowed
    // struct, and the suite runs once per process.
    let owned = Box::leak(Box::new((down, up, inject)));
    let weights = GatedResidualWeights {
        norm: Box::leak(norm_weights.into_boxed_slice()),
        down: &owned.0,
        up: &owned.1,
        inject: &owned.2,
    };
    (hyper_input, weights)
}

fn assert_close(actual: &[f32], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len(), "length must match");
    for (index, (value, expected_value)) in actual.iter().zip(expected).enumerate() {
        let difference = (*value as f64 - expected_value).abs();
        assert!(
            difference <= tolerance,
            "element {index}: {value} vs {expected_value} (difference {difference})"
        );
    }
}

#[test]
fn grouped_norm_normalizes_each_stream_with_its_own_variance() {
    let (hyper_input, weights) = mixing_inputs();
    let normalized = grouped_rms_norm(&mixing_plan(), weights.norm, &hyper_input)
        .expect("lengths match the plan");
    assert_close(
        &normalized,
        &[
            0.401_663_182,
            0.584_237_356,
            1.424_078_555,
            1.460_593_389,
            1.137_147_052,
            0.454_858_821,
            1.326_671_561,
            2.122_674_498,
        ],
        1.0e-5,
    );
}

#[test]
fn gated_mix_matches_the_pinned_reference_values() {
    let (hyper_input, weights) = mixing_inputs();
    let output = gated_mix(&mixing_plan(), &weights, &hyper_input).expect("lengths match");
    assert_close(
        &output.mixed_input,
        &[0.456_878_934, 0.304_467_565, 0.874_527_157, 1.137_607_931],
        1.0e-5,
    );
    assert_close(
        &output.normalized,
        &[
            0.401_663_182,
            0.584_237_356,
            1.424_078_555,
            1.460_593_389,
            1.137_147_052,
            0.454_858_821,
            1.326_671_561,
            2.122_674_498,
        ],
        1.0e-5,
    );
}

#[test]
fn gated_combine_injects_into_the_raw_residual() {
    let (hyper_input, weights) = mixing_inputs();
    let output = gated_mix(&mixing_plan(), &weights, &hyper_input).expect("lengths match");
    let block_output = vec![0.5, -0.5, 1.0, 2.0];
    let combined = gated_combine(
        &mixing_plan(),
        &weights,
        &block_output,
        &hyper_input,
        &output.normalized,
    )
    .expect("lengths match");
    assert_close(
        &combined,
        &[
            1.550_039_821,
            1.449_960_179,
            4.100_079_643,
            6.200_159_285,
            5.722_210_790,
            5.277_789_210,
            8.444_421_579,
            10.888_843_158,
        ],
        1.0e-5,
    );
}

#[test]
fn average_variant_pools_and_broadcasts() {
    let (hyper_input, _) = mixing_inputs();
    let mixed = average_mix(&mixing_plan(), &hyper_input).expect("lengths match");
    assert_close(&mixed, &[3.0, 4.0, 5.0, 6.0], 1.0e-6);
    let combined = average_combine(&mixing_plan(), &[0.5, -0.5, 1.0, 2.0], &hyper_input)
        .expect("lengths match");
    assert_close(
        &combined,
        &[1.5, 1.5, 4.0, 6.0, 5.5, 5.5, 8.0, 10.0],
        1.0e-6,
    );
}

#[test]
fn stream_algebra_rejects_degenerate_plans_and_mismatched_lengths() {
    let (hyper_input, weights) = mixing_inputs();
    let plan = mixing_plan();
    assert_eq!(
        grouped_rms_norm(
            &StreamMixingPlan {
                stream_count: 0,
                ..plan
            },
            weights.norm,
            &hyper_input
        )
        .expect_err("zero streams must fail"),
        StreamAlgebraError::ZeroStreamCount
    );
    assert_eq!(
        grouped_rms_norm(&plan, &weights.norm[..7], &hyper_input)
            .expect_err("short norm weights must fail"),
        StreamAlgebraError::NormWeightLengthMismatch {
            expected: 8,
            provided: 7,
        }
    );
    assert_eq!(
        gated_mix(&plan, &weights, &hyper_input[..7]).expect_err("short hyper input must fail"),
        StreamAlgebraError::HyperInputLengthMismatch {
            expected: 8,
            provided: 7,
        }
    );
    let output = gated_mix(&plan, &weights, &hyper_input).expect("lengths match");
    assert_eq!(
        gated_combine(&plan, &weights, &[0.5], &hyper_input, &output.normalized)
            .expect_err("short block output must fail"),
        StreamAlgebraError::BlockOutputLengthMismatch {
            expected: 4,
            provided: 1,
        }
    );
}
