//! Can the Neural Engine's share of a 4-bit checkpoint stay compressed?
//!
//! The dense A/B gave the Neural Engine fp16 copies of 4-bit weights, about
//! 6 GB extra for Ornith-1.5-9B before #376's footprint multiplier. An ML
//! Program can carry compressed weights through `constexpr_blockwise_shift_scale`
//! or `constexpr_lut_to_dense`; these journeys map which forms Core ML places
//! on the Neural Engine, what each costs in speed and fidelity, and how much
//! process memory the viable ones save.
//!
//! Every probe starts from the same 4-bit checkpoint (one scale per 64 inputs
//! of a row) and re-encodes the checkpoint's weights into the probed form, so
//! "fidelity" is how far that form sits from the checkpoint and "accuracy" is
//! how far the engine's result sits from an fp32 product of that form's own
//! weights. The footprint journeys rely on the runner giving each one its own
//! process, because loads earlier in the same process blur later deltas.

use astronomical_neural_engine::core_ml::{ComputePlacement, CoreMlComputeUnits};
use astronomical_neural_engine::documents::projection::MlProgramWeights;
use astronomical_neural_engine::program_shape::ProgramShape;
use astronomical_neural_engine::weights::affine_quantized::{
    AffineQuantizedWeights, QuantizationLayout,
};
use astronomical_neural_engine::weights::half_precision::f16_bits_slice_to_f32;
use astronomical_neural_engine_research::numerical_comparison::{
    NumericalComparison, compare_values,
};
use astronomical_test_harness::deterministic_values::{
    PROJECTION_WEIGHT_STANDARD_DEVIATION, bell_shaped_values,
};
use serial_test::serial;

use crate::support::journey_constants::{MINIMUM_COSINE_SIMILARITY, PREFILL_CHUNK_TOKEN_COUNT};
use crate::support::measured_projection::{
    ProjectionEvidence, measure_projection_with_weights, print_evidence,
};
use crate::support::projection_fixture::ProjectionFixture;

const QUANTIZED_WEIGHT_SEED: u64 = 0x5eed_0163;
// 2,048-deep parts run at the Neural Engine's full rate (contraction-split
// journeys), so every 4,096-deep projection here is cut in two.
const HIDDEN_SIZE_CONTRACTION_PARTS: usize = 2;
const CONTRACTION_PART_WIDTH: usize = HIDDEN_SIZE / HIDDEN_SIZE_CONTRACTION_PARTS;
const HIDDEN_SIZE: usize = 4096;
const MIXER_OUTPUT_CHANNELS: usize = 2048;
// Ornith-1.5-9B's fused MLP gate and up projection, its largest per layer.
const MLP_GATE_AND_UP_CHANNELS: usize = 24_576;

/// One candidate encoding of the checkpoint for the Neural Engine.
#[derive(Clone, Copy)]
struct CompressedForm {
    label: &'static str,
    is_lookup_table: bool,
    layout: QuantizationLayout,
}

const fn compressed_form(
    label: &'static str,
    is_lookup_table: bool,
    code_bits: u8,
    block_output_rows: usize,
    block_input_columns: usize,
) -> CompressedForm {
    CompressedForm {
        label,
        is_lookup_table,
        layout: QuantizationLayout {
            code_bits,
            block_output_rows,
            block_input_columns,
        },
    }
}

/// The form the footprint journey measures: the probe found shift-scale on
/// the CPU in every shape, per-row tables on a slow Neural Engine path
/// (26.8 ms against 1.9), and 4-bit tables shared by enough rows to be fast
/// too coarse for the checkpoint (cosine 0.9925). 8-bit tables shared by 16
/// rows keep the full fp16 rate within about 1% of the checkpoint.
const EIGHT_BIT_LOOKUP_PER_16_ROWS: CompressedForm = compressed_form(
    "8-bit lookup, one table per 16 rows",
    true,
    8,
    16,
    CONTRACTION_PART_WIDTH,
);

/// The probe matrix: both operations, both code widths, and blocks from the
/// checkpoint's own (1 row × 64 inputs) up to a whole contraction part.
const PROBED_FORMS: [CompressedForm; 10] = [
    compressed_form("4-bit shift-scale, checkpoint blocks", false, 4, 1, 64),
    compressed_form("4-bit lookup, checkpoint blocks", true, 4, 1, 64),
    compressed_form(
        "4-bit lookup, one table per row",
        true,
        4,
        1,
        CONTRACTION_PART_WIDTH,
    ),
    compressed_form(
        "4-bit lookup, one table per 16 rows",
        true,
        4,
        16,
        CONTRACTION_PART_WIDTH,
    ),
    compressed_form(
        "4-bit lookup, one table per part",
        true,
        4,
        MIXER_OUTPUT_CHANNELS,
        CONTRACTION_PART_WIDTH,
    ),
    compressed_form("8-bit shift-scale, checkpoint blocks", false, 8, 1, 64),
    compressed_form(
        "8-bit shift-scale, one scale per row",
        false,
        8,
        1,
        CONTRACTION_PART_WIDTH,
    ),
    compressed_form(
        "8-bit lookup, one table per row",
        true,
        8,
        1,
        CONTRACTION_PART_WIDTH,
    ),
    EIGHT_BIT_LOOKUP_PER_16_ROWS,
    compressed_form(
        "8-bit lookup, one table per part",
        true,
        8,
        MIXER_OUTPUT_CHANNELS,
        CONTRACTION_PART_WIDTH,
    ),
];

impl CompressedForm {
    fn ml_program_weights(
        self,
        quantized_weights: &AffineQuantizedWeights,
    ) -> MlProgramWeights<'_> {
        if self.is_lookup_table {
            MlProgramWeights::LookupTable(quantized_weights)
        } else {
            MlProgramWeights::BlockwiseShiftScale(quantized_weights)
        }
    }
}

/// The checkpoint's weights re-encoded into one form, with the fp32
/// reference for that form's own weights.
struct EncodedProjection {
    encoded_weights: AffineQuantizedWeights,
    fidelity_to_checkpoint: NumericalComparison,
    fixture: ProjectionFixture,
}

fn checkpoint_fp32_weights(output_channels: usize, input_channels: usize) -> Vec<f32> {
    let original_weights = bell_shaped_values(
        QUANTIZED_WEIGHT_SEED,
        output_channels * input_channels,
        PROJECTION_WEIGHT_STANDARD_DEVIATION,
    );
    let checkpoint = AffineQuantizedWeights::quantize(
        &original_weights,
        output_channels,
        input_channels,
        QuantizationLayout::CHECKPOINT_INT4,
    )
    .expect("the weights cut into checkpoint blocks");
    f16_bits_slice_to_f32(&checkpoint.dequantized_fp16())
}

fn encode_checkpoint(
    checkpoint_weights: &[f32],
    output_channels: usize,
    input_channels: usize,
    layout: QuantizationLayout,
) -> EncodedProjection {
    let shape = ProgramShape::new(input_channels, output_channels, PREFILL_CHUNK_TOKEN_COUNT)
        .expect("the projection shape is valid");
    let encoded_weights = AffineQuantizedWeights::quantize(
        checkpoint_weights,
        output_channels,
        input_channels,
        layout,
    )
    .expect("the layout tiles the weights");
    let encoded_fp16_weights = encoded_weights.dequantized_fp16();
    let fidelity_to_checkpoint = compare_values(
        checkpoint_weights,
        &f16_bits_slice_to_f32(&encoded_fp16_weights),
    )
    .expect("both weight sets have the same length");
    EncodedProjection {
        encoded_weights,
        fidelity_to_checkpoint,
        fixture: ProjectionFixture::with_fp16_weights(shape, encoded_fp16_weights),
    }
}

fn print_probe_line(
    form: CompressedForm,
    encoded: &EncodedProjection,
    evidence: &ProjectionEvidence,
) {
    eprintln!(
        "[neural-engine-journey] {:<40} | {:<12?} | {:>7.2} ms {:>6.2} TFLOP/s | accuracy cos {:.8} \
         max {:.2e} RMS | fidelity cos {:.6} max {:.2e} RMS",
        form.label,
        evidence.placement.preferred,
        evidence.timing.median.as_secs_f64() * 1000.0,
        evidence
            .timing
            .median_teraflops(encoded.fixture.shape.floating_point_operations()),
        evidence.comparison.cosine_similarity,
        evidence.comparison.maximum_error_relative_to_reference_rms,
        encoded.fidelity_to_checkpoint.cosine_similarity,
        encoded
            .fidelity_to_checkpoint
            .maximum_error_relative_to_reference_rms,
    );
}

fn assert_on_neural_engine_and_accurate(evidence: &ProjectionEvidence) {
    assert_eq!(evidence.placement.preferred, ComputePlacement::NeuralEngine);
    assert_accurate(evidence);
}

/// Whichever engine Core ML chose, the arithmetic must still be right.
fn assert_accurate(evidence: &ProjectionEvidence) {
    assert_eq!(evidence.comparison.non_finite_candidate_count, 0);
    assert!(
        evidence.comparison.cosine_similarity >= MINIMUM_COSINE_SIMILARITY,
        "cosine similarity {} is below {MINIMUM_COSINE_SIMILARITY}",
        evidence.comparison.cosine_similarity
    );
}

/// Records placement, speed, accuracy, and fidelity for every probed form.
/// Placement is printed rather than asserted, because mapping the refusals
/// is the point (the checkpoint's own 64-input blocks go to the CPU in both
/// operations); accuracy is asserted for every form on whichever engine ran
/// it.
#[test]
#[serial(neural_engine)]
#[ignore = "probes compressed weight forms on the real Neural Engine"]
fn should_map_which_compressed_weight_forms_the_neural_engine_accepts() {
    let checkpoint_weights = checkpoint_fp32_weights(MIXER_OUTPUT_CHANNELS, HIDDEN_SIZE);
    for (form_index, form) in PROBED_FORMS.iter().enumerate() {
        eprintln!(
            "[neural-engine-journey] probe {}/{}: {}",
            form_index + 1,
            PROBED_FORMS.len(),
            form.label
        );
        let encoded = encode_checkpoint(
            &checkpoint_weights,
            MIXER_OUTPUT_CHANNELS,
            HIDDEN_SIZE,
            form.layout,
        );
        let evidence = measure_projection_with_weights(
            &encoded.fixture,
            CoreMlComputeUnits::CpuAndNeuralEngine,
            HIDDEN_SIZE_CONTRACTION_PARTS,
            form.ml_program_weights(&encoded.encoded_weights),
        );
        print_probe_line(*form, &encoded, &evidence);
        assert_accurate(&evidence);
    }
}

/// The control for every probe: the same program with the checkpoint's
/// weights as fp16 separates what the ML Program format does from what
/// compression does.
#[test]
#[serial(neural_engine)]
#[ignore = "compiles an fp16 ML Program for the real Neural Engine"]
fn should_place_an_fp16_ml_program_projection_on_the_neural_engine() {
    let checkpoint_weights = checkpoint_fp32_weights(MIXER_OUTPUT_CHANNELS, HIDDEN_SIZE);
    let encoded = encode_checkpoint(
        &checkpoint_weights,
        MIXER_OUTPUT_CHANNELS,
        HIDDEN_SIZE,
        QuantizationLayout::CHECKPOINT_INT4,
    );

    let evidence = measure_projection_with_weights(
        &encoded.fixture,
        CoreMlComputeUnits::CpuAndNeuralEngine,
        HIDDEN_SIZE_CONTRACTION_PARTS,
        MlProgramWeights::Fp16(&encoded.fixture.fp16_weights),
    );

    print_evidence(
        CoreMlComputeUnits::CpuAndNeuralEngine,
        &encoded.fixture,
        &evidence,
    );
    assert_on_neural_engine_and_accurate(&evidence);
}

/// The document and weight form a footprint journey loads.
#[derive(Clone, Copy)]
enum FootprintCandidate {
    MlProgramFp16,
    MlProgramCompressed(CompressedForm),
}

impl FootprintCandidate {
    fn label(self) -> &'static str {
        match self {
            Self::MlProgramFp16 => "ML Program, fp16",
            Self::MlProgramCompressed(form) => form.label,
        }
    }

    /// The fp16 candidates re-encode into the checkpoint's own layout, which
    /// returns the checkpoint's weights unchanged, so every candidate computes
    /// with the same weights the compressed ones are compared against.
    fn layout(self) -> QuantizationLayout {
        match self {
            Self::MlProgramCompressed(form) => form.layout,
            Self::MlProgramFp16 => QuantizationLayout::CHECKPOINT_INT4,
        }
    }

    fn ml_program_weights(self, encoded: &EncodedProjection) -> MlProgramWeights<'_> {
        match self {
            Self::MlProgramFp16 => MlProgramWeights::Fp16(&encoded.fixture.fp16_weights),
            Self::MlProgramCompressed(form) => form.ml_program_weights(&encoded.encoded_weights),
        }
    }
}

fn measure_mlp_gate_and_up_footprint(candidate: FootprintCandidate) {
    eprintln!(
        "[neural-engine-journey] ── MLP gate+up 4096 → 24576, {}",
        candidate.label()
    );
    let checkpoint_weights = checkpoint_fp32_weights(MLP_GATE_AND_UP_CHANNELS, HIDDEN_SIZE);
    let encoded = encode_checkpoint(
        &checkpoint_weights,
        MLP_GATE_AND_UP_CHANNELS,
        HIDDEN_SIZE,
        candidate.layout(),
    );
    let evidence = measure_projection_with_weights(
        &encoded.fixture,
        CoreMlComputeUnits::CpuAndNeuralEngine,
        HIDDEN_SIZE_CONTRACTION_PARTS,
        candidate.ml_program_weights(&encoded),
    );
    print_evidence(
        CoreMlComputeUnits::CpuAndNeuralEngine,
        &encoded.fixture,
        &evidence,
    );
    eprintln!(
        "  weights vs checkpoint: cosine {:.6}, max abs {:.2e} of weight RMS",
        encoded.fidelity_to_checkpoint.cosine_similarity,
        encoded
            .fidelity_to_checkpoint
            .maximum_error_relative_to_reference_rms
    );
    assert_on_neural_engine_and_accurate(&evidence);
}

#[test]
#[serial(neural_engine)]
#[ignore = "loads a 100-million-weight projection on the real Neural Engine"]
fn should_measure_the_footprint_of_mlp_gate_and_up_as_ml_program_fp16() {
    measure_mlp_gate_and_up_footprint(FootprintCandidate::MlProgramFp16);
}

#[test]
#[serial(neural_engine)]
#[ignore = "loads a 100-million-weight projection on the real Neural Engine"]
fn should_measure_the_footprint_of_mlp_gate_and_up_as_ml_program_eight_bit_lookup() {
    measure_mlp_gate_and_up_footprint(FootprintCandidate::MlProgramCompressed(
        EIGHT_BIT_LOOKUP_PER_16_ROWS,
    ));
}
