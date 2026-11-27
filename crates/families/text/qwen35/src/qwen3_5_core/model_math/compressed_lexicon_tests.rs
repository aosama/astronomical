//! Contracts for reading learned representations out of a compressed
//! lexicon: the selected entries come back unpacked, exactly as unpacking
//! the whole lexicon would give them, in the order asked for.

use mlx_bridge::{MlxArray, MlxDtype};
use runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::file_serial;

use super::compressed_lexicon::CompressedLexicon;
use super::decoder_layer_weights::Qwen3_5AffineWeights;

const ENTRY_COUNT: i32 = 16;
const NUMBERS_PER_ENTRY: i32 = 128;
const BITS_PER_NUMBER: i32 = 4;
const NUMBERS_PER_GROUP: i32 = 64;

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(512 * 1024 * 1024, 512 * 1024 * 1024)
            .expect("the lexicon test memory limits are valid"),
    )
    .expect("the direct MLX runtime initializes")
}

/// A lexicon's three arrays, quantized from pseudo-random entries that
/// include values far from zero.
struct LexiconArrays {
    packed_entries: MlxArray,
    entry_scales: MlxArray,
    entry_offsets: MlxArray,
}

impl LexiconArrays {
    fn drawn(runtime: &MlxRuntime) -> Self {
        let random_key = runtime.random_key(2_701).expect("a key builds");
        let entries = runtime
            .random_normal(
                &[ENTRY_COUNT, NUMBERS_PER_ENTRY],
                MlxDtype::Float32,
                0.0,
                3.0,
                &random_key,
            )
            .expect("the entries are drawn");
        let (packed_entries, entry_scales, entry_offsets) = runtime
            .quantize_affine(&entries, NUMBERS_PER_GROUP, BITS_PER_NUMBER)
            .expect("the entries quantize");
        Self {
            packed_entries,
            entry_scales,
            entry_offsets,
        }
    }

    fn lexicon(&self) -> CompressedLexicon<'_> {
        CompressedLexicon {
            packed_entries: &self.packed_entries,
            entry_scales: &self.entry_scales,
            entry_offsets: &self.entry_offsets,
            bits_per_number: BITS_PER_NUMBER,
            numbers_per_group: NUMBERS_PER_GROUP,
        }
    }

    /// Every entry unpacked: what each selected row must equal.
    fn every_entry_unpacked(&self, runtime: &MlxRuntime) -> Vec<f32> {
        runtime
            .dequantize_affine(
                &self.packed_entries,
                &self.entry_scales,
                &self.entry_offsets,
                NUMBERS_PER_GROUP,
                BITS_PER_NUMBER,
            )
            .expect("the whole lexicon unpacks")
            .to_vec_f32()
            .expect("the unpacked lexicon reads back")
    }
}

fn entry_of(every_entry: &[f32], entry_index: usize) -> &[f32] {
    let width = NUMBERS_PER_ENTRY as usize;
    &every_entry[entry_index * width..(entry_index + 1) * width]
}

fn symbols(runtime: &MlxRuntime, symbol_ids: &[i32]) -> MlxArray {
    runtime
        .array_from_i32(symbol_ids, &[symbol_ids.len() as i32])
        .expect("the symbols upload")
}

#[test]
#[file_serial]
fn should_return_each_selected_entry_exactly_as_unpacking_the_whole_lexicon() {
    let runtime = test_runtime();
    let lexicon_arrays = LexiconArrays::drawn(&runtime);
    let every_entry = lexicon_arrays.every_entry_unpacked(&runtime);

    let representations = lexicon_arrays
        .lexicon()
        .learned_representations_of(&runtime, &symbols(&runtime, &[3, 11, 0]))
        .expect("the representations are read")
        .to_vec_f32()
        .expect("the representations read back");

    for (row_index, entry_index) in [3_usize, 11, 0].into_iter().enumerate() {
        assert_eq!(
            entry_of(&representations, row_index),
            entry_of(&every_entry, entry_index),
            "row {row_index} must be entry {entry_index}, bit for bit"
        );
    }
}

#[test]
#[file_serial]
fn should_return_one_row_for_a_single_symbol_at_the_lexicons_last_entry() {
    let runtime = test_runtime();
    let lexicon_arrays = LexiconArrays::drawn(&runtime);
    let every_entry = lexicon_arrays.every_entry_unpacked(&runtime);
    let last_entry = ENTRY_COUNT - 1;

    let representation = lexicon_arrays
        .lexicon()
        .learned_representations_of(&runtime, &symbols(&runtime, &[last_entry]))
        .expect("the representation is read");

    assert_eq!(representation.shape(), vec![1, NUMBERS_PER_ENTRY]);
    assert_eq!(
        representation.to_vec_f32().expect("it reads back"),
        entry_of(&every_entry, last_entry as usize)
    );
}

#[test]
#[file_serial]
fn should_repeat_a_symbol_asked_for_twice() {
    let runtime = test_runtime();
    let lexicon_arrays = LexiconArrays::drawn(&runtime);

    let representations = lexicon_arrays
        .lexicon()
        .learned_representations_of(&runtime, &symbols(&runtime, &[7, 7]))
        .expect("the representations are read")
        .to_vec_f32()
        .expect("the representations read back");

    assert_eq!(entry_of(&representations, 0), entry_of(&representations, 1));
}

#[test]
#[file_serial]
fn should_refuse_scales_that_do_not_cover_the_entries_groups() {
    let runtime = test_runtime();
    let lexicon_arrays = LexiconArrays::drawn(&runtime);
    let one_group_of_scales = runtime
        .slice(
            &lexicon_arrays.entry_scales,
            &[0, 0],
            &[ENTRY_COUNT, 1],
            &[1, 1],
        )
        .expect("the scales slice");
    let mismatched_lexicon = CompressedLexicon {
        entry_scales: &one_group_of_scales,
        ..lexicon_arrays.lexicon()
    };

    let outcome =
        mismatched_lexicon.learned_representations_of(&runtime, &symbols(&runtime, &[0, 1]));

    assert!(
        outcome.is_err(),
        "MLX must refuse scales for one group of two"
    );
}

#[test]
#[file_serial]
fn should_read_affine_weights_as_a_compressed_lexicon() {
    let runtime = test_runtime();
    let lexicon_arrays = LexiconArrays::drawn(&runtime);
    let every_entry = lexicon_arrays.every_entry_unpacked(&runtime);
    let affine_weights = Qwen3_5AffineWeights::Quantized {
        packed_weight: lexicon_arrays.packed_entries,
        quantization_scales: lexicon_arrays.entry_scales,
        quantization_biases: lexicon_arrays.entry_offsets,
        quantization_bits: BITS_PER_NUMBER,
        quantization_group_size: NUMBERS_PER_GROUP,
    };

    let compressed_lexicon = affine_weights
        .as_compressed_lexicon()
        .expect("affine-quantized weights read as a compressed lexicon");

    assert_eq!(compressed_lexicon.bits_per_number, BITS_PER_NUMBER);
    assert_eq!(compressed_lexicon.numbers_per_group, NUMBERS_PER_GROUP);
    let representation = compressed_lexicon
        .learned_representations_of(&runtime, &symbols(&runtime, &[5]))
        .expect("the representation is read")
        .to_vec_f32()
        .expect("it reads back");
    assert_eq!(representation, entry_of(&every_entry, 5));
}

#[test]
#[file_serial]
fn should_not_read_bfloat16_weights_as_a_compressed_lexicon() {
    let runtime = test_runtime();
    let full_precision_weights = Qwen3_5AffineWeights::NativeBfloat16 {
        weight: runtime
            .array_from_f32(&[0.0; 8], &[2, 4])
            .expect("the weights upload"),
    };

    assert!(full_precision_weights.as_compressed_lexicon().is_none());
}
