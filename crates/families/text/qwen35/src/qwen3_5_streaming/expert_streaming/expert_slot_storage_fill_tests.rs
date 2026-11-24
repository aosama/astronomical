//! Direct-MLX contracts of the slot filler: an expert read from storage
//! lands in exactly its slot of every table array, from a file that stores
//! an expert's slices together and from one that stores them apart, and a
//! read that cannot be made whole is reported.

use std::io::Write;
use std::path::Path;

use runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::file_serial;

use super::ExpertSlotStorageFiller;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AffineWeights;
use crate::qwen3_5_streaming::expert_streaming::expert_pager::Qwen3_5PagedExpertWeights;
use engine_core::expert_paging::{
    QuantizationMode, QuantizedExpertLayerPlan, QuantizedTensorSource, SafetensorsDtype,
};

pub(super) const EXPERT_COUNT: usize = 4;
pub(super) const TABLE_SLOT_COUNT: usize = 3;
/// uint32 words one expert holds in a weight, and in its scales or biases.
const WEIGHT_WORDS: usize = 2;
const SCALE_WORDS: usize = 1;
pub(super) const TENSOR_NAMES: [(&str, &str, usize); 9] = [
    ("gate_proj", "weight", WEIGHT_WORDS),
    ("gate_proj", "scales", SCALE_WORDS),
    ("gate_proj", "biases", SCALE_WORDS),
    ("up_proj", "weight", WEIGHT_WORDS),
    ("up_proj", "scales", SCALE_WORDS),
    ("up_proj", "biases", SCALE_WORDS),
    ("down_proj", "weight", WEIGHT_WORDS),
    ("down_proj", "scales", SCALE_WORDS),
    ("down_proj", "biases", SCALE_WORDS),
];
const RECORD_BYTES: usize = 3 * (WEIGHT_WORDS + 2 * SCALE_WORDS) * 4;

pub(super) fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(512 * 1024 * 1024, 512 * 1024 * 1024)
            .expect("the test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

/// The word every element of one tensor of one expert holds, so a slot read
/// back names the tensor and the expert it came from.
pub(super) fn tagged_word(tensor_index: usize, expert_id: usize) -> u32 {
    (0xA000 + tensor_index * 0x100 + expert_id) as u32
}

/// Where each tensor's slice of expert 0 starts and how far apart
/// consecutive experts' slices are, for the two file layouts.
#[derive(Clone, Copy)]
pub(super) enum FileLayout {
    /// An expert's nine slices back to back, one record per expert.
    RecordsTogether,
    /// Each tensor's experts back to back, tensor after tensor.
    TensorsApart,
}

fn tensor_placement(layout: FileLayout, tensor_index: usize) -> (u64, usize) {
    let mut bytes_before_tensor = 0_usize;
    for (_, _, word_count) in &TENSOR_NAMES[..tensor_index] {
        bytes_before_tensor += word_count * 4;
    }
    match layout {
        FileLayout::RecordsTogether => (bytes_before_tensor as u64, RECORD_BYTES),
        FileLayout::TensorsApart => (
            (bytes_before_tensor * EXPERT_COUNT) as u64,
            TENSOR_NAMES[tensor_index].2 * 4,
        ),
    }
}

pub(super) fn write_expert_file(layout: FileLayout) -> tempfile::NamedTempFile {
    let mut file_bytes = vec![0_u8; RECORD_BYTES * EXPERT_COUNT];
    for (tensor_index, (_, _, word_count)) in TENSOR_NAMES.iter().enumerate() {
        let (first_expert_offset, expert_stride_bytes) = tensor_placement(layout, tensor_index);
        for expert_id in 0..EXPERT_COUNT {
            let slice_start = first_expert_offset as usize + expert_id * expert_stride_bytes;
            for word_index in 0..*word_count {
                let word_start = slice_start + word_index * 4;
                file_bytes[word_start..word_start + 4]
                    .copy_from_slice(&tagged_word(tensor_index, expert_id).to_le_bytes());
            }
        }
    }
    let mut file = tempfile::NamedTempFile::new().expect("a scratch expert file");
    file.write_all(&file_bytes).expect("the expert file writes");
    file.flush().expect("the expert file flushes");
    file
}

pub(super) fn layer_plan(layout: FileLayout, expert_file: &Path) -> QuantizedExpertLayerPlan {
    let mut tensor_sources = Vec::with_capacity(TENSOR_NAMES.len());
    for (tensor_index, (projection_name, parameter_name, word_count)) in
        TENSOR_NAMES.iter().enumerate()
    {
        let (tensor_payload_offset, expert_stride_bytes) = tensor_placement(layout, tensor_index);
        tensor_sources.push(QuantizedTensorSource {
            tensor_name: format!("layers.0.mlp.switch_mlp.{projection_name}.{parameter_name}"),
            projection_name: (*projection_name).to_owned(),
            parameter_name: (*parameter_name).to_owned(),
            quantization_bits: 4,
            quantization_group_size: 64,
            source_file: expert_file.to_path_buf(),
            source_file_size_bytes: (RECORD_BYTES * EXPERT_COUNT) as u64,
            dtype: SafetensorsDtype::Uint32,
            full_shape: vec![EXPERT_COUNT, *word_count],
            tensor_payload_offset,
            bytes_per_expert: word_count * 4,
            expert_stride_bytes,
            expert_capacity: EXPERT_COUNT,
        });
    }
    QuantizedExpertLayerPlan {
        layer_prefix: "layers.0.mlp".to_owned(),
        tensor_sources,
        jangtq2_projection_rotations: Vec::new(),
        expert_capacity: EXPERT_COUNT,
        quantization_bits: 4,
        quantization_group_size: 64,
        quantization_mode: QuantizationMode::Affine,
    }
}

fn zeroed_words(runtime: &MlxRuntime, word_count: usize) -> mlx_bridge::MlxArray {
    runtime
        .array_from_u32(
            &vec![0_u32; TABLE_SLOT_COUNT * word_count],
            &[TABLE_SLOT_COUNT as i32, word_count as i32],
        )
        .expect("the table array builds")
}

fn zeroed_projection(runtime: &MlxRuntime) -> Qwen3_5AffineWeights {
    Qwen3_5AffineWeights::Quantized {
        packed_weight: zeroed_words(runtime, WEIGHT_WORDS),
        quantization_scales: zeroed_words(runtime, SCALE_WORDS),
        quantization_biases: zeroed_words(runtime, SCALE_WORDS),
        quantization_bits: 4,
        quantization_group_size: 64,
    }
}

fn zeroed_table(runtime: &MlxRuntime) -> Qwen3_5PagedExpertWeights {
    let no_rotation = jangtq2_qwen35::configuration::Jangtq2CodebookRotation::NoRotation;
    Qwen3_5PagedExpertWeights {
        gate_projection: zeroed_projection(runtime),
        up_projection: zeroed_projection(runtime),
        down_projection: zeroed_projection(runtime),
        gate_activation_rotation: no_rotation,
        up_activation_rotation: no_rotation,
        down_activation_rotation: no_rotation,
    }
}

/// Every table array's words, in `TENSOR_NAMES` order.
pub(super) fn table_words(table: &Qwen3_5PagedExpertWeights) -> Vec<Vec<u32>> {
    let mut words_by_tensor = Vec::with_capacity(TENSOR_NAMES.len());
    for projection in [
        &table.gate_projection,
        &table.up_projection,
        &table.down_projection,
    ] {
        let Qwen3_5AffineWeights::Quantized {
            packed_weight,
            quantization_scales,
            quantization_biases,
            ..
        } = projection
        else {
            panic!("the test table is affine-quantized");
        };
        for table_array in [packed_weight, quantization_scales, quantization_biases] {
            words_by_tensor.push(
                table_array
                    .to_vec_u32()
                    .expect("the table array reads back"),
            );
        }
    }
    words_by_tensor
}

/// Fills slots 0 and 2 with experts 3 and 1 and checks every array.
fn assert_experts_land_in_their_slots(layout: FileLayout) {
    let runtime = test_runtime();
    let expert_file = write_expert_file(layout);
    let plan = layer_plan(layout, expert_file.path());
    let filler = ExpertSlotStorageFiller::open(std::slice::from_ref(&plan))
        .expect("the filler opens the expert file");
    let mut table = zeroed_table(&runtime);
    let read_count = filler
        .fill_slots(&plan, &mut table, &[(3, 0), (1, 2)])
        .expect("the slots fill");
    // Together, an expert is one read; apart, one read for each of its nine
    // slices (experts 1 and 3 are not neighbours, so none join).
    let expected_read_count = match layout {
        FileLayout::RecordsTogether => 2,
        FileLayout::TensorsApart => 18,
    };
    assert_eq!(read_count, expected_read_count);
    for (tensor_index, tensor_words) in table_words(&table).iter().enumerate() {
        let word_count = TENSOR_NAMES[tensor_index].2;
        let mut expected_words = vec![tagged_word(tensor_index, 3); word_count];
        expected_words.extend(vec![0_u32; word_count]);
        expected_words.extend(vec![tagged_word(tensor_index, 1); word_count]);
        assert_eq!(
            tensor_words, &expected_words,
            "tensor {tensor_index}: slot 0 holds expert 3, slot 1 is untouched, slot 2 holds expert 1"
        );
    }
}

#[test]
#[file_serial]
fn should_read_each_expert_into_its_slot_when_its_slices_are_stored_together() {
    assert_experts_land_in_their_slots(FileLayout::RecordsTogether);
}

#[test]
#[file_serial]
fn should_read_each_expert_into_its_slot_when_its_slices_are_stored_apart() {
    assert_experts_land_in_their_slots(FileLayout::TensorsApart);
}

#[test]
#[file_serial]
fn should_change_nothing_when_no_slot_is_asked_for() {
    let runtime = test_runtime();
    let expert_file = write_expert_file(FileLayout::RecordsTogether);
    let plan = layer_plan(FileLayout::RecordsTogether, expert_file.path());
    let filler = ExpertSlotStorageFiller::open(std::slice::from_ref(&plan))
        .expect("the filler opens the expert file");
    let mut table = zeroed_table(&runtime);
    let read_count = filler
        .fill_slots(&plan, &mut table, &[])
        .expect("an empty fill succeeds");
    assert_eq!(read_count, 0);
    for tensor_words in table_words(&table) {
        assert!(tensor_words.iter().all(|word| *word == 0));
    }
}

#[test]
#[file_serial]
fn should_report_a_slot_the_table_does_not_have() {
    let runtime = test_runtime();
    let expert_file = write_expert_file(FileLayout::RecordsTogether);
    let plan = layer_plan(FileLayout::RecordsTogether, expert_file.path());
    let filler = ExpertSlotStorageFiller::open(std::slice::from_ref(&plan))
        .expect("the filler opens the expert file");
    let mut table = zeroed_table(&runtime);
    assert!(
        filler
            .fill_slots(&plan, &mut table, &[(0, TABLE_SLOT_COUNT)])
            .is_err()
    );
}

#[test]
#[file_serial]
fn should_report_one_slot_planned_for_two_experts() {
    let runtime = test_runtime();
    let expert_file = write_expert_file(FileLayout::RecordsTogether);
    let plan = layer_plan(FileLayout::RecordsTogether, expert_file.path());
    let filler = ExpertSlotStorageFiller::open(std::slice::from_ref(&plan))
        .expect("the filler opens the expert file");
    let mut table = zeroed_table(&runtime);
    assert!(
        filler
            .fill_slots(&plan, &mut table, &[(0, 1), (2, 1)])
            .is_err()
    );
}

#[test]
#[file_serial]
fn should_report_an_expert_the_file_does_not_hold() {
    let runtime = test_runtime();
    let expert_file = write_expert_file(FileLayout::RecordsTogether);
    let plan = layer_plan(FileLayout::RecordsTogether, expert_file.path());
    let filler = ExpertSlotStorageFiller::open(std::slice::from_ref(&plan))
        .expect("the filler opens the expert file");
    let mut table = zeroed_table(&runtime);
    // Expert 9 would start past the end of a four-expert file.
    assert!(filler.fill_slots(&plan, &mut table, &[(9, 0)]).is_err());
}

#[test]
#[file_serial]
fn should_refuse_a_table_that_is_not_stored_affine_quantized() {
    let runtime = test_runtime();
    let expert_file = write_expert_file(FileLayout::RecordsTogether);
    let plan = layer_plan(FileLayout::RecordsTogether, expert_file.path());
    let filler = ExpertSlotStorageFiller::open(std::slice::from_ref(&plan))
        .expect("the filler opens the expert file");
    let mut table = zeroed_table(&runtime);
    assert!(ExpertSlotStorageFiller::can_fill(&table));
    table.up_projection = Qwen3_5AffineWeights::NativeBfloat16 {
        weight: runtime
            .array_from_u32(&[0_u32; 6], &[TABLE_SLOT_COUNT as i32, 2])
            .expect("the native weight builds"),
    };
    assert!(!ExpertSlotStorageFiller::can_fill(&table));
    assert!(filler.fill_slots(&plan, &mut table, &[(0, 0)]).is_err());
}

#[test]
fn should_report_an_expert_file_that_cannot_be_opened() {
    let missing_plan = layer_plan(
        FileLayout::RecordsTogether,
        Path::new("/nonexistent/astronomical/experts.records"),
    );
    assert!(ExpertSlotStorageFiller::open(std::slice::from_ref(&missing_plan)).is_err());
}
