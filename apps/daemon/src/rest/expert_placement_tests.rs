//! Sentence contract for the shared expert placement copy.

use astronomical_ipc_protocol::{ExpertLayerSeatingCounts, ExpertMemoryMode};

use super::expert_placement::{
    MAXIMUM_STOCK_LABEL_CHARACTERS, expert_placement, expert_residency_json,
};
use crate::{ExpertResidencySnapshot, WorkerActivity};

const APPLE_GIGABYTE_BYTES: u64 = 1_073_741_824;

fn measured_residency(
    total_layer_count: u32,
    resident_expert_payload_bytes: u64,
    complete_expert_payload_byte_count: u64,
) -> ExpertResidencySnapshot {
    ExpertResidencySnapshot {
        total_layer_count,
        resident_expert_count: 0,
        resident_expert_payload_bytes,
        complete_expert_payload_byte_count,
        expert_capacity: 256,
        layer_seating: ExpertLayerSeatingCounts::default(),
    }
}

#[test]
fn should_say_no_model_is_loaded_when_no_model_is_ready() {
    let placement = expert_placement(
        None,
        WorkerActivity::Idle,
        Some(ExpertMemoryMode::Hybrid),
        Some(measured_residency(
            40,
            APPLE_GIGABYTE_BYTES,
            APPLE_GIGABYTE_BYTES,
        )),
    );
    assert_eq!(placement.stock_sentence, "No model loaded");
    assert_eq!(placement.reply_sentence, "");

    let blank_model_id = expert_placement(
        Some(""),
        WorkerActivity::Generating,
        Some(ExpertMemoryMode::Paged),
        None,
    );
    assert_eq!(blank_model_id.stock_sentence, "No model loaded");
    assert_eq!(blank_model_id.reply_sentence, "");
}

#[test]
fn should_say_the_library_size_is_unmeasured_for_hybrid_and_paged() {
    let hybrid_placement = expert_placement(
        Some("Ornith"),
        WorkerActivity::Idle,
        Some(ExpertMemoryMode::Hybrid),
        None,
    );
    assert_eq!(hybrid_placement.stock_sentence, "Size not measured");
    let paged_placement = expert_placement(
        Some("Ornith"),
        WorkerActivity::Idle,
        Some(ExpertMemoryMode::Paged),
        Some(measured_residency(40, 0, 0)),
    );
    assert_eq!(paged_placement.stock_sentence, "Size not measured");
}

#[test]
fn should_say_all_in_ram_when_the_loaded_model_is_not_sparse() {
    let image_placement = expert_placement(
        Some("fictional/flux-model"),
        WorkerActivity::ImageGeneration,
        None,
        None,
    );
    assert_eq!(image_placement.stock_sentence, "All in RAM");
    assert_eq!(image_placement.reply_sentence, "");

    let resident_without_a_measurement = expert_placement(
        Some("dense"),
        WorkerActivity::Idle,
        Some(ExpertMemoryMode::Resident),
        Some(measured_residency(0, 0, 0)),
    );
    assert_eq!(resident_without_a_measurement.stock_sentence, "All in RAM");
}

#[test]
fn should_say_every_measured_layer_is_in_ram() {
    let placement = expert_placement(
        Some("Ornith"),
        WorkerActivity::Idle,
        Some(ExpertMemoryMode::Hybrid),
        Some(measured_residency(
            40,
            2 * APPLE_GIGABYTE_BYTES,
            2 * APPLE_GIGABYTE_BYTES,
        )),
    );
    assert_eq!(placement.stock_sentence, "40 layers in RAM");
}

#[test]
fn should_say_all_in_ram_when_the_layer_count_is_zero() {
    let placement = expert_placement(
        Some("Ornith"),
        WorkerActivity::Idle,
        Some(ExpertMemoryMode::Resident),
        Some(measured_residency(
            0,
            APPLE_GIGABYTE_BYTES,
            APPLE_GIGABYTE_BYTES,
        )),
    );
    assert_eq!(placement.stock_sentence, "All in RAM");
}

#[test]
fn should_say_all_on_ssd_when_nothing_is_resident() {
    let placement = expert_placement(
        Some("Ornith"),
        WorkerActivity::Idle,
        Some(ExpertMemoryMode::Paged),
        Some(measured_residency(40, 0, 4 * APPLE_GIGABYTE_BYTES)),
    );
    assert_eq!(placement.stock_sentence, "All on SSD");
}

#[test]
fn should_name_the_gigabytes_still_on_the_ssd() {
    let placement = expert_placement(
        Some("Ornith"),
        WorkerActivity::Idle,
        Some(ExpertMemoryMode::Hybrid),
        Some(measured_residency(
            40,
            APPLE_GIGABYTE_BYTES,
            3 * APPLE_GIGABYTE_BYTES,
        )),
    );
    assert_eq!(placement.stock_sentence, "2.00 GB on SSD");
}

#[test]
fn should_name_seated_bytes_while_a_hybrid_reply_still_remembers_the_ssd() {
    let placement = expert_placement(
        Some("Ornith"),
        WorkerActivity::Generating,
        Some(ExpertMemoryMode::Hybrid),
        Some(measured_residency(
            40,
            2 * APPLE_GIGABYTE_BYTES,
            2 * APPLE_GIGABYTE_BYTES,
        )),
    );
    assert_eq!(placement.stock_sentence, "40 layers in RAM");
    assert_eq!(
        placement.reply_sentence,
        "This reply has read experts from the SSD."
    );
}

#[test]
fn should_say_this_reply_is_reading_from_ram_while_the_reply_runs() {
    let placement = expert_placement(
        Some("Ornith"),
        WorkerActivity::PromptProcessing,
        Some(ExpertMemoryMode::Resident),
        Some(measured_residency(
            40,
            APPLE_GIGABYTE_BYTES,
            APPLE_GIGABYTE_BYTES,
        )),
    );
    assert_eq!(
        placement.reply_sentence,
        "This reply is reading experts from RAM."
    );
    let preparing = expert_placement(
        Some("Ornith"),
        WorkerActivity::GenerationPreparation,
        Some(ExpertMemoryMode::Resident),
        None,
    );
    assert_eq!(
        preparing.reply_sentence,
        "This reply is reading experts from RAM."
    );
}

#[test]
fn should_say_this_reply_is_streaming_while_paged_generation_runs() {
    let placement = expert_placement(
        Some("Ornith"),
        WorkerActivity::Generating,
        Some(ExpertMemoryMode::Paged),
        None,
    );
    assert_eq!(placement.stock_sentence, "Size not measured");
    assert_eq!(
        placement.reply_sentence,
        "This reply is streaming experts from the SSD."
    );
}

#[test]
fn should_describe_the_last_reply_when_hybrid_is_idle() {
    let placement = expert_placement(
        Some("Ornith"),
        WorkerActivity::Idle,
        Some(ExpertMemoryMode::Hybrid),
        Some(measured_residency(
            40,
            APPLE_GIGABYTE_BYTES,
            APPLE_GIGABYTE_BYTES,
        )),
    );
    assert_eq!(
        placement.reply_sentence,
        "The last reply read experts from the SSD."
    );
}

#[test]
fn should_say_experts_stream_when_needed_when_paging_is_idle() {
    let placement = expert_placement(
        Some("Ornith"),
        WorkerActivity::Idle,
        Some(ExpertMemoryMode::Paged),
        None,
    );
    assert_eq!(
        placement.reply_sentence,
        "Experts stream from the SSD when a reply needs them."
    );
}

#[test]
fn should_keep_a_resident_idle_library_on_the_ram_sentence() {
    let placement = expert_placement(
        Some("Ornith"),
        WorkerActivity::Idle,
        Some(ExpertMemoryMode::Resident),
        Some(measured_residency(
            40,
            APPLE_GIGABYTE_BYTES,
            APPLE_GIGABYTE_BYTES,
        )),
    );
    assert_eq!(
        placement.reply_sentence,
        "Experts are being served from RAM."
    );
}

#[test]
fn should_not_call_image_generation_this_reply() {
    let placement = expert_placement(
        Some("fictional/flux-model"),
        WorkerActivity::ImageGeneration,
        Some(ExpertMemoryMode::Hybrid),
        None,
    );
    assert_eq!(
        placement.reply_sentence,
        "The last reply read experts from the SSD."
    );
    assert!(!placement.reply_sentence.contains("This reply"));
}

#[test]
fn should_leave_the_reply_sentence_empty_when_the_mode_is_absent() {
    let placement = expert_placement(Some("dense"), WorkerActivity::Generating, None, None);
    assert_eq!(placement.reply_sentence, "");
    assert_eq!(placement.stock_sentence, "All in RAM");
}

#[test]
fn should_omit_retained_counts_when_no_layer_has_been_counted() {
    let residency_json = expert_residency_json(Some(measured_residency(40, 1, 2)));
    assert!(residency_json.get("retained_expert_counts").is_none());
    assert!(expert_residency_json(None).is_null());
}

#[test]
fn should_publish_only_the_seated_layer_prefix() {
    let mut expert_residency = measured_residency(2, 1, 2);
    expert_residency.layer_seating = ExpertLayerSeatingCounts::from_retained_counts(&[0, 256]);
    let residency_json = expert_residency_json(Some(expert_residency));
    assert_eq!(
        residency_json["retained_expert_counts"],
        serde_json::json!([0, 256])
    );
}

fn stock_label(
    ready_model_id: Option<&str>,
    expert_memory_mode: Option<ExpertMemoryMode>,
    expert_residency: Option<ExpertResidencySnapshot>,
) -> String {
    expert_placement(
        ready_model_id,
        WorkerActivity::Idle,
        expert_memory_mode,
        expert_residency,
    )
    .stock_sentence
}

#[test]
fn should_keep_every_stock_label_on_one_line_of_the_menu_cell() {
    let hundreds_of_gigabytes = 999 * APPLE_GIGABYTE_BYTES;
    let widest_labels = [
        stock_label(None, Some(ExpertMemoryMode::Hybrid), None),
        stock_label(Some("qwen"), Some(ExpertMemoryMode::Paged), None),
        stock_label(Some("qwen"), Some(ExpertMemoryMode::Resident), None),
        stock_label(
            Some("qwen"),
            Some(ExpertMemoryMode::Hybrid),
            Some(measured_residency(
                999,
                hundreds_of_gigabytes,
                hundreds_of_gigabytes,
            )),
        ),
        stock_label(
            Some("qwen"),
            Some(ExpertMemoryMode::Hybrid),
            Some(measured_residency(
                0,
                hundreds_of_gigabytes,
                hundreds_of_gigabytes,
            )),
        ),
        stock_label(
            Some("qwen"),
            Some(ExpertMemoryMode::Hybrid),
            Some(measured_residency(999, 0, hundreds_of_gigabytes)),
        ),
        stock_label(
            Some("qwen"),
            Some(ExpertMemoryMode::Hybrid),
            Some(measured_residency(999, 1, hundreds_of_gigabytes)),
        ),
    ];
    for widest_label in widest_labels {
        assert!(
            widest_label.chars().count() <= MAXIMUM_STOCK_LABEL_CHARACTERS,
            "{widest_label:?} is too long for one line"
        );
    }
}
