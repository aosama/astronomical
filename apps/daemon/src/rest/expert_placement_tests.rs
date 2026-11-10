//! Sentence contract for the shared expert placement copy.

use astronomical_ipc_protocol::{ExpertLayerSeatingCounts, ExpertMemoryMode};

use super::expert_placement::{
    ExpertPlacementCopy, MAXIMUM_STOCK_LABEL_CHARACTERS, expert_placement, expert_residency_json,
};
use crate::{ExpertResidencySnapshot, WorkerActivity};

const APPLE_GIGABYTE_BYTES: u64 = 1_073_741_824;

fn measured_residency(
    total_layer_count: u32,
    resident_expert_payload_bytes: u64,
    complete_expert_payload_byte_count: u64,
) -> ExpertResidencySnapshot {
    ExpertResidencySnapshot {
        dense_layers: None,
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

fn dense_model_residency() -> ExpertResidencySnapshot {
    ExpertResidencySnapshot {
        dense_layers: None,
        expert_capacity: 0,
        ..measured_residency(0, 0, 0)
    }
}

#[test]
fn should_call_a_dense_model_dense_instead_of_naming_experts() {
    for activity in [WorkerActivity::Idle, WorkerActivity::Generating] {
        let placement = expert_placement(
            Some("Qwen3.5-0.8B"),
            activity,
            Some(ExpertMemoryMode::Resident),
            Some(dense_model_residency()),
        );

        assert_eq!(placement.reply_sentence, "Dense model, no experts.");
        assert_eq!(placement.stock_sentence, "All in RAM");
    }
}

#[test]
fn should_keep_the_expert_sentence_for_a_sparse_model_with_expert_capacity() {
    let placement = expert_placement(
        Some("Ornith"),
        WorkerActivity::Idle,
        Some(ExpertMemoryMode::Resident),
        Some(measured_residency(0, 0, 0)),
    );

    assert_eq!(
        placement.reply_sentence,
        "Experts are being served from RAM."
    );
}

#[test]
fn should_keep_the_mode_sentence_before_any_residency_is_measured() {
    let placement = expert_placement(
        Some("not-yet-measured"),
        WorkerActivity::Idle,
        Some(ExpertMemoryMode::Resident),
        None,
    );

    assert_eq!(
        placement.reply_sentence,
        "Experts are being served from RAM."
    );
}

#[test]
fn should_say_nothing_about_a_dense_residency_left_over_with_no_model_loaded() {
    let placement = expert_placement(
        None,
        WorkerActivity::Idle,
        Some(ExpertMemoryMode::Resident),
        Some(dense_model_residency()),
    );

    assert_eq!(placement.reply_sentence, "");
}

#[test]
fn should_not_call_a_sparse_model_dense_when_only_its_expert_capacity_is_unnamed() {
    let sparse_residency_without_a_capacity = ExpertResidencySnapshot {
        dense_layers: None,
        expert_capacity: 0,
        ..measured_residency(40, 0, 2 * APPLE_GIGABYTE_BYTES)
    };

    let placement = expert_placement(
        Some("Ornith"),
        WorkerActivity::Idle,
        Some(ExpertMemoryMode::Resident),
        Some(sparse_residency_without_a_capacity),
    );

    assert_eq!(
        placement.reply_sentence,
        "Experts are being served from RAM."
    );
}

fn streaming_dense_residency(
    kept_layer_count: u32,
    streamed_layer_payload_bytes: u64,
) -> ExpertResidencySnapshot {
    ExpertResidencySnapshot {
        dense_layers: Some(astronomical_ipc_protocol::WorkerDenseLayerResidency {
            decoder_layer_count: 32,
            kept_layer_count,
            streamed_layer_payload_bytes,
        }),
        ..dense_model_residency()
    }
}

fn dense_placement(residency: ExpertResidencySnapshot) -> ExpertPlacementCopy {
    expert_placement(
        Some("Ornith-1.5-9B-MLX-6bit-vision"),
        WorkerActivity::Idle,
        Some(ExpertMemoryMode::Resident),
        Some(residency),
    )
}

#[test]
fn should_say_how_much_a_streaming_dense_model_reads_from_the_ssd() {
    let placement = dense_placement(streaming_dense_residency(7, 4_437_000_000));

    assert_eq!(placement.stock_sentence, "4.13 GB on SSD");
    assert_eq!(
        placement.reply_sentence,
        "Dense model: 7 of 32 layers in RAM, 25 read from the SSD."
    );
}

#[test]
fn should_say_every_layer_streams_when_none_is_kept() {
    let placement = dense_placement(streaming_dense_residency(0, 5_680_000_000));

    assert_eq!(placement.stock_sentence, "5.29 GB on SSD");
    assert_eq!(
        placement.reply_sentence,
        "Dense model: all 32 layers read from the SSD."
    );
}

#[test]
fn should_count_the_layers_in_ram_when_a_dense_model_keeps_them_all() {
    let placement = dense_placement(streaming_dense_residency(32, 0));

    assert_eq!(placement.stock_sentence, "32 layers in RAM");
    assert_eq!(
        placement.reply_sentence,
        "Dense model: all 32 layers in RAM."
    );
}

#[test]
fn should_keep_the_dense_label_within_the_menu_cell_for_a_large_model() {
    let placement = dense_placement(streaming_dense_residency(1, 987_000_000_000));

    assert!(placement.stock_sentence.chars().count() <= MAXIMUM_STOCK_LABEL_CHARACTERS);
}

#[test]
fn should_say_no_model_loaded_even_with_a_stale_dense_report() {
    let placement = expert_placement(
        None,
        WorkerActivity::Idle,
        Some(ExpertMemoryMode::Resident),
        Some(streaming_dense_residency(7, 4_437_000_000)),
    );

    assert_eq!(placement.stock_sentence, "No model loaded");
    assert_eq!(placement.reply_sentence, "");
}

#[test]
fn should_publish_a_dense_model_layer_counts_in_the_residency_document() {
    let residency_json = expert_residency_json(Some(streaming_dense_residency(7, 4_437_000_000)));

    assert_eq!(residency_json["dense_layers"]["decoder_layer_count"], 32);
    assert_eq!(residency_json["dense_layers"]["kept_layer_count"], 7);
    assert_eq!(
        residency_json["dense_layers"]["streamed_layer_payload_bytes"],
        4_437_000_000_u64
    );
}

#[test]
fn should_leave_dense_layers_out_of_a_sparse_residency_document() {
    let residency_json = expert_residency_json(Some(measured_residency(40, 1, 2)));

    assert!(residency_json.get("dense_layers").is_none());
}

#[test]
fn should_describe_the_dense_layers_even_before_a_memory_mode_is_reported() {
    let placement = expert_placement(
        Some("Ornith-1.5-9B-MLX-6bit-vision"),
        WorkerActivity::Idle,
        None,
        Some(streaming_dense_residency(7, 4_437_000_000)),
    );

    assert_eq!(
        placement.reply_sentence,
        "Dense model: 7 of 32 layers in RAM, 25 read from the SSD."
    );
}

#[test]
fn should_keep_the_dense_sentence_while_a_reply_runs() {
    let placement = expert_placement(
        Some("Ornith-1.5-9B-MLX-6bit-vision"),
        WorkerActivity::Generating,
        Some(ExpertMemoryMode::Resident),
        Some(streaming_dense_residency(7, 4_437_000_000)),
    );

    assert_eq!(
        placement.reply_sentence,
        "Dense model: 7 of 32 layers in RAM, 25 read from the SSD."
    );
}

#[test]
fn should_call_more_kept_layers_than_the_model_has_all_in_ram() {
    let mut residency = streaming_dense_residency(40, 0);
    if let Some(dense_layers) = residency.dense_layers.as_mut() {
        dense_layers.kept_layer_count = 40;
    }

    assert_eq!(
        dense_placement(residency).reply_sentence,
        "Dense model: all 32 layers in RAM."
    );
}

#[test]
fn should_not_count_layers_for_a_dense_report_with_no_layers() {
    let residency = ExpertResidencySnapshot {
        dense_layers: Some(astronomical_ipc_protocol::WorkerDenseLayerResidency::default()),
        ..dense_model_residency()
    };

    let placement = dense_placement(residency);

    assert_eq!(placement.stock_sentence, "All in RAM");
    assert_eq!(placement.reply_sentence, "Dense model, no experts.");
}

#[test]
fn should_keep_the_dense_layers_when_the_daemon_takes_a_worker_residency() {
    let worker_residency = astronomical_ipc_protocol::WorkerExpertResidencySnapshot {
        dense_layers: Some(astronomical_ipc_protocol::WorkerDenseLayerResidency {
            decoder_layer_count: 32,
            kept_layer_count: 7,
            streamed_layer_payload_bytes: 4_437_000_000,
        }),
        total_layer_count: 0,
        resident_expert_count: 0,
        resident_expert_payload_bytes: 0,
        complete_expert_payload_byte_count: 0,
        expert_capacity: 0,
        layer_seating: ExpertLayerSeatingCounts::default(),
    };

    let daemon_residency = ExpertResidencySnapshot::from(worker_residency);

    assert_eq!(daemon_residency.dense_layers, worker_residency.dense_layers);
}
