use astronomical_model_serving::{PagedDecodeLayerDisposition, QuantizedExpertPageManifest};

fn retained_page_manifest() -> QuantizedExpertPageManifest {
    QuantizedExpertPageManifest {
        expert_ids: vec![1, 3],
        page_slot_by_global_expert_id: vec![u32::MAX, 0, u32::MAX, 1, u32::MAX],
        source_manifests: Vec::new(),
        payload_byte_count: 200,
    }
}

#[test]
fn should_stream_an_entire_decode_layer_when_no_page_is_retained() {
    let decode_layer_disposition = PagedDecodeLayerDisposition::from_retained_page(None, &[1, 4]);

    assert_eq!(
        decode_layer_disposition,
        PagedDecodeLayerDisposition::StreamEntireLayer
    );
}

#[test]
fn should_split_a_decode_layer_when_the_retained_page_covers_only_some_routed_experts() {
    let retained_page_manifest = retained_page_manifest();

    let decode_layer_disposition = PagedDecodeLayerDisposition::from_retained_page(
        Some(&retained_page_manifest),
        &[3, 0, 1, 4],
    );

    match decode_layer_disposition {
        PagedDecodeLayerDisposition::SplitRetainedAndMissing(route_partition) => {
            assert_eq!(route_partition.retained_expert_ids, vec![1, 3]);
            assert_eq!(route_partition.missing_expert_ids, vec![0, 4]);
        }
        PagedDecodeLayerDisposition::StreamEntireLayer => {
            panic!("a partial retained page should split retained and missing experts");
        }
    }
}

#[test]
fn should_stream_an_entire_decode_layer_when_the_retained_page_misses_every_routed_expert() {
    let retained_page_manifest = retained_page_manifest();

    let decode_layer_disposition =
        PagedDecodeLayerDisposition::from_retained_page(Some(&retained_page_manifest), &[0, 4]);

    assert_eq!(
        decode_layer_disposition,
        PagedDecodeLayerDisposition::StreamEntireLayer
    );
}
