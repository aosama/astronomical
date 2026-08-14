//! Owned decode-layer decisions must outlive the retained-page cache borrow.
//!
//! The last test is the panic we actually hit: look up a missing page under
//! `borrow()`, copy the disposition, drop that borrow, then `borrow_mut()` to
//! record a disk load. Keeping the first borrow live across the recording
//! panics with `RefCell already borrowed`.

use std::cell::RefCell;

use astronomical_model_serving::{
    ExpertWeightPage, PagedDecodeLayerDisposition, QuantizedExpertPageManifest,
    RetainedExpertLayerCache,
};

#[derive(Debug)]
struct FakeExpertPage {
    payload_bytes: u64,
}

impl ExpertWeightPage for FakeExpertPage {
    fn resident_payload_byte_count(&self) -> u64 {
        self.payload_bytes
    }
}

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

#[test]
fn should_record_a_streamed_disk_load_after_releasing_the_empty_layer_borrow() {
    let retained_expert_layers = RefCell::new(RetainedExpertLayerCache::<FakeExpertPage>::new(1));
    let decode_layer_disposition = {
        let retained_expert_cache = retained_expert_layers.borrow();
        assert!(
            retained_expert_cache.retained_layer(0).is_none(),
            "phase-aware retention may leave a layer empty until a mandatory route read"
        );
        PagedDecodeLayerDisposition::from_retained_page(
            retained_expert_cache
                .retained_layer(0)
                .map(|_retained_expert_page| {
                    unreachable!("an empty layer has no retained page manifest")
                }),
            &[3, 7],
        )
    };

    assert_eq!(
        decode_layer_disposition,
        PagedDecodeLayerDisposition::StreamEntireLayer
    );
    retained_expert_layers.borrow_mut().record_disk_load(2, 1);
    assert_eq!(
        retained_expert_layers
            .borrow()
            .statistics()
            .disk_page_load_count,
        2
    );
}
