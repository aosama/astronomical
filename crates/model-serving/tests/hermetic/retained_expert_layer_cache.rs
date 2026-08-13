use astronomical_model_serving::{ExpertWeightPage, RetainedExpertLayerCache};

#[derive(Debug)]
struct FakeExpertLayer {
    payload_bytes: u64,
}

impl ExpertWeightPage for FakeExpertLayer {
    fn resident_payload_byte_count(&self) -> u64 {
        self.payload_bytes
    }
}

#[test]
fn should_retain_complete_layers_within_the_ram_ceiling() {
    let mut retained_layers = RetainedExpertLayerCache::new(4);
    retained_layers.update_maximum_resident_payload_bytes(300);

    assert!(retained_layers.retain_complete_layer(0, FakeExpertLayer { payload_bytes: 100 }));
    assert!(retained_layers.retain_complete_layer(1, FakeExpertLayer { payload_bytes: 100 }));
    assert!(retained_layers.retain_complete_layer(2, FakeExpertLayer { payload_bytes: 100 }));
    assert!(!retained_layers.retain_complete_layer(3, FakeExpertLayer { payload_bytes: 100 }));

    let statistics = retained_layers.statistics();
    assert_eq!(statistics.entry_count, 3);
    assert_eq!(statistics.resident_payload_byte_count, 300);
}

#[test]
fn should_reclaim_highest_layers_for_request_pressure() {
    let mut retained_layers = RetainedExpertLayerCache::new(4);
    retained_layers.update_maximum_resident_payload_bytes(400);
    for layer_index in 0..4 {
        assert!(
            retained_layers
                .retain_complete_layer(layer_index, FakeExpertLayer { payload_bytes: 100 },)
        );
    }

    assert!(retained_layers.limit_for_request_pressure(150));

    let statistics = retained_layers.statistics();
    assert_eq!(statistics.entry_count, 2);
    assert_eq!(statistics.resident_payload_byte_count, 200);
    assert_eq!(statistics.eviction_count, 2);
    assert!(retained_layers.retained_layer(0).is_some());
    assert!(retained_layers.retained_layer(1).is_some());
    assert!(retained_layers.retained_layer(2).is_none());
    assert!(retained_layers.retained_layer(3).is_none());
}

#[test]
fn should_fill_the_largest_exact_prefix_for_heterogeneous_layer_sizes() {
    let mut retained_layers = RetainedExpertLayerCache::new(4);
    retained_layers.update_maximum_resident_payload_bytes(320);

    assert!(retained_layers.retain_complete_layer(0, FakeExpertLayer { payload_bytes: 150 }));
    assert!(retained_layers.retain_complete_layer(1, FakeExpertLayer { payload_bytes: 80 }));
    assert!(retained_layers.retain_complete_layer(2, FakeExpertLayer { payload_bytes: 90 }));
    assert!(!retained_layers.retain_complete_layer(3, FakeExpertLayer { payload_bytes: 20 }));

    let statistics = retained_layers.statistics();
    assert_eq!(statistics.entry_count, 3);
    assert_eq!(statistics.resident_payload_byte_count, 320);
}

#[test]
fn should_report_whether_an_exact_layer_payload_fits_before_loading_it() {
    let mut retained_layers = RetainedExpertLayerCache::new(2);
    retained_layers.update_maximum_resident_payload_bytes(250);
    assert!(retained_layers.retain_complete_layer(0, FakeExpertLayer { payload_bytes: 150 }));

    assert!(retained_layers.can_retain_additional_payload_bytes(100));
    assert!(!retained_layers.can_retain_additional_payload_bytes(101));
}

#[test]
fn should_restore_the_normal_limit_after_temporary_request_pressure_ends() {
    let mut retained_layers = RetainedExpertLayerCache::new(4);
    retained_layers.update_maximum_resident_payload_bytes(400);
    for layer_index in 0..4 {
        assert!(
            retained_layers
                .retain_complete_layer(layer_index, FakeExpertLayer { payload_bytes: 100 },)
        );
    }

    assert!(retained_layers.limit_for_request_pressure(150));
    assert_eq!(
        retained_layers
            .statistics()
            .maximum_resident_payload_byte_count,
        250,
    );
    assert!(retained_layers.resume_after_request_pressure());

    let resumed_statistics = retained_layers.statistics();
    assert_eq!(resumed_statistics.maximum_resident_payload_byte_count, 400);
    assert!(retained_layers.can_retain_additional_payload_bytes(200));
    assert!(retained_layers.retain_complete_layer(2, FakeExpertLayer { payload_bytes: 100 },));
    assert!(retained_layers.retain_complete_layer(3, FakeExpertLayer { payload_bytes: 100 },));
}
