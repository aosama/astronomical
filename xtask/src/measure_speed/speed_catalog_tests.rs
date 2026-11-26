use std::collections::BTreeMap;

use super::{MeasurementPlace, catalog_disagreements_with_ledger, write_measured_speed};
use crate::release_speed::speed_ledger::speed_ledger_tests::{ledger, measurement};

const PLACE: MeasurementPlace<'static> = MeasurementPlace {
    measured_on: "Mac with an M5 Pro and 48 GB",
    astronomical_version: "0.2.129",
};

const CATALOG_WITHOUT_SPEED: &str = r#"{
    "schema_version": 5,
    "entries": [
        {
            "huggingface_id": "example-org/Fast",
            "role": "chat"
        },
        {
            "huggingface_id": "example-org/Other",
            "role": "chat"
        }
    ]
}
"#;

fn medians(
    essay_short: u64,
    essay_long: u64,
    code_short: u64,
    code_long: u64,
) -> BTreeMap<&'static str, u64> {
    BTreeMap::from([
        ("essay_short_prompt", essay_short),
        ("essay_long_prompt", essay_long),
        ("code_short_prompt", code_short),
        ("code_long_prompt", code_long),
    ])
}

#[test]
fn should_add_a_speed_block_to_one_entry_and_leave_the_rest_unchanged() {
    let updated_text = write_measured_speed(
        CATALOG_WITHOUT_SPEED,
        "example-org/Fast",
        &medians(85, 77, 106, 87),
        &PLACE,
    )
    .expect("the entry should be updated");

    assert!(updated_text.contains(
        "            \"role\": \"chat\",\n            \"measured_speed\": {\n                \"measured_on\": \"Mac with an M5 Pro and 48 GB\",\n                \"astronomical_version\": \"0.2.129\",\n                \"tokens_per_second\": {\n                    \"essay_short_prompt\": 85,\n                    \"essay_long_prompt\": 77,\n                    \"code_short_prompt\": 106,\n                    \"code_long_prompt\": 87\n                }\n            }\n        },"
    ), "{updated_text}");
    assert!(updated_text.ends_with("\"huggingface_id\": \"example-org/Other\",\n            \"role\": \"chat\"\n        }\n    ]\n}\n"));
}

#[test]
fn should_replace_an_existing_speed_block_rather_than_add_a_second() {
    let first_text = write_measured_speed(
        CATALOG_WITHOUT_SPEED,
        "example-org/Fast",
        &medians(85, 77, 106, 87),
        &PLACE,
    )
    .expect("first write");
    let second_text = write_measured_speed(
        &first_text,
        "example-org/Fast",
        &medians(90, 80, 110, 90),
        &PLACE,
    )
    .expect("second write");

    assert_eq!(second_text.matches("\"measured_speed\"").count(), 1);
    assert!(second_text.contains("\"essay_short_prompt\": 90"));
    assert_eq!(
        write_measured_speed(
            &second_text,
            "example-org/Fast",
            &medians(85, 77, 106, 87),
            &PLACE
        )
        .expect("third write"),
        first_text,
        "writing the same figures again gives the same text"
    );
}

#[test]
fn should_refuse_an_unknown_model_or_a_measured_on_that_breaks_the_json() {
    assert!(
        write_measured_speed(
            CATALOG_WITHOUT_SPEED,
            "example-org/Missing",
            &medians(1, 1, 1, 1),
            &PLACE
        )
        .is_err()
    );

    let quoted_place = MeasurementPlace {
        measured_on: "a \"quoted\" Mac",
        astronomical_version: "0.2.129",
    };
    let updated_text = write_measured_speed(
        CATALOG_WITHOUT_SPEED,
        "example-org/Fast",
        &medians(1, 1, 1, 1),
        &quoted_place,
    )
    .expect("quotes are escaped, not refused");
    assert!(updated_text.contains(r#""measured_on": "a \"quoted\" Mac""#));
}

#[test]
fn should_reproduce_the_shipped_catalogs_speed_blocks_byte_for_byte() {
    let catalog_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../registry/download_catalog.json");
    let shipped_text =
        std::fs::read_to_string(catalog_path).expect("the shipped catalog should be readable");
    let shipped_document: serde_json::Value =
        serde_json::from_str(&shipped_text).expect("the catalog is JSON");
    let measured_entry = shipped_document["entries"]
        .as_array()
        .expect("entries")
        .iter()
        .find(|entry| entry.get("measured_speed").is_some())
        .expect("the shipped catalog has a measured entry");
    let task_speeds = &measured_entry["measured_speed"]["tokens_per_second"];
    let shipped_medians = medians(
        task_speeds["essay_short_prompt"].as_u64().expect("rate"),
        task_speeds["essay_long_prompt"].as_u64().expect("rate"),
        task_speeds["code_short_prompt"].as_u64().expect("rate"),
        task_speeds["code_long_prompt"].as_u64().expect("rate"),
    );
    let shipped_place = MeasurementPlace {
        measured_on: measured_entry["measured_speed"]["measured_on"]
            .as_str()
            .expect("text"),
        astronomical_version: measured_entry["measured_speed"]["astronomical_version"]
            .as_str()
            .expect("text"),
    };

    let rewritten_text = write_measured_speed(
        &shipped_text,
        measured_entry["huggingface_id"].as_str().expect("identity"),
        &shipped_medians,
        &shipped_place,
    )
    .expect("the shipped entry should be rewritable");

    assert_eq!(
        rewritten_text, shipped_text,
        "re-writing the same figures must not move a byte"
    );
}

fn catalog_with_fast_measured(version: &str, code_long_rate: u64) -> String {
    write_measured_speed(
        CATALOG_WITHOUT_SPEED,
        "example-org/Fast",
        &medians(70, 66, 71, code_long_rate),
        &MeasurementPlace {
            measured_on: PLACE.measured_on,
            astronomical_version: version,
        },
    )
    .expect("a catalog with a measured entry")
}

fn fast_at_own_limit(reply_median: f64) -> crate::release_speed::speed_ledger::SpeedMeasurement {
    let mut fast_measurement = measurement("Fast", "code_long_prompt", reply_median);
    fast_measurement.maximum_mlx_memory_gb = None;
    fast_measurement
}

#[test]
fn should_find_no_disagreement_when_the_catalog_holds_the_ledger_figures() {
    let newest_ledger = ledger("0.2.135", vec![fast_at_own_limit(64.6)]);

    assert_eq!(
        catalog_disagreements_with_ledger(
            &catalog_with_fast_measured("0.2.135", 65),
            &newest_ledger
        ),
        Ok(Vec::new())
    );
}

#[test]
fn should_report_a_catalog_rate_or_version_the_ledger_does_not_hold() {
    let newest_ledger = ledger("0.2.135", vec![fast_at_own_limit(64.6)]);

    let other_rate = catalog_disagreements_with_ledger(
        &catalog_with_fast_measured("0.2.135", 60),
        &newest_ledger,
    )
    .expect("a parsable catalog");
    let older_version = catalog_disagreements_with_ledger(
        &catalog_with_fast_measured("0.2.129", 65),
        &newest_ledger,
    )
    .expect("a parsable catalog");

    assert_eq!(other_rate.len(), 1);
    assert!(other_rate[0].starts_with("Fast code_long_prompt"));
    assert_eq!(older_version.len(), 1);
    assert!(catalog_disagreements_with_ledger("not json", &newest_ledger).is_err());
}

#[test]
fn should_not_hold_the_catalog_to_figures_under_a_lowered_limit_or_of_unlisted_models() {
    let newest_ledger = ledger(
        "0.2.135",
        vec![measurement("Fast", "code_long_prompt", 30.0), {
            let mut unlisted = fast_at_own_limit(10.0);
            unlisted.model_id = "Unlisted".to_owned();
            unlisted
        }],
    );

    assert_eq!(
        catalog_disagreements_with_ledger(
            &catalog_with_fast_measured("0.2.129", 65),
            &newest_ledger
        ),
        Ok(Vec::new())
    );
}

#[test]
fn should_keep_each_catalog_speed_equal_to_the_newest_ledger() {
    let repository_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let Some(newest_ledger) = crate::release_speed::speed_ledger::newest_ledger(&repository_root)
        .expect("the shipped ledgers should be readable")
    else {
        return;
    };
    let catalog_text =
        std::fs::read_to_string(repository_root.join("registry/download_catalog.json"))
            .expect("the shipped catalog should be readable");

    assert_eq!(
        catalog_disagreements_with_ledger(&catalog_text, &newest_ledger),
        Ok(Vec::new()),
        "run `cargo xtask measure-speed --reference-set --write-ledger --write-catalog` so the catalog shows the newest release's figures"
    );
}
