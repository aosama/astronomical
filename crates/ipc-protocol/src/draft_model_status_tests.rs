use super::{DraftModelGeneration, DraftModelLocation, DraftModelStatus};

#[test]
fn should_serialize_a_drafting_model_with_its_generation_location_and_tree_size() {
    let drafting_status = DraftModelStatus::Drafting {
        generation: DraftModelGeneration::Dflash,
        location: DraftModelLocation::Bundled,
        tree_node_budget: 4,
    };
    let status_json = serde_json::to_value(&drafting_status).expect("status serializes");
    assert_eq!(
        status_json,
        serde_json::json!({
            "state": "drafting",
            "generation": "dflash",
            "location": "bundled",
            "tree_node_budget": 4
        })
    );
    assert!(drafting_status.is_drafting());
}

#[test]
fn should_round_trip_every_state() {
    for draft_model_status in [
        DraftModelStatus::None,
        DraftModelStatus::Drafting {
            generation: DraftModelGeneration::Dflash2,
            location: DraftModelLocation::Library,
            tree_node_budget: 0,
        },
        DraftModelStatus::Unavailable {
            reason: String::new(),
        },
    ] {
        let encoded_status = serde_json::to_string(&draft_model_status).expect("serializes");
        let decoded_status: DraftModelStatus =
            serde_json::from_str(&encoded_status).expect("deserializes");
        assert_eq!(decoded_status, draft_model_status);
    }
}

#[test]
fn should_not_count_an_unavailable_or_absent_drafter_as_drafting() {
    assert!(!DraftModelStatus::None.is_drafting());
    assert!(
        !DraftModelStatus::Unavailable {
            reason: "the model is streaming from the SSD".to_owned()
        }
        .is_drafting()
    );
}

#[test]
fn should_reject_an_unknown_state_or_field() {
    assert!(serde_json::from_str::<DraftModelStatus>(r#"{"state":"warming"}"#).is_err());
    assert!(
        serde_json::from_str::<DraftModelStatus>(
            r#"{"state":"unavailable","reason":"x","drafter_path":"/x"}"#
        )
        .is_err()
    );
}
