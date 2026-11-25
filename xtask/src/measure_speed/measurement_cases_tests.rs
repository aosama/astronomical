use super::{MeasurementCase, MemoryLimitRequest, reference_cases_for};
use crate::measure_speed::reference_set::ReferenceCase;

fn reference_case(model_id: &str, maximum_mlx_memory_gb: Option<u64>) -> ReferenceCase {
    ReferenceCase {
        model_id: model_id.to_owned(),
        maximum_mlx_memory_gb,
        why: "a reference case".to_owned(),
    }
}

fn reference_set() -> Vec<ReferenceCase> {
    vec![
        reference_case("Model-A", None),
        reference_case("Model-A", Some(16)),
        reference_case("Model-B", Some(34)),
    ]
}

#[test]
fn should_measure_every_reference_case_under_its_own_limit_when_none_is_named() {
    let chosen_cases = reference_cases_for(reference_set(), &[]).expect("the whole set");

    assert_eq!(
        chosen_cases,
        vec![
            MeasurementCase {
                model_id: "Model-A".to_owned(),
                memory_limit_request: MemoryLimitRequest::ServeUnder(None),
            },
            MeasurementCase {
                model_id: "Model-A".to_owned(),
                memory_limit_request: MemoryLimitRequest::ServeUnder(Some(16)),
            },
            MeasurementCase {
                model_id: "Model-B".to_owned(),
                memory_limit_request: MemoryLimitRequest::ServeUnder(Some(34)),
            },
        ]
    );
}

#[test]
fn should_narrow_the_set_to_every_case_of_a_named_model() {
    let chosen_cases =
        reference_cases_for(reference_set(), &["Model-A".to_owned()]).expect("Model-A's cases");

    assert_eq!(chosen_cases.len(), 2);
    assert!(
        chosen_cases
            .iter()
            .all(|chosen_case| chosen_case.model_id == "Model-A")
    );
}

#[test]
fn should_refuse_a_named_model_the_set_does_not_hold() {
    assert!(
        reference_cases_for(
            reference_set(),
            &["Model-A".to_owned(), "Model-C".to_owned()]
        )
        .is_err()
    );
}
