use super::{
    MeasurementCase, MemoryLimitRequest, reference_cases_for, refuse_unless_served_as_requested,
};
use crate::measure_speed::reference_set::ReferenceCase;

fn reference_case(model_id: &str, maximum_mlx_memory_gb: Option<u64>) -> ReferenceCase {
    ReferenceCase {
        model_id: model_id.to_owned(),
        maximum_mlx_memory_gb,
        long_prompt_tokens: (model_id == "Model-B").then_some(2_000),
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
                long_prompt_tokens: 10_000,
            },
            MeasurementCase {
                model_id: "Model-A".to_owned(),
                memory_limit_request: MemoryLimitRequest::ServeUnder(Some(16)),
                long_prompt_tokens: 10_000,
            },
            MeasurementCase {
                model_id: "Model-B".to_owned(),
                memory_limit_request: MemoryLimitRequest::ServeUnder(Some(34)),
                long_prompt_tokens: 2_000,
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

#[test]
fn should_accept_only_the_limit_a_reference_case_asked_for() {
    assert_eq!(
        refuse_unless_served_as_requested(MemoryLimitRequest::ServeUnder(Some(16)), Some(16)),
        Ok(())
    );
    assert_eq!(
        refuse_unless_served_as_requested(MemoryLimitRequest::ServeUnder(None), None),
        Ok(())
    );
    assert_eq!(
        refuse_unless_served_as_requested(MemoryLimitRequest::KeepEngineLimit, Some(20)),
        Ok(()),
        "a named model keeps whatever limit the engine has"
    );
    assert!(
        refuse_unless_served_as_requested(MemoryLimitRequest::ServeUnder(Some(34)), Some(32))
            .is_err()
    );
    assert!(
        refuse_unless_served_as_requested(MemoryLimitRequest::ServeUnder(None), Some(16)).is_err()
    );
}
