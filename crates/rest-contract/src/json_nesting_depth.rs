//! The one JSON nesting-depth measure every schema bound uses: tool
//! parameters on all three generation surfaces and structured-output
//! schemas. One definition keeps those bounds from drifting apart.

use serde_json::Value;

/// Counts nested arrays and objects; a scalar has depth 0 and `{}` depth 1.
pub(crate) fn json_nesting_depth(json_value: &Value) -> usize {
    let mut deepest_child_depth = 0;
    match json_value {
        Value::Array(array_values) => {
            for array_value in array_values {
                deepest_child_depth = deepest_child_depth.max(json_nesting_depth(array_value));
            }
        }
        Value::Object(object_values) => {
            for object_value in object_values.values() {
                deepest_child_depth = deepest_child_depth.max(json_nesting_depth(object_value));
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => return 0,
    }
    deepest_child_depth.saturating_add(1)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::json_nesting_depth;

    #[test]
    fn should_give_scalars_depth_zero() {
        assert_eq!(json_nesting_depth(&json!(18)), 0);
        assert_eq!(json_nesting_depth(&json!(null)), 0);
    }

    #[test]
    fn should_give_empty_containers_depth_one() {
        assert_eq!(json_nesting_depth(&json!({})), 1);
        assert_eq!(json_nesting_depth(&json!([])), 1);
    }

    #[test]
    fn should_follow_the_deepest_branch() {
        let schema = json!({"shallow": 1, "deep": {"deeper": [{"deepest": true}]}});

        assert_eq!(json_nesting_depth(&schema), 4);
    }
}
