use std::fmt;

use serde::de::{Error as _, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Number, Value};

pub(crate) const DUPLICATE_JSON_FIELD_MARKER: &str = "duplicate JSON object field";
const MAXIMUM_DUPLICATE_FIELD_CHARACTERS: usize = 256;

/// Recursive JSON value that rejects a repeated object key before replacement can occur.
pub(crate) struct DuplicateAwareJsonValue(pub(crate) Value);

impl<'de> Deserialize<'de> for DuplicateAwareJsonValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(DuplicateAwareJsonValueVisitor)
    }
}

struct DuplicateAwareJsonValueVisitor;

impl<'de> Visitor<'de> for DuplicateAwareJsonValueVisitor {
    type Value = DuplicateAwareJsonValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("one JSON value whose object fields are unique")
    }

    fn visit_bool<E>(self, boolean_value: bool) -> Result<Self::Value, E> {
        Ok(DuplicateAwareJsonValue(Value::Bool(boolean_value)))
    }

    fn visit_i64<E>(self, signed_value: i64) -> Result<Self::Value, E> {
        Ok(DuplicateAwareJsonValue(Value::Number(Number::from(
            signed_value,
        ))))
    }

    fn visit_u64<E>(self, unsigned_value: u64) -> Result<Self::Value, E> {
        Ok(DuplicateAwareJsonValue(Value::Number(Number::from(
            unsigned_value,
        ))))
    }

    fn visit_f64<E>(self, float_value: f64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        let json_number = Number::from_f64(float_value)
            .ok_or_else(|| E::custom("JSON numbers must be finite"))?;
        Ok(DuplicateAwareJsonValue(Value::Number(json_number)))
    }

    fn visit_str<E>(self, string_value: &str) -> Result<Self::Value, E> {
        Ok(DuplicateAwareJsonValue(Value::String(
            string_value.to_owned(),
        )))
    }

    fn visit_string<E>(self, string_value: String) -> Result<Self::Value, E> {
        Ok(DuplicateAwareJsonValue(Value::String(string_value)))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(DuplicateAwareJsonValue(Value::Null))
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(DuplicateAwareJsonValue(Value::Null))
    }

    fn visit_seq<A>(self, mut sequence_values: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut unique_sequence_values = Vec::new();
        while let Some(sequence_value) =
            sequence_values.next_element::<DuplicateAwareJsonValue>()?
        {
            unique_sequence_values.push(sequence_value.0);
        }
        Ok(DuplicateAwareJsonValue(Value::Array(
            unique_sequence_values,
        )))
    }

    fn visit_map<A>(self, mut object_fields: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut unique_object_fields = Map::new();
        while let Some((field_name, field_value)) =
            object_fields.next_entry::<String, DuplicateAwareJsonValue>()?
        {
            if unique_object_fields.contains_key(&field_name) {
                return Err(A::Error::custom(format!(
                    "{DUPLICATE_JSON_FIELD_MARKER} '{}'",
                    bounded_field_name(&field_name)
                )));
            }
            unique_object_fields.insert(field_name, field_value.0);
        }
        Ok(DuplicateAwareJsonValue(Value::Object(unique_object_fields)))
    }
}

fn bounded_field_name(unbounded_field_name: &str) -> String {
    unbounded_field_name
        .chars()
        .take(MAXIMUM_DUPLICATE_FIELD_CHARACTERS)
        .collect()
}
