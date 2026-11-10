//! A JSON value that remembers the order its object members arrived in.
//!
//! Decision models read a request's state and option lists as text, so member
//! order is part of the model input: `{"a": 1, "b": 2}` and `{"b": 2, "a": 1}`
//! tokenize differently and can answer differently. `serde_json::Value` sorts
//! object keys unless its `preserve_order` feature is on, and that feature is
//! unified across the whole workspace by Cargo, so turning it on here would
//! silently change every other crate's maps. This type keeps the order with
//! no feature switch.

use std::fmt;

use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};

/// One JSON value whose objects keep their members in arrival order.
#[derive(Clone, Debug, PartialEq)]
pub enum OrderedJsonValue {
    Null,
    Boolean(bool),
    /// Every integer that fits a signed 64-bit value, positive or negative.
    SignedInteger(i64),
    /// Only integers above `i64::MAX`; smaller ones are `SignedInteger` so a
    /// number has exactly one representation.
    UnsignedInteger(u64),
    Float(f64),
    Text(String),
    Array(Vec<OrderedJsonValue>),
    Object(Vec<OrderedJsonMember>),
}

/// One `key: value` pair of an ordered JSON object.
#[derive(Clone, Debug, PartialEq)]
pub struct OrderedJsonMember {
    pub key: String,
    pub value: OrderedJsonValue,
}

impl OrderedJsonValue {
    /// The text of a JSON string, or `None` for every other kind of value.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            _ => None,
        }
    }

    /// The elements of a JSON array, or `None` for every other kind of value.
    #[must_use]
    pub fn as_array(&self) -> Option<&[OrderedJsonValue]> {
        match self {
            Self::Array(elements) => Some(elements),
            _ => None,
        }
    }

    /// The members of a JSON object, or `None` for every other kind of value.
    #[must_use]
    pub fn as_object(&self) -> Option<&[OrderedJsonMember]> {
        match self {
            Self::Object(members) => Some(members),
            _ => None,
        }
    }

    /// The value stored under `key` in a JSON object, if this is an object that has it.
    #[must_use]
    pub fn member(&self, key: &str) -> Option<&OrderedJsonValue> {
        let members = self.as_object()?;
        for object_member in members {
            if object_member.key == key {
                return Some(&object_member.value);
            }
        }
        None
    }

    /// The JSON type name used in error messages.
    #[must_use]
    pub const fn type_name(&self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Boolean(_) => "boolean",
            Self::SignedInteger(_) | Self::UnsignedInteger(_) | Self::Float(_) => "number",
            Self::Text(_) => "string",
            Self::Array(_) => "array",
            Self::Object(_) => "object",
        }
    }
}

/// Writes the value back as JSON with every object's members in their kept order.
///
/// This is what lets a request cross a process boundary and still be read by
/// the model as the caller wrote it.
impl Serialize for OrderedJsonValue {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Null => serializer.serialize_unit(),
            Self::Boolean(boolean) => serializer.serialize_bool(*boolean),
            Self::SignedInteger(integer) => serializer.serialize_i64(*integer),
            Self::UnsignedInteger(integer) => serializer.serialize_u64(*integer),
            Self::Float(float) => serializer.serialize_f64(*float),
            Self::Text(text) => serializer.serialize_str(text),
            Self::Array(elements) => {
                let mut sequence = serializer.serialize_seq(Some(elements.len()))?;
                for element in elements {
                    sequence.serialize_element(element)?;
                }
                sequence.end()
            }
            Self::Object(members) => {
                let mut map = serializer.serialize_map(Some(members.len()))?;
                for object_member in members {
                    map.serialize_entry(&object_member.key, &object_member.value)?;
                }
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for OrderedJsonValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(OrderedJsonVisitor)
    }
}

struct OrderedJsonVisitor;

impl<'de> Visitor<'de> for OrderedJsonVisitor {
    type Value = OrderedJsonValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("any JSON value")
    }

    fn visit_bool<E>(self, boolean: bool) -> Result<Self::Value, E> {
        Ok(OrderedJsonValue::Boolean(boolean))
    }

    fn visit_i64<E>(self, integer: i64) -> Result<Self::Value, E> {
        Ok(OrderedJsonValue::SignedInteger(integer))
    }

    fn visit_u64<E>(self, integer: u64) -> Result<Self::Value, E> {
        match i64::try_from(integer) {
            Ok(signed_integer) => Ok(OrderedJsonValue::SignedInteger(signed_integer)),
            Err(_) => Ok(OrderedJsonValue::UnsignedInteger(integer)),
        }
    }

    fn visit_f64<E>(self, float: f64) -> Result<Self::Value, E> {
        Ok(OrderedJsonValue::Float(float))
    }

    fn visit_str<E>(self, text: &str) -> Result<Self::Value, E> {
        Ok(OrderedJsonValue::Text(text.to_owned()))
    }

    fn visit_string<E>(self, text: String) -> Result<Self::Value, E> {
        Ok(OrderedJsonValue::Text(text))
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(OrderedJsonValue::Null)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(OrderedJsonValue::Null)
    }

    fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_any(Self)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
        let mut elements = Vec::with_capacity(sequence.size_hint().unwrap_or(0).min(1_024));
        while let Some(element) = sequence.next_element::<OrderedJsonValue>()? {
            elements.push(element);
        }
        Ok(OrderedJsonValue::Array(elements))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut members: Vec<OrderedJsonMember> = Vec::new();
        while let Some((key, value)) = map.next_entry::<String, OrderedJsonValue>()? {
            insert_member_like_a_python_dict(&mut members, key, value);
        }
        Ok(OrderedJsonValue::Object(members))
    }
}

/// Adds one member, giving a repeated key the reference implementations' meaning.
///
/// The reference decision models parse requests with Python's `json.loads`,
/// where a repeated key keeps its first position and takes its last value.
/// Matching that keeps the text a model reads identical for the rare request
/// that repeats a key.
fn insert_member_like_a_python_dict(
    members: &mut Vec<OrderedJsonMember>,
    key: String,
    value: OrderedJsonValue,
) {
    for existing_member in members.iter_mut() {
        if existing_member.key == key {
            existing_member.value = value;
            return;
        }
    }
    members.push(OrderedJsonMember { key, value });
}

#[cfg(test)]
#[path = "ordered_json_tests.rs"]
mod tests;
