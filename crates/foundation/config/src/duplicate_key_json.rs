//! Performs bounded recursive JSON parsing before serde DTO deserialization.

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

use serde::de::{DeserializeSeed, Error as _, MapAccess, SeqAccess, Visitor};

use crate::AstronomicalConfigError;

const DUPLICATE_KEY_MARKER: &str = "astronomical-duplicate-key:";
const MAXIMUM_JSON_DEPTH: usize = 128;

pub(crate) fn parse_json_rejecting_duplicates(
    config_file_path: &Path,
    config_file_bytes: &[u8],
) -> Result<serde_json::Value, AstronomicalConfigError> {
    let mut deserializer = serde_json::Deserializer::from_slice(config_file_bytes);
    let parsed_json = JsonValueSeed { depth: 0 }
        .deserialize(&mut deserializer)
        .and_then(|parsed_json| {
            deserializer.end()?;
            Ok(parsed_json)
        });
    match parsed_json {
        Ok(parsed_json) => Ok(parsed_json),
        Err(source) => {
            let error_text = source.to_string();
            if let Some(marker_position) = error_text.find(DUPLICATE_KEY_MARKER) {
                let duplicate_key = error_text[marker_position + DUPLICATE_KEY_MARKER.len()..]
                    .split(" at line")
                    .next()
                    .unwrap_or("unknown")
                    .to_owned();
                return Err(AstronomicalConfigError::DuplicateConfigKey {
                    config_file_path: config_file_path.to_owned(),
                    duplicate_key,
                });
            }
            Err(AstronomicalConfigError::ParseConfigFile {
                config_file_path: config_file_path.to_owned(),
                source,
            })
        }
    }
}

struct JsonValueSeed {
    depth: usize,
}

impl<'de> DeserializeSeed<'de> for JsonValueSeed {
    type Value = serde_json::Value;

    fn deserialize<Deserializer>(
        self,
        deserializer: Deserializer,
    ) -> Result<Self::Value, Deserializer::Error>
    where
        Deserializer: serde::Deserializer<'de>,
    {
        if self.depth > MAXIMUM_JSON_DEPTH {
            return Err(Deserializer::Error::custom(
                "configuration JSON exceeds the maximum nesting depth",
            ));
        }
        deserializer.deserialize_any(JsonValueVisitor { depth: self.depth })
    }
}

struct JsonValueVisitor {
    depth: usize,
}

impl<'de> Visitor<'de> for JsonValueVisitor {
    type Value = serde_json::Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(value.into())
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(value.into())
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(value.into())
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E> {
        Ok(value.into())
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(value.into())
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(value.into())
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(serde_json::Value::Null)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(serde_json::Value::Null)
    }

    fn visit_seq<Sequence>(self, mut sequence: Sequence) -> Result<Self::Value, Sequence::Error>
    where
        Sequence: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element_seed(JsonValueSeed {
            depth: self.depth + 1,
        })? {
            values.push(value);
        }
        Ok(serde_json::Value::Array(values))
    }

    fn visit_map<Object>(self, mut object: Object) -> Result<Self::Value, Object::Error>
    where
        Object: MapAccess<'de>,
    {
        let mut properties = BTreeMap::new();
        while let Some(property_name) = object.next_key::<String>()? {
            if properties.contains_key(&property_name) {
                return Err(Object::Error::custom(format!(
                    "{DUPLICATE_KEY_MARKER}{property_name}"
                )));
            }
            let property_value = object.next_value_seed(JsonValueSeed {
                depth: self.depth + 1,
            })?;
            properties.insert(property_name, property_value);
        }
        Ok(serde_json::Value::Object(properties.into_iter().collect()))
    }
}
