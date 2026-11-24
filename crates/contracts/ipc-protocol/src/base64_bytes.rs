//! Serde support for binary payloads (image files, speech audio) that must
//! remain compact on the JSON IPC wire.

use base64::prelude::{BASE64_STANDARD, Engine as _};
use serde::{Deserialize, Deserializer, Serializer};

pub fn serialize<SerializerType>(
    payload_bytes: &[u8],
    serializer: SerializerType,
) -> Result<SerializerType::Ok, SerializerType::Error>
where
    SerializerType: Serializer,
{
    serializer.serialize_str(&BASE64_STANDARD.encode(payload_bytes))
}

pub fn deserialize<'de, DeserializerType>(
    deserializer: DeserializerType,
) -> Result<Vec<u8>, DeserializerType::Error>
where
    DeserializerType: Deserializer<'de>,
{
    let base64_payload = String::deserialize(deserializer)?;
    BASE64_STANDARD
        .decode(base64_payload)
        .map_err(serde::de::Error::custom)
}
