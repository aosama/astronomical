//! Removing local paths from the documents a converter copies.
//!
//! Checkpoints carry configs written on the machine that trained them, and
//! those often record where files were on that machine. A converted model is
//! shared, so nothing that names a place on a disk may reach its files.

use serde_json::{Map, Value};

/// Hugging Face records where a model was loaded from under this key.
const SOURCE_LOCATION_KEY: &str = "_name_or_path";

/// `document` with every source-location key and every string that looks
/// like a local path removed, at any depth.
pub(crate) fn without_local_paths(document: Value) -> Value {
    match document {
        Value::Object(members) => {
            let mut kept_members = Map::new();
            for (member_name, member_value) in members {
                if member_name == SOURCE_LOCATION_KEY || is_local_path_text(&member_value) {
                    continue;
                }
                kept_members.insert(member_name, without_local_paths(member_value));
            }
            Value::Object(kept_members)
        }
        Value::Array(elements) => {
            let mut kept_elements = Vec::with_capacity(elements.len());
            for element in elements {
                if !is_local_path_text(&element) {
                    kept_elements.push(without_local_paths(element));
                }
            }
            Value::Array(kept_elements)
        }
        other_value => other_value,
    }
}

/// Whether `value` is text that names a place on a local disk: an absolute
/// path, a home-relative path, a relative path that climbs or starts at the
/// current folder, or a Windows drive path.
fn is_local_path_text(value: &Value) -> bool {
    let Some(text) = value.as_str() else {
        return false;
    };
    let starts_like_a_path = ["/", "~/", "./", "../", "file:"]
        .iter()
        .any(|path_start| text.starts_with(path_start));
    let text_bytes = text.as_bytes();
    let is_windows_drive_path = text_bytes.len() > 2
        && text_bytes[0].is_ascii_alphabetic()
        && text_bytes[1] == b':'
        && (text_bytes[2] == b'\\' || text_bytes[2] == b'/');
    starts_like_a_path || is_windows_drive_path
}

#[cfg(test)]
#[path = "local_paths_tests.rs"]
mod tests;
