//! The protocol-buffer wire primitives a Core ML model document needs.
//!
//! Core ML's `.mlmodel` format is a protocol buffer. Writing the handful of
//! fields directly keeps the crate free of a protobuf code generator and of
//! Apple's Python converters, the same choice #587 made for the predictor.

const WIRE_TYPE_VARINT: u32 = 0;
const WIRE_TYPE_LENGTH_DELIMITED: u32 = 2;

pub fn append_varint(document: &mut Vec<u8>, mut unsigned_value: u64) {
    while unsigned_value >= 0x80 {
        document.push((unsigned_value as u8) | 0x80);
        unsigned_value >>= 7;
    }
    document.push(unsigned_value as u8);
}

fn append_field_key(document: &mut Vec<u8>, field_number: u32, wire_type: u32) {
    append_varint(document, u64::from((field_number << 3) | wire_type));
}

pub fn append_varint_field(document: &mut Vec<u8>, field_number: u32, unsigned_value: u64) {
    append_field_key(document, field_number, WIRE_TYPE_VARINT);
    append_varint(document, unsigned_value);
}

pub fn append_bytes_field(document: &mut Vec<u8>, field_number: u32, field_bytes: &[u8]) {
    append_field_key(document, field_number, WIRE_TYPE_LENGTH_DELIMITED);
    append_varint(document, field_bytes.len() as u64);
    document.extend_from_slice(field_bytes);
}

pub fn append_string_field(document: &mut Vec<u8>, field_number: u32, text: &str) {
    append_bytes_field(document, field_number, text.as_bytes());
}

#[cfg(test)]
#[path = "protobuf_wire_tests.rs"]
mod tests;
