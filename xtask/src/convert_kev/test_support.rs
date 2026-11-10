//! Builders for Kev's own checkpoint files in the converter's tests: a
//! `head.pt` archive as `torch.save` writes Kev's, from the shared zip and
//! pickle builders.

use crate::model_files::test_support::{PickleBuilder, TestEntry, stored_zip_bytes};

/// The settings a test `head.pt` carries beside its four tensors.
pub(crate) struct TestHeadSettings<'a> {
    pub(crate) pointer_dimension: u16,
    pub(crate) hidden_size: u16,
    pub(crate) temperature: f64,
    pub(crate) base_revision: &'a str,
    pub(crate) option_isolation: bool,
}

/// A `head.pt` archive as `torch.save` writes Kev's: the head's four float32
/// tensors filled with `0.5, 1.5, 2.5, …` in each tensor, and its settings.
pub(crate) fn kev_head_archive_bytes(head_settings: &TestHeadSettings<'_>) -> Vec<u8> {
    let pointer_dimension = head_settings.pointer_dimension;
    let pickle_bytes = PickleBuilder::default()
        .opcode(b'}')
        .opcode(b'(')
        .text("head")
        .global("collections", "OrderedDict")
        .opcode(b')')
        .opcode(b'R')
        .opcode(b'(')
        .text("q.weight")
        .float32_matrix("0", pointer_dimension, head_settings.hidden_size)
        .text("q.bias")
        .float32_vector("1", pointer_dimension)
        .text("k.weight")
        .float32_matrix("2", pointer_dimension, head_settings.hidden_size)
        .text("k.bias")
        .float32_vector("3", pointer_dimension)
        .opcode(b'u')
        .text("base")
        .text("example-org/example-base")
        .text("base_revision")
        .text(head_settings.base_revision)
        .text("temperature")
        .float(head_settings.temperature)
        .text("option_isolation")
        .opcode(if head_settings.option_isolation {
            0x88
        } else {
            0x89
        })
        .opcode(b'u')
        .stop();
    let matrix_bytes = ascending_float32_bytes(
        usize::from(pointer_dimension) * usize::from(head_settings.hidden_size),
    );
    let vector_bytes = ascending_float32_bytes(usize::from(pointer_dimension));
    stored_zip_bytes(&[
        TestEntry {
            entry_name: "head/data.pkl",
            entry_bytes: &pickle_bytes,
            compression_method: 0,
        },
        TestEntry {
            entry_name: "head/byteorder",
            entry_bytes: b"little",
            compression_method: 0,
        },
        TestEntry {
            entry_name: "head/data/0",
            entry_bytes: &matrix_bytes,
            compression_method: 0,
        },
        TestEntry {
            entry_name: "head/data/1",
            entry_bytes: &vector_bytes,
            compression_method: 0,
        },
        TestEntry {
            entry_name: "head/data/2",
            entry_bytes: &matrix_bytes,
            compression_method: 0,
        },
        TestEntry {
            entry_name: "head/data/3",
            entry_bytes: &vector_bytes,
            compression_method: 0,
        },
    ])
}

/// `0.5, 1.5, 2.5, …` as little-endian float32 bytes.
pub(crate) fn ascending_float32_bytes(value_count: usize) -> Vec<u8> {
    let mut value_bytes = Vec::with_capacity(value_count * 4);
    for value_index in 0..value_count {
        value_bytes.extend((value_index as f32 + 0.5).to_le_bytes());
    }
    value_bytes
}
