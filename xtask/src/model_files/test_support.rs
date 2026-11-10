//! Builders for the files the model-file readers parse, shared by their
//! tests and by the converters' tests: zip archives as `torch.save` lays
//! them out, and pickles assembled opcode by opcode.

/// One archive entry for `stored_zip_bytes`: its name, its bytes, and the
/// compression method to declare.
pub(crate) struct TestEntry<'a> {
    pub(crate) entry_name: &'a str,
    pub(crate) entry_bytes: &'a [u8],
    pub(crate) compression_method: u16,
}

/// A zip archive holding `entries`, laid out as `torch.save` lays one out:
/// local headers and data, then the central directory, then the end record.
pub(crate) fn stored_zip_bytes(entries: &[TestEntry<'_>]) -> Vec<u8> {
    let mut archive_bytes = Vec::new();
    let mut local_header_offsets = Vec::new();
    for entry in entries {
        local_header_offsets.push(archive_bytes.len() as u32);
        archive_bytes.extend(0x0403_4B50_u32.to_le_bytes());
        archive_bytes.extend([0_u8; 4]);
        archive_bytes.extend(entry.compression_method.to_le_bytes());
        archive_bytes.extend([0_u8; 8]);
        archive_bytes.extend((entry.entry_bytes.len() as u32).to_le_bytes());
        archive_bytes.extend((entry.entry_bytes.len() as u32).to_le_bytes());
        archive_bytes.extend((entry.entry_name.len() as u16).to_le_bytes());
        // A local extra field the central directory does not repeat, as
        // PyTorch writes to align tensor data.
        archive_bytes.extend(3_u16.to_le_bytes());
        archive_bytes.extend(entry.entry_name.as_bytes());
        archive_bytes.extend([0xAA_u8; 3]);
        archive_bytes.extend(entry.entry_bytes);
    }
    let directory_start = archive_bytes.len() as u32;
    for (entry, local_header_offset) in entries.iter().zip(&local_header_offsets) {
        archive_bytes.extend(0x0201_4B50_u32.to_le_bytes());
        archive_bytes.extend([0_u8; 6]);
        archive_bytes.extend(entry.compression_method.to_le_bytes());
        archive_bytes.extend([0_u8; 8]);
        archive_bytes.extend((entry.entry_bytes.len() as u32).to_le_bytes());
        archive_bytes.extend((entry.entry_bytes.len() as u32).to_le_bytes());
        archive_bytes.extend((entry.entry_name.len() as u16).to_le_bytes());
        archive_bytes.extend([0_u8; 12]);
        archive_bytes.extend(local_header_offset.to_le_bytes());
        archive_bytes.extend(entry.entry_name.as_bytes());
    }
    let directory_length = archive_bytes.len() as u32 - directory_start;
    archive_bytes.extend(0x0605_4B50_u32.to_le_bytes());
    archive_bytes.extend([0_u8; 4]);
    archive_bytes.extend((entries.len() as u16).to_le_bytes());
    archive_bytes.extend((entries.len() as u16).to_le_bytes());
    archive_bytes.extend(directory_length.to_le_bytes());
    archive_bytes.extend(directory_start.to_le_bytes());
    archive_bytes.extend([0_u8; 2]);
    archive_bytes
}

/// Assembles pickle bytes one opcode at a time.
#[derive(Default)]
pub(crate) struct PickleBuilder {
    pickle_bytes: Vec<u8>,
}

impl PickleBuilder {
    pub(crate) fn opcode(mut self, opcode: u8) -> Self {
        self.pickle_bytes.push(opcode);
        self
    }

    pub(crate) fn text(mut self, text: &str) -> Self {
        self.pickle_bytes.push(b'X');
        self.pickle_bytes.extend((text.len() as u32).to_le_bytes());
        self.pickle_bytes.extend(text.as_bytes());
        self
    }

    pub(crate) fn small_integer(mut self, integer: u8) -> Self {
        self.pickle_bytes.extend([b'K', integer]);
        self
    }

    pub(crate) fn medium_integer(mut self, integer: u16) -> Self {
        self.pickle_bytes.push(b'M');
        self.pickle_bytes.extend(integer.to_le_bytes());
        self
    }

    pub(crate) fn integer(mut self, integer: i32) -> Self {
        self.pickle_bytes.push(b'J');
        self.pickle_bytes.extend(integer.to_le_bytes());
        self
    }

    pub(crate) fn float(mut self, float: f64) -> Self {
        self.pickle_bytes.push(b'G');
        self.pickle_bytes.extend(float.to_be_bytes());
        self
    }

    pub(crate) fn global(mut self, module_name: &str, attribute_name: &str) -> Self {
        self.pickle_bytes.push(b'c');
        self.pickle_bytes
            .extend(format!("{module_name}\n{attribute_name}\n").as_bytes());
        self
    }

    pub(crate) fn memo_put(mut self, memo_key: u8) -> Self {
        self.pickle_bytes.extend([b'q', memo_key]);
        self
    }

    pub(crate) fn memo_get(mut self, memo_key: u8) -> Self {
        self.pickle_bytes.extend([b'h', memo_key]);
        self
    }

    /// `_rebuild_tensor_v2(storage, 0, (rows, columns), (columns, 1), False, OrderedDict())`
    /// for a float32 storage under `storage_key`, as `torch.save` writes it.
    pub(crate) fn float32_matrix(
        self,
        storage_key: &str,
        row_count: u16,
        column_count: u16,
    ) -> Self {
        self.global("torch._utils", "_rebuild_tensor_v2")
            .opcode(b'(')
            .opcode(b'(')
            .text("storage")
            .global("torch", "FloatStorage")
            .text(storage_key)
            .text("cpu")
            .integer(i32::from(row_count) * i32::from(column_count))
            .opcode(b't')
            .opcode(b'Q')
            .small_integer(0)
            .medium_integer(row_count)
            .medium_integer(column_count)
            .opcode(0x86)
            .medium_integer(column_count)
            .small_integer(1)
            .opcode(0x86)
            .opcode(0x89)
            .global("collections", "OrderedDict")
            .opcode(b')')
            .opcode(b'R')
            .opcode(b't')
            .opcode(b'R')
    }

    /// `_rebuild_tensor_v2(storage, 0, (length,), (1,), False, OrderedDict())`.
    pub(crate) fn float32_vector(self, storage_key: &str, length: u16) -> Self {
        self.global("torch._utils", "_rebuild_tensor_v2")
            .opcode(b'(')
            .opcode(b'(')
            .text("storage")
            .global("torch", "FloatStorage")
            .text(storage_key)
            .text("cpu")
            .integer(i32::from(length))
            .opcode(b't')
            .opcode(b'Q')
            .small_integer(0)
            .medium_integer(length)
            .opcode(0x85)
            .small_integer(1)
            .opcode(0x85)
            .opcode(0x89)
            .global("collections", "OrderedDict")
            .opcode(b')')
            .opcode(b'R')
            .opcode(b't')
            .opcode(b'R')
    }

    pub(crate) fn stop(self) -> Vec<u8> {
        let mut pickle_bytes = self.opcode(b'.').pickle_bytes;
        pickle_bytes.insert(0, 2);
        pickle_bytes.insert(0, 0x80);
        pickle_bytes
    }
}

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
